"""``MorphModel``: bind components to an image, resolve priors, render, sample.

This is the seam where data-independent declarations meet the data. Building a
model resolves every omitted parameter into a concrete prior (via the
:class:`~noobfriend.inference.morphology._priors.PriorPolicy`, seeded from the
image), materialises a per-band flux for each component, precomputes the static
render geometry / PSF / fit-mask per band, and assembles a numpyro model whose
likelihood is a robust Student-t over the unmasked pixels.

:meth:`MorphModel.preview` renders the prior-mean scene (per component and summed)
for a pre-sampling sanity check; :meth:`MorphModel.sample` runs NUTS and returns a
:class:`~noobfriend.inference.morphology._result.MorphResult`. This module is
internal; ``MorphModel`` is reached via :meth:`NoobImage.model`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from noobfriend.inference.morphology._geometry import build_geometry
from noobfriend.inference.morphology._param import (
    LogUniform,
    Normal,
    Param,
    ParamNode,
    Prior,
    TruncNormal,
    Uniform,
    collect_params,
    parse_spec,
)
from noobfriend.inference.morphology._priors import (
    HighzPriorPolicy,
    compute_seeds,
)
from noobfriend.inference.morphology._profile import (
    Exponential,
    PointSource,
    Profile,
    Sersic,
)
from noobfriend.inference.morphology._psf import resolve_psf
from noobfriend.inference.morphology._render import (
    convolve_and_bin,
    render_point,
    render_sersic,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from astropy.cosmology import FLRW

    from noobfriend.inference.morphology._image import NoobImage
    from noobfriend.inference.morphology._priors import PriorPolicy
    from noobfriend.inference.morphology._result import MorphResult


@dataclass(frozen=True)
class _BandStatic:
    """Precomputed static render inputs for one band."""

    name: str
    e_grid: jax.Array
    n_grid: jax.Array
    col_grid: jax.Array
    row_grid: jax.Array
    affine: jax.Array
    pix_center: tuple[float, float]
    oversample: int
    subpixel_area: float
    oversampled_shape: tuple[int, int]
    psf: jax.Array
    obs: jax.Array
    obs_err: jax.Array
    use_idx: np.ndarray
    bg_scale: float
    lik_scale: float  # log-likelihood weight (1/n_corr) for correlated noise


@dataclass(frozen=True)
class _CompRender:
    """A component reduced to what the renderer needs."""

    name: str
    kind: str  # "sersic" | "exponential" | "point"
    geom: dict[str, ParamNode]
    flux_by_band: dict[str, ParamNode]


@dataclass(frozen=True)
class Preview:
    """A rendered prior-mean (or given-value) scene for inspection.

    Attributes
    ----------
    bands : list[str]
        Band order.
    data, total, residual : dict[str, numpy.ndarray]
        Per-band observed image, summed model, and ``data - model``.
    components : dict[str, dict[str, numpy.ndarray]]
        Per-band, per-component model image.
    use_mask : dict[str, numpy.ndarray]
        Per-band boolean mask, ``True`` where the pixel enters the likelihood.
    """

    bands: list[str]
    data: dict[str, np.ndarray]
    total: dict[str, np.ndarray]
    residual: dict[str, np.ndarray]
    components: dict[str, dict[str, np.ndarray]]
    use_mask: dict[str, np.ndarray]


class MorphModel:
    """A built spatial-decomposition model: resolved priors, renderer, likelihood.

    Construct via :meth:`build` (or :meth:`NoobImage.model`). Treated as immutable.

    Attributes
    ----------
    image : NoobImage
        The imaging being fit.
    components : tuple[Profile, ...]
        The declared profile components.
    nu : float or None
        Student-t degrees of freedom for the likelihood; ``None`` for Gaussian.
    background : bool
        Whether a per-band constant background is fit.
    cosmology : astropy.cosmology.FLRW or None
        Cosmology for angular-to-kpc conversions in the result.
    """

    def __init__(
        self,
        image: NoobImage,
        components: tuple[Profile, ...],
        bands: list[_BandStatic],
        comp_render: list[_CompRender],
        free: list[Param],
        names: dict[int, str],
        priors: dict[int, Prior],
        *,
        nu: float | None,
        background: bool,
        cosmology: FLRW | None,
    ) -> None:
        self.image = image
        self.components = components
        self._bands = bands
        self._comp_render = comp_render
        self._free = free
        self._names = names
        self._priors = priors
        self.nu = nu
        self.background = background
        self.cosmology = cosmology

    # ------------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        image: NoobImage,
        components: Sequence[Profile],
        *,
        oversample: int = 3,
        policy: PriorPolicy | None = None,
        nu: float | None = 4.0,
        background: bool = True,
        auto_mask: bool = False,
        mask_nsigma: float = 2.0,
        joint_mask: bool = True,
        fit_radius_arcsec: float | None = None,
        correlated_noise: bool = False,
        cosmology: FLRW | None = None,
    ) -> MorphModel:
        """Bind components to an image and resolve everything for sampling.

        Parameters
        ----------
        image : NoobImage
            The imaging to fit.
        components : sequence of Profile
            The declared components; must be non-empty with unique names.
        oversample : int, default 3
            Subpixel oversampling (tie to the PSF's sampling).
        policy : PriorPolicy, optional
            Default-prior policy; defaults to :class:`HighzPriorPolicy`.
        nu : float or None, default 4.0
            Student-t dof for the robust likelihood; ``None`` uses a Gaussian.
        background : bool, default True
            Fit a per-band constant background.
        auto_mask : bool, default False
            Derive and add neighbour exclude-masks. When enabled, neighbour
            detections are projected across bands by default and a hard fit radius
            is applied unless ``fit_radius_arcsec`` is set explicitly.
        mask_nsigma : float, default 2.0
            Detection threshold for the auto-mask.
        joint_mask : bool, default True
            If ``auto_mask`` is enabled, project neighbour detections from every
            band into every other band before taking the union. Set ``False`` for
            the older per-band-only mask.
        fit_radius_arcsec : float, optional
            Hard sky radius for pixels entering the likelihood. If omitted, no
            radius cut is applied when ``auto_mask=False``; when ``auto_mask=True``,
            each band uses ``min(0.5 arcsec, 0.4 * stamp_half_size)``.
        correlated_noise : bool, default False
            Calibrate for spatially-correlated (e.g. drizzle) noise from each
            band's source-free sky: rescale the per-pixel error to the true sky
            scatter and temper the per-band log-likelihood by the noise-correlation
            area, so the posterior is not overconfident. A full covariance-matrix
            likelihood is a TODO (see :meth:`numpyro_model`).
        cosmology : astropy.cosmology.FLRW, optional
            Cosmology for kpc conversion in the result (defaults to Planck18).

        Returns
        -------
        MorphModel

        Raises
        ------
        ValueError
            If there are no components, names collide, a band is unknown to a flux
            mapping, ``fit_radius_arcsec <= 0``, a band has no usable pixels after
            masks, or a parameter inside an expression lacks an explicit prior.
        """
        comps = tuple(components)
        if not comps:
            raise ValueError("need at least one component.")
        _check_unique_names(comps)
        policy = policy or HighzPriorPolicy()

        geoms = {
            name: build_geometry(band.wcs, image.center, band.shape, oversample)
            for name, band in image.bands.items()
        }
        exclude = _exclude_masks(
            image,
            geoms,
            auto_mask,
            mask_nsigma,
            joint_mask=joint_mask,
            fit_radius_arcsec=fit_radius_arcsec,
        )
        seeds = compute_seeds(image.bands, geoms, exclude=exclude)

        resolver = _Resolver(image, comps, policy, seeds)
        comp_render = resolver.resolve_components()
        free, names, priors = resolver.free, resolver.names, resolver.priors

        bands = [
            _build_band_static(
                name,
                image.bands[name],
                geoms[name],
                oversample,
                exclude[name],
                correlated_noise=correlated_noise,
                mask_nsigma=mask_nsigma,
            )
            for name in image.bands
        ]
        return cls(
            image,
            comps,
            bands,
            comp_render,
            free,
            names,
            priors,
            nu=nu,
            background=background,
            cosmology=cosmology,
        )

    # ------------------------------------------------------------------ priors

    @property
    def priors(self) -> dict[str, Prior]:
        """The resolved prior for every free parameter, keyed by its sampled name."""
        return {self._names[id(p)]: self._priors[id(p)] for p in self._free}

    @property
    def n_free(self) -> int:
        """Number of free (sampled) parameters, excluding any background terms."""
        return len(self._free)

    # ------------------------------------------------------------------ render

    def _render_comp_over(
        self, cr: _CompRender, bs: _BandStatic, env: dict[Param, Any]
    ) -> jax.Array:
        """Render one component onto the oversampled grid."""
        flux = cr.flux_by_band[bs.name].evaluate(env)
        if cr.kind == "point":
            return render_point(
                bs.col_grid,
                bs.row_grid,
                bs.affine,
                bs.pix_center,
                bs.oversample,
                x0=cr.geom["x0"].evaluate(env),
                y0=cr.geom["y0"].evaluate(env),
                flux=flux,
            )
        g = {a: node.evaluate(env) for a, node in cr.geom.items()}
        n = 1.0 if cr.kind == "exponential" else g["n"]
        return render_sersic(
            bs.e_grid,
            bs.n_grid,
            bs.subpixel_area,
            x0=g["x0"],
            y0=g["y0"],
            r_eff=g["r_eff"],
            n=n,
            q=g["q"],
            theta=g["theta"],
            flux=flux,
        )

    def _band_native(self, bs: _BandStatic, env: dict[Param, Any]) -> jax.Array:
        """Sum components, convolve once, bin to native resolution."""
        over = jnp.zeros(bs.oversampled_shape)
        for cr in self._comp_render:
            over = over + self._render_comp_over(cr, bs, env)
        return convolve_and_bin(over, bs.psf, bs.oversample)

    # ------------------------------------------------------------------ numpyro

    def numpyro_model(self) -> Any:
        """Return the numpyro model callable (samples params, renders, scores).

        Correlated noise (when ``correlated_noise=True`` at build) is handled in
        two pieces: ``obs_err`` is already rescaled per band so the per-pixel sigma
        matches the true sky scatter (fixing an underestimated error map), and the
        per-band log-likelihood is tempered by ``lik_scale`` (``1 / n_corr``, the
        noise-correlation area) so the posterior reflects the true number of
        independent pixels rather than the raw pixel count. A full covariance /
        Fourier-domain likelihood (the rigorous treatment of the off-diagonal
        correlation structure) is a TODO -- it needs a stationary, rectangular
        unmasked region and an FFT per evaluation, which conflicts with the
        neighbour mask.
        """
        import numpyro
        import numpyro.distributions as dist

        def model() -> None:
            env = {
                p: numpyro.sample(self._names[id(p)], _dist(self._priors[id(p)]))
                for p in self._free
            }
            for bs in self._bands:
                native = self._band_native(bs, env).reshape(-1)[bs.use_idx]
                if self.background:
                    native = native + numpyro.sample(
                        f"bg_{bs.name}", dist.Normal(0.0, bs.bg_scale)
                    )
                like = (
                    dist.Normal(native, bs.obs_err)
                    if self.nu is None
                    else dist.StudentT(self.nu, native, bs.obs_err)
                )
                with numpyro.handlers.scale(scale=bs.lik_scale):
                    numpyro.sample(f"obs_{bs.name}", like, obs=bs.obs)

        return model

    # ------------------------------------------------------------------ preview

    def preview(self, values: Mapping[str, float] | None = None) -> Preview:
        """Render the prior-mean (or supplied) scene, per component and summed.

        Parameters
        ----------
        values : mapping of str to float, optional
            Overrides for individual parameters, keyed by their sampled name. When
            background fitting is enabled, ``bg_<band>`` is included in the summed
            model if present; otherwise the preview background is zero.

        Returns
        -------
        Preview
        """
        values_in = values or {}
        env = self._point_env(values_in)
        data, total, residual, comps_out, masks = {}, {}, {}, {}, {}
        for bs in self._bands:
            band = self.image.bands[bs.name]
            per_comp: dict[str, np.ndarray] = {}
            tot = np.zeros(band.shape)
            for cr in self._comp_render:
                over = self._render_comp_over(cr, bs, env)
                native = np.asarray(convolve_and_bin(over, bs.psf, bs.oversample))
                per_comp[cr.name] = native
                tot = tot + native
            if self.background:
                tot = tot + float(values_in.get(f"bg_{bs.name}", 0.0))
            use = np.zeros(band.data.size, dtype=bool)
            use[bs.use_idx] = True
            data[bs.name] = band.data
            total[bs.name] = tot
            residual[bs.name] = band.data - tot
            comps_out[bs.name] = per_comp
            masks[bs.name] = use.reshape(band.shape)
        return Preview(
            bands=[bs.name for bs in self._bands],
            data=data,
            total=total,
            residual=residual,
            components=comps_out,
            use_mask=masks,
        )

    def _point_env(self, values: Mapping[str, float]) -> dict[Param, Any]:
        """Return a representative value for each parameter (prior centre or override)."""
        env: dict[Param, Any] = {}
        for p in self._free:
            name = self._names[id(p)]
            env[p] = (
                float(values[name])
                if name in values
                else _prior_point(self._priors[id(p)])
            )
        return env

    # ------------------------------------------------------------------ sample

    def sample(
        self,
        *,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 1,
        seed: int = 0,
        progress_bar: bool = False,
        **nuts_kwargs: Any,
    ) -> MorphResult:
        """Run NUTS and return a :class:`MorphResult`.

        Parameters
        ----------
        draws, tune : int
            Post-warmup samples and warmup steps per chain.
        chains : int, default 1
            Number of chains.
        seed : int, default 0
            PRNG seed.
        progress_bar : bool, default False
            Show numpyro's progress bar.
        **nuts_kwargs
            Forwarded to :class:`numpyro.infer.NUTS` (e.g. ``target_accept_prob``).

        Returns
        -------
        MorphResult
        """
        from numpyro.infer import MCMC, NUTS, init_to_median

        from noobfriend.inference.morphology._result import MorphResult

        nuts_kwargs.setdefault("init_strategy", init_to_median)
        kernel = NUTS(self.numpyro_model(), **nuts_kwargs)
        mcmc = MCMC(
            kernel,
            num_warmup=tune,
            num_samples=draws,
            num_chains=chains,
            progress_bar=progress_bar,
        )
        mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
        return MorphResult(self, mcmc)


# --------------------------------------------------------------------------- #
# Prior resolution
# --------------------------------------------------------------------------- #


class _Resolver:
    """Walks components, registering free parameters and resolving their priors."""

    def __init__(
        self,
        image: NoobImage,
        components: tuple[Profile, ...],
        policy: PriorPolicy,
        seeds: Any,
    ) -> None:
        self.image = image
        self.components = components
        self.policy = policy
        self.seeds = seeds
        self.free: list[Param] = []
        self.names: dict[int, str] = {}
        self.priors: dict[int, Prior] = {}
        self._seen: set[int] = set()
        self._used_names: set[str] = set()

    def resolve_components(self) -> list[_CompRender]:
        """Register all parameters and return the render-ready component list."""
        out: list[_CompRender] = []
        for comp in self.components:
            kind = _kind_of(comp)
            for axis, node in comp.params.items():
                self._register_node(node, comp.name, axis, None)
            flux_by_band = self._materialise_flux(comp)
            out.append(_CompRender(comp.name, kind, dict(comp.params), flux_by_band))
        return out

    def _materialise_flux(self, comp: Profile) -> dict[str, ParamNode]:
        """Expand the component's flux recipe to a node per band."""
        flux = comp.flux
        per_band = dict(flux.per_band) if flux.per_band is not None else None
        if per_band is not None:
            extra = set(per_band) - set(self.image.bands)
            if extra:
                raise ValueError(
                    f"{comp.name}: flux names band(s) {sorted(extra)} not in the image."
                )
        out: dict[str, ParamNode] = {}
        for band in self.image.bands:
            raw = per_band.get(band) if per_band is not None else flux.template
            if isinstance(raw, ParamNode):
                node = raw  # explicit shared parameter / expression
            else:
                node = parse_spec(raw, where=f"{comp.name}.flux[{band!r}]")
            self._register_node(node, comp.name, "flux", band)
            out[band] = node
        return out

    def _register_node(
        self, node: ParamNode, comp: str, axis: str, band: str | None
    ) -> None:
        """Register the free parameters of one node."""
        is_bare = isinstance(node, Param)
        for p in collect_params(node):
            if id(p) in self._seen:
                continue
            self._seen.add(id(p))
            if p.prior is not None:
                prior = p.prior
            elif is_bare and node is p:
                prior = self.policy.prior_for(axis, self.seeds, band=band)
            else:
                raise ValueError(
                    f"{comp}.{axis}: parameter {p.name or '<anon>'} inside an "
                    "expression needs an explicit prior."
                )
            self.free.append(p)
            self.priors[id(p)] = prior
            self.names[id(p)] = self._name_for(p, comp, axis, band)

    def _name_for(self, p: Param, comp: str, axis: str, band: str | None) -> str:
        """Return a unique sampled name for a free parameter."""
        if p.name is not None:
            if p.name in self._used_names:
                raise ValueError(f"duplicate parameter name {p.name!r}.")
            self._used_names.add(p.name)
            return p.name
        base = f"{comp}_{axis}" + (f"_{band}" if band is not None else "")
        name = base
        i = 2
        while name in self._used_names:
            name = f"{base}_{i}"
            i += 1
        self._used_names.add(name)
        return name


# --------------------------------------------------------------------------- #
# Static band assembly
# --------------------------------------------------------------------------- #


def _exclude_masks(
    image: NoobImage,
    geoms: dict[str, Any],
    auto_mask: bool,
    mask_nsigma: float,
    *,
    joint_mask: bool = True,
    fit_radius_arcsec: float | None = None,
) -> dict[str, np.ndarray]:
    """Per-band exclude masks for user masks, radius cuts, and neighbours.

    Computed once and shared by the seeding and the likelihood, so the centroid
    seed and the fitted pixels see the same exclusions.
    """
    from noobfriend.inference.morphology._mask import (
        neighbour_mask_band,
        project_mask_to_geometry,
        radial_exclude_mask,
        target_protection_mask,
    )

    radii = {
        name: _resolved_fit_radius(geom, fit_radius_arcsec, auto_mask)
        for name, geom in geoms.items()
    }

    protect: dict[str, np.ndarray] = {}
    neighbours: dict[str, np.ndarray] = {}
    if auto_mask:
        for name, band in image.bands.items():
            radius = radii[name]
            protect[name] = target_protection_mask(
                band.data,
                band.error,
                geoms[name],
                nsigma=mask_nsigma,
                core_radius_arcsec=_target_core_radius(geoms[name], radius),
                max_radius_arcsec=radius,
            )
            neighbours[name] = neighbour_mask_band(
                band.data, geoms[name], protect[name], nsigma=mask_nsigma
            )

    out: dict[str, np.ndarray] = {}
    for name, band in image.bands.items():
        ex = np.zeros(band.shape, dtype=bool)
        if band.mask is not None:
            ex |= band.mask
        radius = radii[name]
        if radius is not None:
            ex |= radial_exclude_mask(geoms[name], radius)
        if auto_mask:
            if joint_mask:
                neighbour = np.zeros(band.shape, dtype=bool)
                for src_name, src_mask in neighbours.items():
                    neighbour |= project_mask_to_geometry(
                        src_mask, geoms[src_name], geoms[name]
                    )
            else:
                neighbour = neighbours[name]
            ex |= neighbour & ~protect[name]
        out[name] = ex
    return out


def _resolved_fit_radius(
    geom: Any, requested: float | None, auto_mask: bool
) -> float | None:
    """Return the hard fit radius for one band, or ``None`` when disabled."""
    if requested is not None:
        radius = float(requested)
        if radius <= 0:
            raise ValueError(f"fit_radius_arcsec must be > 0, got {requested}.")
        return radius
    if not auto_mask:
        return None
    return min(0.5, 0.4 * _stamp_half_size_arcsec(geom))


def _stamp_half_size_arcsec(geom: Any) -> float:
    """Return half of the smaller native stamp dimension in arcsec."""
    rows, cols = geom.native_shape
    return 0.5 * min(rows, cols) * _native_pixscale_arcsec(geom)


def _native_pixscale_arcsec(geom: Any) -> float:
    """Return the representative native pixel scale in arcsec."""
    return float(np.sqrt(geom.subpixel_area) * geom.oversample)


def _target_core_radius(geom: Any, fit_radius: float | None) -> float:
    """Small geometric target-protection radius used before aperture growth."""
    radius = min(max(1.5 * _native_pixscale_arcsec(geom), 0.1), 0.2)
    if fit_radius is not None:
        radius = min(radius, 0.5 * fit_radius)
    return max(radius, np.finfo(float).eps)


def _build_band_static(
    name: str,
    band: Any,
    geom: Any,
    oversample: int,
    exclude: np.ndarray,
    *,
    correlated_noise: bool = False,
    mask_nsigma: float = 2.0,
) -> _BandStatic:
    """Precompute the static render inputs and fit mask for one band."""
    data = np.asarray(band.data, dtype=float)
    err = np.asarray(band.error, dtype=float)
    rows, cols = data.shape
    o = geom.oversample

    valid = np.isfinite(data) & np.isfinite(err) & (err > 0)
    use = valid & ~exclude
    use_idx = np.flatnonzero(use.reshape(-1))
    if use_idx.size == 0:
        raise ValueError(
            f"band {name!r} has no usable pixels after applying finite data/error "
            "checks and fit masks."
        )

    beta, lik_scale = 1.0, 1.0
    if correlated_noise:
        beta, lik_scale = _correlated_noise_factors(
            data, err, valid, exclude, mask_nsigma
        )
    err = err * beta

    col = jnp.arange(cols * o, dtype=float)
    row = jnp.arange(rows * o, dtype=float)
    col_grid, row_grid = jnp.meshgrid(col, row)

    pixscale = float(np.sqrt(geom.subpixel_area) * o)
    psf = resolve_psf(band.psf, band=name, oversample=oversample, pixscale=pixscale)
    bg_scale = float(np.median(err[use]))

    return _BandStatic(
        name=name,
        e_grid=jnp.asarray(geom.e_grid),
        n_grid=jnp.asarray(geom.n_grid),
        col_grid=col_grid,
        row_grid=row_grid,
        affine=jnp.asarray(geom.affine),
        pix_center=geom.pix_center,
        oversample=o,
        subpixel_area=geom.subpixel_area,
        oversampled_shape=(rows * o, cols * o),
        psf=jnp.asarray(psf),
        obs=jnp.asarray(data.reshape(-1)[use_idx]),
        obs_err=jnp.asarray(err.reshape(-1)[use_idx]),
        use_idx=use_idx,
        bg_scale=bg_scale,
        lik_scale=lik_scale,
    )


def _correlated_noise_factors(
    data: np.ndarray,
    err: np.ndarray,
    valid: np.ndarray,
    exclude: np.ndarray,
    mask_nsigma: float,
) -> tuple[float, float]:
    """Return ``(beta, lik_scale)`` from a band's source-free sky.

    ``beta = sqrt(C(0) / <err^2>)`` rescales the per-pixel error to the true sky
    scatter; ``lik_scale = 1 / n_corr`` with ``n_corr = sum_d C(d) / C(0)`` is the
    log-likelihood weight tempering the per-band fit by the noise-correlation area
    (``n_corr = 1`` for white noise). Falls back to ``(1, 1)`` when the sky has too
    few pixels to estimate the autocovariance reliably.
    """
    from noobfriend.core.imgutils import noise_autocovariance, segment

    max_lag = 8
    labels = segment(data, nsigma=mask_nsigma)
    sky = (labels == 0) & valid & ~exclude
    if int(sky.sum()) < 4 * (2 * max_lag + 1) ** 2:
        return 1.0, 1.0

    ac = noise_autocovariance(data, mask=sky, error=err, max_lag=max_lag)
    variance = ac.variance
    if variance <= 0 or not ac.error_var or ac.error_var <= 0:
        return 1.0, 1.0
    beta = float(np.sqrt(variance / ac.error_var))
    n_corr = max(float(ac.cov.sum() / variance), 1.0)
    return beta, 1.0 / n_corr


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _kind_of(comp: Profile) -> str:
    """Map a profile class to a render kind string."""
    if isinstance(comp, Sersic):
        return "sersic"
    if isinstance(comp, Exponential):
        return "exponential"
    if isinstance(comp, PointSource):
        return "point"
    raise ValueError(f"unsupported profile type {type(comp).__name__}.")


def _check_unique_names(components: tuple[Profile, ...]) -> None:
    """Raise if two components share a name."""
    seen: set[str] = set()
    for comp in components:
        if comp.name in seen:
            raise ValueError(f"duplicate component name {comp.name!r}.")
        seen.add(comp.name)


def _dist(prior: Prior) -> Any:
    """Translate a declarative prior into a numpyro distribution."""
    import numpyro.distributions as dist

    if isinstance(prior, Uniform):
        return dist.Uniform(prior.low, prior.high)
    if isinstance(prior, Normal):
        return dist.Normal(prior.loc, prior.scale)
    if isinstance(prior, TruncNormal):
        return dist.TruncatedNormal(
            prior.loc, prior.scale, low=prior.low, high=prior.high
        )
    if isinstance(prior, LogUniform):
        return dist.LogUniform(prior.low, prior.high)
    raise ValueError(f"no numpyro distribution for {type(prior).__name__}.")


def _prior_point(prior: Prior) -> float:
    """Return a representative central value of a prior, for previews / init."""
    if isinstance(prior, Uniform):
        return 0.5 * (prior.low + prior.high)
    if isinstance(prior, Normal):
        return prior.loc
    if isinstance(prior, TruncNormal):
        return float(min(max(prior.loc, prior.low), prior.high))
    if isinstance(prior, LogUniform):
        return float(math.sqrt(prior.low * prior.high))
    raise ValueError(f"no point estimate for {type(prior).__name__}.")
