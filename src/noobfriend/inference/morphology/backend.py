"""Lazy sampler backend selection for morphology fitting."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from noobfriend.inference.morphology.components import (
    Background,
    EllipticalOffsetFrom,
    FixedCenter,
    FreeCenter,
    Point,
    Sersic,
)
from noobfriend.inference.morphology.data import NoobImage, NoobImageSet
from noobfriend.inference.morphology.introspection import scene_parameters
from noobfriend.inference.morphology.parameters import (
    FluxPart,
    LogUniform,
    Normal,
    Parameter,
    PerBand,
    Prior,
    TruncNormal,
    Uniform,
)
from noobfriend.inference.morphology.parameters import uv_backend_message
from noobfriend.inference.morphology.result import FitDiagnostics, FitResult
from noobfriend.inference.morphology.render import render_scene
from noobfriend.inference.morphology.scene import Scene

BackendName = Literal["auto", "cpu", "cuda"]


class MissingSamplerBackendError(ImportError):
    """Raised when an optional sampler backend is not installed."""


def require_numpyro_backend() -> None:
    """Raise with a uv-oriented install hint if JAX/NumPyro is unavailable."""
    missing = [
        package
        for package in ["jax", "numpyro"]
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise MissingSamplerBackendError(uv_backend_message())


def choose_jax_platform(preferred: BackendName = "auto") -> str:
    """Return ``"cuda"`` or ``"cpu"`` for a lazy JAX backend."""
    require_numpyro_backend()
    import jax

    devices = jax.devices()
    has_gpu = any(device.platform in {"gpu", "cuda"} for device in devices)
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        if not has_gpu:
            raise RuntimeError(
                "JAX CUDA backend was requested, but no GPU device is visible."
            )
        return "cuda"
    return "cuda" if has_gpu else "cpu"


@dataclass(frozen=True)
class NumpyroNUTSBackend:
    """Configuration shell for the future Numpyro NUTS sampler."""

    backend: BackendName = "auto"
    chains: int = 4
    tune: int = 1000
    draws: int = 1000
    target_accept: float = 0.95
    max_tree_depth: int = 8
    record_log_likelihood: bool = True

    def platform(self) -> str:
        """Resolve the runtime JAX platform."""
        return choose_jax_platform(self.backend)

    def fit(
        self,
        images: NoobImageSet,
        scene: Scene,
        *,
        fixed_params: dict[str, float] | None = None,
        init_params: dict[str, float] | None = None,
        seed: int = 0,
    ) -> FitResult:
        """Sample a morphology scene with NumPyro NUTS."""
        platform = self.platform()
        import jax
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value

        fixed = dict(fixed_params or {})
        params = scene_parameters(scene)
        sampled_params = tuple(param for param in params if param.name not in fixed)
        priors = _require_priors(sampled_params)
        init_values = {
            param.name: param.init
            for param in sampled_params
            if param.init is not None and param.name not in (init_params or {})
        }
        init_values.update(init_params or {})
        init_values = _sanitize_init_values(sampled_params, init_values)
        init_strategy = (
            init_to_value(values=init_values) if init_values else init_to_median
        )
        statics = tuple(_ImageStatic.from_image(image) for image in images)

        def model() -> None:
            values: dict[str, Any] = dict(fixed)
            for param in sampled_params:
                values[param.name] = numpyro.sample(
                    param.name, _numpyro_dist(priors[param.name], dist)
                )
            loglikes = []
            for static in statics:
                mean = _jax_render_image(static, scene, values, jnp)
                obs = static.data[static.mask]
                err = static.error[static.mask]
                pred = mean[static.mask]
                like = dist.Normal(pred, err)
                if self.record_log_likelihood:
                    ll = like.log_prob(obs)
                    loglikes.append(ll)
                    numpyro.factor(f"obs_{static.name}", jnp.sum(ll))
                else:
                    numpyro.sample(f"obs_{static.name}", like, obs=obs)
            if self.record_log_likelihood:
                numpyro.deterministic("log_likelihood", jnp.concatenate(loglikes))

        kernel = NUTS(
            model,
            target_accept_prob=self.target_accept,
            max_tree_depth=self.max_tree_depth,
            init_strategy=init_strategy,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=self.tune,
            num_samples=self.draws,
            num_chains=self.chains,
            chain_method="vectorized" if platform == "cuda" else "parallel",
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",))
        samples = mcmc.get_samples(group_by_chain=True)
        flat_samples = mcmc.get_samples(group_by_chain=False)
        median = {
            name: float(np.nanmedian(np.asarray(value)))
            for name, value in flat_samples.items()
            if name != "log_likelihood"
        }
        median.update(fixed)
        rendered = render_scene(images, scene, median)
        extra = mcmc.get_extra_fields(group_by_chain=True)
        return FitResult(
            posterior=samples,
            diagnostics=_fit_diagnostics(samples, extra),
            rendered=rendered,
            log_likelihood=np.asarray(samples.get("log_likelihood"))
            if "log_likelihood" in samples
            else None,
        )


@dataclass(frozen=True)
class _ImageStatic:
    """JAX-ready static arrays for one image."""

    name: str
    data: Any
    error: Any
    mask: Any
    x: Any
    y: Any
    pixel_scale: float
    shape: tuple[int, int]
    kernel: Any

    @classmethod
    def from_image(cls, image: NoobImage) -> "_ImageStatic":
        """Build static arrays from a prepared image."""
        import jax.numpy as jnp

        x, y = image.arcsec_grid()
        return cls(
            name=image.name,
            data=jnp.asarray(image.data),
            error=jnp.asarray(image.error),
            mask=jnp.asarray(image.mask, dtype=bool),
            x=jnp.asarray(x),
            y=jnp.asarray(y),
            pixel_scale=float(image.pixel_scale),
            shape=image.shape,
            kernel=jnp.asarray(image.psf_kernel()),
        )


def _require_priors(params: tuple[Parameter, ...]) -> dict[str, Prior]:
    """Return priors, requiring every sampled parameter to declare one."""
    priors: dict[str, Prior] = {}
    missing = []
    for param in params:
        if param.prior is None:
            missing.append(param.name)
        else:
            priors[param.name] = param.prior
    if missing:
        raise ValueError(f"sampled parameters need priors: {', '.join(missing)}")
    return priors


def _sanitize_init_values(
    params: tuple[Parameter, ...], init_values: dict[str, float]
) -> dict[str, float]:
    """Clip explicit sampler starts into each parameter's prior support."""
    by_name = {param.name: param for param in params}
    sanitized: dict[str, float] = {}
    for name, value in init_values.items():
        param = by_name.get(name)
        if param is None:
            continue
        sanitized[name] = _safe_init_value(float(value), param)
    return sanitized


def _safe_init_value(value: float, param: Parameter) -> float:
    """Return a finite initial value with positive density under ``param.prior``."""
    prior = param.prior
    if not np.isfinite(value):
        value = _prior_midpoint(prior)
    if isinstance(prior, Uniform):
        eps = 1e-6 * max(abs(prior.high - prior.low), 1.0)
        return float(np.clip(value, prior.low + eps, prior.high - eps))
    if isinstance(prior, LogUniform):
        low = prior.low * (1.0 + 1e-6)
        high = prior.high / (1.0 + 1e-6)
        if value <= 0 or not np.isfinite(value):
            value = np.sqrt(prior.low * prior.high)
        return float(np.clip(value, low, high))
    if isinstance(prior, TruncNormal):
        eps = 1e-6 * max(abs(prior.high - prior.low), 1.0)
        return float(np.clip(value, prior.low + eps, prior.high - eps))
    if isinstance(prior, Normal):
        return float(value if np.isfinite(value) else prior.loc)
    return float(value)


def _prior_midpoint(prior: Prior | None) -> float:
    """Return a deterministic fallback start from a prior."""
    if isinstance(prior, Uniform):
        return 0.5 * (prior.low + prior.high)
    if isinstance(prior, LogUniform):
        return float(np.sqrt(prior.low * prior.high))
    if isinstance(prior, TruncNormal):
        return float(np.clip(prior.loc, prior.low, prior.high))
    if isinstance(prior, Normal):
        return float(prior.loc)
    return 0.0


def _numpyro_dist(prior: Prior, dist: Any) -> Any:
    """Translate a declarative prior into a NumPyro distribution."""
    if isinstance(prior, Uniform):
        return dist.Uniform(prior.low, prior.high)
    if isinstance(prior, Normal):
        return dist.Normal(prior.loc, prior.scale)
    if isinstance(prior, TruncNormal):
        return dist.TruncatedNormal(
            prior.loc, prior.scale, low=prior.low, high=prior.high
        )
    if isinstance(prior, LogUniform):
        return dist.TransformedDistribution(
            dist.Uniform(np.log(prior.low), np.log(prior.high)),
            dist.transforms.ExpTransform(),
        )
    raise TypeError(f"unsupported prior type {type(prior)!r}")


def _fit_diagnostics(samples: dict[str, Any], extra: dict[str, Any]) -> FitDiagnostics:
    """Return sampler diagnostics from posterior samples."""
    divergences = None
    if "diverging" in extra:
        divergences = int(np.asarray(extra["diverging"]).sum())
    try:
        import arviz as az

        idata = az.from_dict(
            {"posterior": {name: np.asarray(value) for name, value in samples.items()}}
        )
        summary = az.summary(idata, kind="diagnostics")
        return FitDiagnostics(
            max_rhat=float(summary["r_hat"].max(skipna=True)),
            min_ess=float(summary["ess_bulk"].min(skipna=True)),
            divergences=divergences,
        )
    except Exception:
        return FitDiagnostics(divergences=divergences)


def _jax_render_image(
    static: _ImageStatic, scene: Scene, values: dict[str, Any], jnp: Any
) -> Any:
    """Render one image in JAX."""
    total = jnp.zeros(static.shape)
    for component in scene:
        if isinstance(component, Background):
            level = _resolve_flux_jax(component.level, static.name, values, jnp)
            model = jnp.full(static.shape, level)
            total = total + model
            continue
        flux = _resolve_flux_jax(component.flux, static.name, values, jnp)
        if isinstance(component, Point):
            center = _resolve_center_jax(component.center, scene, values, jnp)
            model = _jax_point(static, center, flux, jnp)
        elif isinstance(component, Sersic):
            center = _resolve_center_jax(component.center, scene, values, jnp)
            if component.shape is None:
                raise ValueError("Sersic component has no shape.")
            model = _jax_sersic(static, component, center, values, flux, jnp)
        else:
            raise TypeError(f"unsupported component type {type(component)!r}")
        total = total + _jax_convolve(model, static.kernel)
    return total


def _resolve_scalar_jax(value: Any, values: dict[str, Any]) -> Any:
    """Resolve a scalar spec in a JAX model."""
    if isinstance(value, Parameter):
        return values[value.name]
    if isinstance(value, str):
        return values[value]
    if isinstance(value, bool):
        raise TypeError("boolean scalar specs are not supported.")
    return float(value)


def _resolve_flux_jax(value: Any, band: str, values: dict[str, Any], jnp: Any) -> Any:
    """Resolve a flux spec in a JAX model."""
    if isinstance(value, FluxPart):
        total = _resolve_flux_jax(value.split.total, band, values, jnp)
        fraction = jnp.clip(
            _resolve_flux_jax(value.split.fraction, band, values, jnp), 0.0, 1.0
        )
        if value.part_name == "host":
            return total * fraction
        return total * (1.0 - fraction)
    if isinstance(value, PerBand):
        if value.values and band in value.values:
            return _resolve_scalar_jax(value.values[band], values)
        if value.prefix is not None:
            return values[f"{value.prefix}_{band}"]
        if value.default is not None:
            return _resolve_scalar_jax(value.default, values)
        raise KeyError(f"no flux value for band {band!r}")
    if isinstance(value, dict):
        return _resolve_scalar_jax(value[band], values)
    return _resolve_scalar_jax(value, values)


def _resolve_center_jax(
    center: Any, scene: Scene, values: dict[str, Any], jnp: Any
) -> tuple[Any, Any]:
    """Resolve a centre object in a JAX model."""
    if isinstance(center, FixedCenter):
        return float(center.x), float(center.y)
    if isinstance(center, FreeCenter):
        return _resolve_scalar_jax(center.x, values), _resolve_scalar_jax(
            center.y, values
        )
    if isinstance(center, EllipticalOffsetFrom):
        ref = scene.component(center.reference)
        ref_x, ref_y = _resolve_center_jax(ref.center, scene, values, jnp)
        r_eff = _resolve_scalar_jax(center.r_eff, values)
        q = _resolve_scalar_jax(center.q, values)
        theta = _resolve_scalar_jax(center.theta, values) * jnp.pi / 180.0
        rho = _resolve_scalar_jax(center.rho, values)
        phi = _resolve_scalar_jax(center.phi, values) * jnp.pi / 180.0
        dx = (
            r_eff
            * rho
            * (jnp.cos(phi) * jnp.cos(theta) - q * jnp.sin(phi) * jnp.sin(theta))
        )
        dy = (
            r_eff
            * rho
            * (jnp.cos(phi) * jnp.sin(theta) + q * jnp.sin(phi) * jnp.cos(theta))
        )
        return ref_x + dx, ref_y + dy
    raise TypeError(f"unsupported center type {type(center)!r}")


def _jax_point(
    static: _ImageStatic, center: tuple[Any, Any], flux: Any, jnp: Any
) -> Any:
    """Render a bilinear point source in JAX."""
    x0, y0 = center
    cx = (static.shape[1] - 1) / 2.0 + x0 / static.pixel_scale
    cy = (static.shape[0] - 1) / 2.0 + y0 / static.pixel_scale
    ix = jnp.floor(cx).astype(jnp.int32)
    iy = jnp.floor(cy).astype(jnp.int32)
    dx = cx - ix
    dy = cy - iy
    out = jnp.zeros(static.shape)
    for yy, wy in [(iy, 1.0 - dy), (iy + 1, dy)]:
        for xx, wx in [(ix, 1.0 - dx), (ix + 1, dx)]:
            valid = (
                (yy >= 0) & (yy < static.shape[0]) & (xx >= 0) & (xx < static.shape[1])
            )
            out = out.at[
                jnp.clip(yy, 0, static.shape[0] - 1),
                jnp.clip(xx, 0, static.shape[1] - 1),
            ].add(jnp.where(valid, flux * wy * wx, 0.0))
    return out


def _jax_sersic(
    static: _ImageStatic,
    component: Sersic,
    center: tuple[Any, Any],
    values: dict[str, Any],
    flux: Any,
    jnp: Any,
) -> Any:
    """Render an unconvolved Sersic profile in JAX."""
    shape = component.shape
    if shape is None:
        raise ValueError("Sersic component has no shape.")
    x0, y0 = center
    r_eff = jnp.maximum(_resolve_scalar_jax(shape.r_eff, values), 1e-6)
    n = jnp.maximum(_resolve_scalar_jax(shape.n, values), 0.05)
    q = jnp.clip(_resolve_scalar_jax(shape.q, values), 1e-3, 1.0)
    theta = _resolve_scalar_jax(shape.theta, values) * jnp.pi / 180.0
    dx = static.x - x0
    dy = static.y - y0
    xp = dx * jnp.cos(theta) + dy * jnp.sin(theta)
    yp = -dx * jnp.sin(theta) + dy * jnp.cos(theta)
    r_ell = jnp.sqrt(xp * xp + (yp / q) ** 2)
    bn = 2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n)
    profile = jnp.exp(-bn * ((r_ell / r_eff) ** (1.0 / n) - 1.0))
    return profile * (flux / jnp.maximum(jnp.sum(profile), 1e-30))


def _jax_convolve(image: Any, kernel: Any) -> Any:
    """Convolve a 2-D image with a 2-D kernel in JAX."""
    import jax.scipy.signal as jsignal

    return jsignal.convolve(image, kernel, mode="same")
