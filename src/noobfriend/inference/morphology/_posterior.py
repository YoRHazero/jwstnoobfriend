"""Collapsed log-posterior assembly.

``build_posterior`` turns (images, model, priors) into a single differentiable
log-density over the latent standard-normal coordinates, plus the closures
needed after sampling: exact amplitude-posterior draws, pointwise
log-likelihood for PSIS-LOO, and full-model rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp

from noobfriend.inference.morphology._collapse import (
    conditional_amplitudes,
    marginal_loglik,
)
from noobfriend.inference.morphology._render import (
    BandStatic,
    prepare_static,
    source_images,
)
from noobfriend.inference.morphology._transforms import Transform, build_transform
from noobfriend.inference.morphology.data import NoobImageSet
from noobfriend.inference.morphology.models import ModelSpec
from noobfriend.inference.morphology.priors import Priors

__all__ = ["Posterior", "build_posterior"]


def _whitened_designs(
    statics: tuple[BandStatic, ...],
    spec: ModelSpec,
    transform: Transform,
    z: Any,
) -> tuple[dict[str, Any], list[Any]]:
    """Return physical parameters and per-band whitened design rows."""
    phys = transform.physical(z, jnp)
    designs = []
    for static in statics:
        invsig = static.invsig.ravel()
        sources = source_images(static, spec, phys)
        rows = sources.reshape(sources.shape[0], -1) * invsig[None, :]
        designs.append(jnp.concatenate([rows, invsig[None, :]], axis=0))
    return phys, designs


@dataclass(frozen=True)
class Posterior:
    """Callable bundle around one model's collapsed posterior.

    Attributes
    ----------
    spec, priors, transform, statics
        The model definition and prepared per-band arrays.
    log_density : callable
        Jitted ``z -> log p(z | y)`` (up to a constant), where ``z`` has the
        standard-normal prior.
    n_obs : int
        Total number of masked pixels across bands (the LOO observation
        count).
    """

    spec: ModelSpec
    priors: Priors
    transform: Transform
    statics: tuple[BandStatic, ...]
    amplitude_scales: tuple[Any, ...]
    log_density: Callable[[Any], Any]
    n_obs: int

    @property
    def dim(self) -> int:
        """Return the latent dimension."""
        return self.transform.dim

    @property
    def band_names(self) -> tuple[str, ...]:
        """Return band names in static order."""
        return tuple(static.name for static in self.statics)

    def whitened_designs(self, z: Any) -> tuple[dict[str, Any], list[Any]]:
        """Return physical parameters and per-band whitened design rows.

        Parameters
        ----------
        z : jax.Array
            Latent coordinates, shape ``(dim,)``.

        Returns
        -------
        tuple
            ``(phys, designs)`` where each design has shape
            ``(n_amplitudes, n_pixels)`` with masked pixels zeroed.
        """
        return _whitened_designs(self.statics, self.spec, self.transform, z)

    def draw_amplitudes(
        self, z: Any, eps: Any, *, pointwise: bool = False
    ) -> tuple[Any, Any | None]:
        """Draw exact amplitude posteriors for one latent sample.

        Parameters
        ----------
        z : jax.Array
            Latent coordinates, shape ``(dim,)``.
        eps : jax.Array
            Standard-normal innovations, shape ``(n_bands, n_amplitudes)``.
        pointwise : bool
            Also return the pointwise conditional log-likelihood over all
            masked pixels (for PSIS-LOO).

        Returns
        -------
        tuple
            ``(amplitudes, loglik)`` with shapes ``(n_bands, n_amplitudes)``
            and ``(n_obs,)`` (or ``None``).
        """
        _, designs = self.whitened_designs(z)
        amps = []
        logliks = []
        for static, x_w, v, eps_band in zip(
            self.statics, designs, self.amplitude_scales, eps
        ):
            mean, factor = conditional_amplitudes(static.y_w.ravel(), x_w, v)
            a = mean + factor @ eps_band
            amps.append(a)
            if pointwise:
                resid_w = static.y_w.ravel() - x_w.T @ a
                invsig = static.invsig.ravel()
                ll = (
                    -0.5 * resid_w**2
                    - 0.5 * jnp.log(2.0 * jnp.pi)
                    + jnp.log(jnp.where(invsig > 0, invsig, 1.0))
                )
                logliks.append(ll[static.mask_idx])
        loglik = jnp.concatenate(logliks) if pointwise else None
        return jnp.stack(amps), loglik

    def render(
        self, z: np.ndarray, amplitudes: np.ndarray
    ) -> dict[str, dict[str, np.ndarray]]:
        """Render per-component and total model images at one point.

        Parameters
        ----------
        z : numpy.ndarray
            Latent coordinates, shape ``(dim,)``.
        amplitudes : numpy.ndarray
            Amplitudes per band, shape ``(n_bands, n_amplitudes)`` in
            component order ``point0, ..., host, background``.

        Returns
        -------
        dict
            ``band -> {component: image, ..., "total": image}``.
        """
        phys = self.transform.physical(jnp.asarray(z), jnp)
        names = self.spec.component_names
        rendered: dict[str, dict[str, np.ndarray]] = {}
        for static, a in zip(self.statics, np.asarray(amplitudes)):
            sources = np.asarray(source_images(static, self.spec, phys))
            per: dict[str, np.ndarray] = {
                "background": np.full(static.shape, float(a[-1]))
            }
            total = per["background"]
            for k, name in enumerate(names):
                image = float(a[k]) * sources[k]
                per[name] = image
                total = total + image
            per["total"] = total
            rendered[static.name] = per
        return rendered


def build_posterior(
    images: NoobImageSet,
    spec: ModelSpec,
    priors: Priors,
    *,
    oversample: int = 4,
) -> Posterior:
    """Build the collapsed posterior for one model.

    Parameters
    ----------
    images : NoobImageSet
        Prepared images.
    spec : ModelSpec
        Model to fit.
    priors : Priors
        Prior configuration; amplitude scales must cover every band.
    oversample : int
        Sersic sub-pixel evaluation factor per axis.

    Returns
    -------
    Posterior
        The jitted log-density and post-sampling helpers.
    """
    statics = tuple(prepare_static(image, oversample=oversample) for image in images)
    transform = build_transform(spec, priors)
    n_sources = spec.n_points + int(spec.has_host)
    scales = []
    for static in statics:
        flux_scale, background_scale = priors.amplitudes.for_band(static.name)
        scales.append(
            jnp.asarray([flux_scale] * n_sources + [background_scale], dtype=float)
        )
    amplitude_scales = tuple(scales)

    def _log_density(z: Any) -> Any:
        _, designs = _whitened_designs(statics, spec, transform, z)
        lp = -0.5 * jnp.sum(z * z)
        for static, x_w, v in zip(statics, designs, amplitude_scales):
            lp = lp + marginal_loglik(static.y_w.ravel(), x_w, v, static.loglik_const)
        return lp

    return Posterior(
        spec=spec,
        priors=priors,
        transform=transform,
        statics=statics,
        amplitude_scales=amplitude_scales,
        log_density=jax.jit(_log_density),
        n_obs=int(sum(static.n_masked for static in statics)),
    )
