"""End-to-end decomposition drivers.

``fit_model`` runs one model through the full pipeline: collapsed-posterior
construction, Sobol multistart MAP with Laplace curvature, dense-mass NUTS
seeded at the MAP modes, exact amplitude recovery, diagnostics, and PSIS-LOO.
``decompose`` fits several members of the model family and compares them
predictively.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.morphology._map import MultistartConfig
from noobfriend.inference.morphology._nuts import NutsConfig
from noobfriend.inference.morphology.data import NoobImageSet
from noobfriend.inference.morphology.models import ModelSpec
from noobfriend.inference.morphology.priors import Priors
from noobfriend.inference.morphology.results import (
    Comparison,
    DecompositionResult,
    ModelFit,
)

if TYPE_CHECKING:
    from noobfriend.inference.morphology._posterior import Posterior

__all__ = [
    "MissingSamplerBackendError",
    "SamplerConfig",
    "decompose",
    "fit_model",
]


class MissingSamplerBackendError(ImportError):
    """Raised when the JAX/NumPyro sampling stack is not installed."""


def _require_backend() -> None:
    """Raise with a uv-oriented install hint if the backend is missing."""
    missing = [
        package
        for package in ("jax", "numpyro", "arviz")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise MissingSamplerBackendError(
            "the morphology sampler needs JAX, NumPyro and ArviZ.\n"
            'Install them with:  uv add "jax[cpu]" numpyro arviz\n'
            "(or the CUDA jax build matching your GPU)."
        )


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for one model fit.

    Parameters
    ----------
    nuts : NutsConfig
        NUTS chain configuration.
    multistart : MultistartConfig
        MAP initializer configuration.
    oversample : int
        Sersic sub-pixel evaluation factor per axis.
    compute_loo : bool
        Compute pointwise log-likelihood and PSIS-LOO.
    loo_thin : int
        Keep every ``loo_thin``-th draw for the pointwise log-likelihood
        (memory control; LOO needs far fewer draws than the posterior).
    amplitude_batch : int
        Draw-batch size for vectorized amplitude recovery.
    precision : str
        ``"float64"`` or ``"float32"``. The marginal likelihood is computed
        in a cancellation-free residual form, so float32 is numerically
        viable and is dramatically faster on consumer GPUs (whose float64
        throughput is 1/32 of float32); float64 remains the conservative
        default. Validate float32 results against a float64 fit once per
        study configuration.
    """

    nuts: NutsConfig = NutsConfig()
    multistart: MultistartConfig = MultistartConfig()
    oversample: int = 4
    compute_loo: bool = True
    loo_thin: int = 4
    amplitude_batch: int = 64
    precision: str = "float64"


def fit_model(
    images: NoobImageSet,
    model: ModelSpec | str,
    *,
    priors: Priors | None = None,
    sampler: SamplerConfig = SamplerConfig(),
    seed: int = 0,
) -> ModelFit:
    """Fit one model of the decomposition family.

    Parameters
    ----------
    images : NoobImageSet
        Prepared images (or an iterable of ``NoobImage``).
    model : ModelSpec or str
        Model to fit, e.g. ``"point+host"``.
    priors : Priors, optional
        Prior configuration; defaults to :meth:`Priors.from_images`.
    sampler : SamplerConfig
        Pipeline configuration.
    seed : int
        Master seed for the initializer, sampler and amplitude draws.

    Returns
    -------
    ModelFit
        Posterior draws, amplitude posteriors, diagnostics and timings.
    """
    _require_backend()
    spec = ModelSpec.parse(model)
    images = images if isinstance(images, NoobImageSet) else NoobImageSet(images)
    priors = priors or Priors.from_images(images)

    if sampler.precision not in ("float64", "float32"):
        raise ValueError("SamplerConfig.precision must be 'float64' or 'float32'.")

    import numpyro

    numpyro.set_host_device_count(sampler.nuts.chains)

    from noobfriend.inference.morphology._map import find_map
    from noobfriend.inference.morphology._nuts import run_nuts
    from noobfriend.inference.morphology._posterior import build_posterior

    import jax

    jax.config.update("jax_enable_x64", sampler.precision == "float64")

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    posterior = build_posterior(images, spec, priors, oversample=sampler.oversample)
    timings["prepare_s"] = time.perf_counter() - start

    start = time.perf_counter()
    map_result = find_map(
        posterior.log_density, posterior.dim, sampler.multistart, seed=seed
    )
    timings["map_s"] = time.perf_counter() - start
    timings.update({f"map.{k}": v for k, v in map_result.timings.items()})

    start = time.perf_counter()
    run = run_nuts(
        posterior.log_density,
        posterior.dim,
        modes=map_result.modes,
        inverse_mass_matrix=map_result.cov,
        config=sampler.nuts,
        seed=seed + 1,
    )
    timings["nuts_s"] = time.perf_counter() - start

    start = time.perf_counter()
    fit = _postprocess(posterior, map_result.modes, run, sampler, seed, images)
    timings["postprocess_s"] = time.perf_counter() - start
    timings["total_s"] = time.perf_counter() - total_start
    return replace(fit, timings={**timings, **fit.timings})


def decompose(
    images: NoobImageSet,
    models: tuple[ModelSpec | str, ...] = ("point", "point+host"),
    *,
    priors: Priors | None = None,
    sampler: SamplerConfig = SamplerConfig(),
    seed: int = 0,
) -> DecompositionResult:
    """Fit several models and compare them with PSIS-LOO.

    Parameters
    ----------
    images : NoobImageSet
        Prepared images (or an iterable of ``NoobImage``).
    models : tuple
        Model family members to fit.
    priors : Priors, optional
        Shared prior configuration; defaults to :meth:`Priors.from_images`.
    sampler : SamplerConfig
        Pipeline configuration shared by all fits.
    seed : int
        Master seed; each model gets a distinct offset.

    Returns
    -------
    DecompositionResult
        All fits plus the predictive comparison.
    """
    images = images if isinstance(images, NoobImageSet) else NoobImageSet(images)
    priors = priors or Priors.from_images(images)
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    fits: dict[str, ModelFit] = {}
    for index, model in enumerate(models):
        spec = ModelSpec.parse(model)
        fit = fit_model(
            images, spec, priors=priors, sampler=sampler, seed=seed + 101 * index
        )
        fits[spec.name] = fit
        timings[f"fit.{spec.name}_s"] = fit.timings.get("total_s", float("nan"))

    comparison: Comparison | None = None
    metrics = {name: fit.loo for name, fit in fits.items() if fit.loo is not None}
    if metrics:
        comparison = Comparison(metrics=metrics)

    timings["total_s"] = time.perf_counter() - total_start
    return DecompositionResult(fits=fits, comparison=comparison, timings=timings)


def _postprocess(
    posterior: "Posterior",
    modes: tuple[tuple[np.ndarray, float], ...],
    run: Any,
    sampler: SamplerConfig,
    seed: int,
    images: NoobImageSet,
) -> ModelFit:
    """Recover physical/amplitude posteriors, diagnostics and LOO."""
    import jax
    import jax.numpy as jnp

    from noobfriend.inference.morphology import _diagnostics

    chains, draws, dim = run.z.shape
    z_flat = run.z.reshape(-1, dim)
    n_bands = len(posterior.statics)
    n_amps = posterior.spec.n_amplitudes

    named = _batched(
        jax.jit(jax.vmap(lambda z: posterior.transform.physical(z, jnp))),
        (z_flat,),
        sampler.amplitude_batch,
    )
    params = {
        name: np.asarray(values).reshape(chains, draws)
        for name, values in named.items()
    }

    rng = np.random.default_rng(seed + 2)
    eps = rng.standard_normal((z_flat.shape[0], n_bands, n_amps))
    amps = _batched(
        jax.jit(jax.vmap(lambda z, e: posterior.draw_amplitudes(z, e)[0])),
        (z_flat, eps),
        sampler.amplitude_batch,
    )
    amps = np.asarray(amps).reshape(chains, draws, n_bands, n_amps)
    component_names = (*posterior.spec.component_names, "background")
    fluxes = {
        component: {
            band: amps[:, :, band_index, comp_index]
            for band_index, band in enumerate(posterior.band_names)
        }
        for comp_index, component in enumerate(component_names)
    }

    loo = None
    chain_weights = None
    if sampler.compute_loo:
        thin = slice(None, None, max(1, sampler.loo_thin))
        z_thin = run.z[:, thin].reshape(-1, dim)
        eps_thin = eps.reshape(chains, draws, n_bands, n_amps)[:, thin].reshape(
            -1, n_bands, n_amps
        )
        loglik = _batched(
            jax.jit(
                jax.vmap(
                    lambda z, e: posterior.draw_amplitudes(z, e, pointwise=True)[1]
                )
            ),
            (z_thin, eps_thin),
            sampler.amplitude_batch,
        )
        loglik = np.asarray(loglik).reshape(chains, -1, posterior.n_obs)
        loo = _diagnostics.psis_loo(loglik)
        chain_weights = _diagnostics.stacking_weights(loglik)

    health = _diagnostics.sampler_health(params, int(run.diverging.sum()))

    z_median = np.median(z_flat, axis=0)
    amp_median = np.median(amps.reshape(-1, n_bands, n_amps), axis=0)
    rendered = posterior.render(z_median, amp_median)

    return ModelFit(
        spec=posterior.spec,
        params=params,
        fluxes=fluxes,
        z=run.z,
        health=health,
        loo=loo,
        rendered=rendered,
        map_modes=modes,
        timings={},
        images=images,
        chain_weights=chain_weights,
    )


def _batched(fn: Any, arrays: tuple[np.ndarray, ...], batch: int) -> Any:
    """Apply a vmapped function over leading-axis chunks and concatenate."""
    import jax.numpy as jnp

    total = arrays[0].shape[0]
    pieces = []
    for lo in range(0, total, batch):
        chunk = tuple(jnp.asarray(a[lo : lo + batch]) for a in arrays)
        pieces.append(fn(*chunk))
    first = pieces[0]
    if isinstance(first, dict):
        return {
            key: np.concatenate([np.asarray(p[key]) for p in pieces]) for key in first
        }
    return np.concatenate([np.asarray(p) for p in pieces])
