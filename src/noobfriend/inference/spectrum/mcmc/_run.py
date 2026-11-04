"""Run NUTS chains in device-sized groups and assemble the InferenceData."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.core.jaxruntime import chain_groups, chain_method, enable_x64, on_cpu
from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.mcmc._names import LATENT, site_frame

if TYPE_CHECKING:
    import xarray as xr

    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace

#: Posterior draws evaluated at once when building correlated-noise LOO terms.
_CHUNK = 1000


@dataclass(frozen=True, slots=True)
class SamplingRun:
    """Raw output of :func:`run_chains`.

    Attributes
    ----------
    posterior
        Every sampled and deterministic site, ``(chain, draw, ...)``.
    sample_stats
        NUTS statistics (divergences, energy, tree depth, ...).
    samples
        The same draws as NumPy arrays by site, for log-likelihood replay.
    elapsed_seconds
        Wall time of the chains.
    groups
        Number of chains run side by side in each group.
    """

    posterior: xr.Dataset
    sample_stats: xr.Dataset
    samples: Mapping[str, np.ndarray]
    elapsed_seconds: float
    groups: tuple[int, ...]


def trace_model(model: Callable[[], None]) -> Mapping[str, Any]:
    """Trace a model once at a valid initial point.

    Returns
    -------
    Mapping
        NumPyro's site trace (name to site dict).
    """
    import jax
    from numpyro.infer.util import initialize_model

    enable_x64()
    with on_cpu():
        return initialize_model(jax.random.PRNGKey(0), model).model_trace


def likelihood_sites(trace: Mapping[str, Any]) -> tuple[str, ...]:
    """Observed sample sites that carry a real distribution (no factors)."""
    from numpyro.distributions.distribution import Unit

    return tuple(
        name
        for name, site in trace.items()
        if site["type"] == "sample"
        and site["is_observed"]
        and not isinstance(site["fn"], Unit)
        and not site.get("infer", {}).get("is_auxiliary", False)
    )


def run_chains(
    model: Callable[[], None],
    *,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    random_seed: int,
    progressbar: bool,
) -> SamplingRun:
    """Run NUTS (dense mass matrix) on the CPU.

    Chains are grouped by the number of CPU devices JAX exposes: each group
    runs its chains in parallel and the groups run one after another, each
    with its own key folded from ``random_seed``.
    """
    import arviz as az
    import jax
    import xarray as xr
    from numpyro.infer import MCMC, NUTS

    enable_x64()
    posterior, stats, samples, groups = [], [], [], []
    start = time.perf_counter()
    with on_cpu():
        for index, size in enumerate(chain_groups(chains, "cpu")):
            mcmc = MCMC(
                NUTS(model, dense_mass=True, target_accept_prob=target_accept),
                num_warmup=tune,
                num_samples=draws,
                num_chains=size,
                chain_method=chain_method(size, "cpu"),
                progress_bar=progressbar,
            )
            key = jax.random.fold_in(jax.random.PRNGKey(random_seed), index)
            mcmc.run(key, extra_fields=("diverging", "energy", "num_steps"))
            tree = az.from_numpyro(mcmc)
            posterior.append(tree["posterior"].dataset)
            stats.append(tree["sample_stats"].dataset)
            samples.append(
                {
                    k: np.asarray(v)
                    for k, v in mcmc.get_samples(group_by_chain=True).items()
                }
            )
            groups.append(size)
    elapsed = time.perf_counter() - start
    chain = np.arange(chains)
    merged = {
        key: np.concatenate([part[key] for part in samples]) for key in samples[0]
    }
    return SamplingRun(
        posterior=xr.concat(posterior, dim="chain").assign_coords(chain=chain),
        sample_stats=xr.concat(stats, dim="chain").assign_coords(chain=chain),
        samples=merged,
        elapsed_seconds=elapsed,
        groups=tuple(groups),
    )


def log_likelihood_dataset(
    workspace: NoobFitWorkspace,
    model: Callable[[], None],
    samples: Mapping[str, np.ndarray],
    sites: tuple[str, ...],
    *,
    replaced: bool,
) -> xr.Dataset:
    """Pointwise log likelihoods for PSIS-LOO.

    ``pixels`` (``chain, draw, pixel``) holds the observation sites' own
    values for white noise, and the exact conditional ``log p(y_i | y_-i)``
    for correlated noise in the library's model. A site that only gives one
    value per frame (correlated noise in a replaced model) contributes that
    single value.

    There is no frame-level term: PSIS over a frame's joint log likelihood
    cannot represent the leave-that-frame-out posterior of the frame's own
    continuum and noise parameters, so its estimates would be biased and
    unreliable (Pareto-k near 1 even on clean data).
    """
    import numpyro
    import xarray as xr

    with on_cpu():
        values = numpyro.infer.log_likelihood(model, dict(samples), batch_ndims=2)
    chains, draws = next(iter(samples.values())).shape[:2]
    pixel_parts, pixel_frames, pixel_index = [], [], []
    for site in sites:
        site_values = np.asarray(values[site], dtype=float).reshape(chains, draws, -1)
        label = site_frame(site)
        noise = workspace.frame_noise.get(label)
        if noise is not None and not replaced and noise.kind != "white":
            pointwise = _correlated_pointwise(workspace, samples, label)
        else:
            pointwise = site_values
        native = np.full(pointwise.shape[-1], -1)
        if noise is not None and pointwise.shape[-1] == noise.size:
            native = np.flatnonzero(workspace.fitted[label])
        pixel_parts.append(pointwise)
        pixel_frames += [label] * pointwise.shape[-1]
        pixel_index.append(native)
    return xr.Dataset(
        {
            "pixels": (
                ("chain", "draw", "pixel"),
                np.concatenate(pixel_parts, axis=-1),
            ),
        },
        coords={
            "chain": np.arange(chains),
            "draw": np.arange(draws),
            "pixel_frame": ("pixel", pixel_frames),
            "pixel_index": ("pixel", np.concatenate(pixel_index)),
        },
    )


def _correlated_pointwise(
    workspace: NoobFitWorkspace, samples: Mapping[str, np.ndarray], frame_id: str
) -> np.ndarray:
    """Exact leave-one-out conditionals of one correlated-noise frame."""
    xp = numpy_namespace()
    frame = workspace.spectrum.frames[frame_id]
    mask = workspace.fitted[frame_id]
    flux = np.asarray(frame.flux)[mask]
    noise = workspace.frame_noise[frame_id]
    chains, draws = next(iter(samples.values())).shape[:2]
    flat = {
        name: values.reshape(chains * draws, *values.shape[2:])
        for name, values in samples.items()
        if not name.endswith(LATENT)
    }
    rho_name = f"frame.{frame_id}.noise.rho"
    out = np.empty((chains * draws, noise.size))
    for start in range(0, chains * draws, _CHUNK):
        part = slice(start, start + _CHUNK)
        values = {name: array[part][:, None] for name, array in flat.items()}
        total = workspace.frame_model(xp, values, frame_id).total
        scale = flat[f"frame.{frame_id}.noise.scale"][part]
        rho = flat[rho_name][part] if rho_name in flat else None
        out[part] = noise.pointwise(xp, flux - total, scale, rho)
    return out.reshape(chains, draws, noise.size)
