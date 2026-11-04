"""NUTS sampling of the collapsed latent posterior via NumPyro.

The sampler runs on the bare latent vector through ``potential_fn`` — no
model DSL. Chains start at the deduplicated MAP modes (jittered), and the
dense mass matrix is initialized from the Laplace covariance so warmup
refines an approximately correct metric instead of learning one from scratch.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from noobfriend.core.jaxruntime import Device

__all__ = ["NutsConfig", "NutsRun", "run_nuts"]


@dataclass(frozen=True)
class NutsConfig:
    """NUTS sampler configuration.

    Parameters
    ----------
    chains, warmup, draws : int
        Chain count and per-chain warmup/posterior lengths. The defaults
        (16 chains, 1000 warmup, 500 draws) are chosen for correctness
        first: many chains give rank-normalized diagnostics the power to
        detect the regime-split posteriors of marginal detections that a
        4-chain run can silently miss, and the long warmup is required for
        the adaptation to settle on low-S/N posteriors. On a GPU the 16
        vectorized chains cost about the same wall time as 4 chains of
        double length; on CPU chains share the core budget, so expect
        roughly 2-3x the wall time of a 4-chain run.
    target_accept : float
        Dual-averaging acceptance target. 0.85 is a sensible default for a
        preconditioned, unconstrained posterior; raising it is not a fix for
        divergences (reparameterize instead).
    max_tree_depth : int
        NUTS tree depth cap. 8 (255 leapfrog steps) is generous for the
        preconditioned collapsed posterior; it also bounds the per-iteration
        cost when a mis-specified model leaves prior-plateau directions in
        which trajectories rarely U-turn.
    init_jitter : float
        Standard deviation of the per-chain initial-point jitter, in units
        of the Laplace posterior scale (not the prior scale).
    mode_window : float
        Only MAP modes within this log-density window below the best mode
        are used as chain seeds; lower stopping points are optimizer
        artifacts, not competing modes.
    """

    chains: int = 16
    warmup: int = 1000
    draws: int = 500
    target_accept: float = 0.85
    max_tree_depth: int = 8
    init_jitter: float = 0.05
    mode_window: float = 20.0


@dataclass(frozen=True)
class NutsRun:
    """Raw NUTS output.

    Attributes
    ----------
    z : numpy.ndarray
        Latent draws, shape ``(chains, draws, dim)``.
    diverging : numpy.ndarray
        Boolean divergence flags, shape ``(chains, draws)``.
    groups : tuple of int
        Sizes of the chain groups that ran one after another.
    """

    z: np.ndarray
    diverging: np.ndarray
    groups: tuple[int, ...]


def run_nuts(
    log_density: Callable[[Any], Any],
    dim: int,
    *,
    modes: tuple[tuple[np.ndarray, float], ...],
    inverse_mass_matrix: np.ndarray,
    config: NutsConfig,
    device: Device,
    seed: int = 0,
) -> NutsRun:
    """Sample the latent posterior with NUTS.

    Chains run in the groups :func:`noobfriend.core.jaxruntime.chain_groups`
    gives for ``device``: all at once (vectorized) on a GPU, as many as there
    are CPU devices (in parallel) on the CPU, groups one after another. Each
    group gets its own slice of the chain starts and its own PRNG key.

    Parameters
    ----------
    log_density : callable
        Jitted latent-space log density.
    dim : int
        Latent dimension.
    modes : tuple
        MAP modes (best first); chains are initialized by cycling through
        them with jitter, so distinct modes double as a multimodality probe.
    inverse_mass_matrix : numpy.ndarray
        Initial dense inverse mass matrix (the Laplace covariance).
    config : NutsConfig
        Sampler configuration.
    device : {"cpu", "gpu"}
        The resolved device; the caller makes it JAX's default device.
    seed : int
        PRNG seed for jitter and sampling.

    Returns
    -------
    NutsRun
        Latent draws grouped by chain, divergence flags and chain groups.
    """
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    from noobfriend.core.jaxruntime import chain_groups, chain_method

    rng = np.random.default_rng(seed)
    # Seed chains only from modes genuinely competing with the best one:
    # L-BFGS stopping points far down the log-density are not modes worth
    # probing, and a chain started there spends the whole run climbing out
    # at maximal tree depth.
    best_value = modes[0][1]
    seeds = [z for z, value in modes if value >= best_value - config.mode_window]
    # Jitter in posterior units, not prior units: directions measured much
    # more tightly than the prior (point positions at high SNR) would
    # otherwise start dozens of posterior sigmas into the tails and stall
    # warmup adaptation.
    scale = np.linalg.cholesky(inverse_mass_matrix)
    inits = np.stack(
        [
            seeds[chain % len(seeds)]
            + config.init_jitter * (scale @ rng.standard_normal(dim))
            for chain in range(config.chains)
        ]
    )

    def potential_fn(z: Any) -> Any:
        return -log_density(z)

    kwargs: dict[str, Any] = {
        "potential_fn": potential_fn,
        "target_accept_prob": config.target_accept,
        "max_tree_depth": config.max_tree_depth,
        "dense_mass": True,
    }
    if "inverse_mass_matrix" in inspect.signature(NUTS.__init__).parameters:
        kwargs["inverse_mass_matrix"] = jnp.asarray(inverse_mass_matrix)
    kernel = NUTS(**kwargs)

    groups = chain_groups(config.chains, device)
    z_parts, diverging_parts = [], []
    start = 0
    for index, size in enumerate(groups):
        mcmc = MCMC(
            kernel,
            num_warmup=config.warmup,
            num_samples=config.draws,
            num_chains=size,
            chain_method=chain_method(size, device),
            progress_bar=False,
        )
        block = inits[start : start + size]
        init_params = jnp.asarray(block) if size > 1 else jnp.asarray(block[0])
        # Warmup and sampling are run as separate phases: the combined
        # ``mcmc.run(num_warmup=...)`` path has shown order-of-magnitude
        # slowdowns after the first call in a process (observed with numpyro
        # 0.21), while the explicit split compiles and runs reliably.
        mcmc.warmup(
            jax.random.fold_in(jax.random.PRNGKey(seed), index),
            init_params=init_params,
            collect_warmup=False,
            extra_fields=("diverging",),
        )
        mcmc.run(
            mcmc.post_warmup_state.rng_key,
            extra_fields=("diverging",),
        )
        z = np.asarray(mcmc.get_samples(group_by_chain=True), dtype=np.float64)
        diverging = np.asarray(
            mcmc.get_extra_fields(group_by_chain=True)["diverging"], dtype=bool
        )
        z_parts.append(z.reshape(size, config.draws, dim))
        diverging_parts.append(diverging.reshape(size, config.draws))
        start += size
    return NutsRun(
        z=np.concatenate(z_parts),
        diverging=np.concatenate(diverging_parts),
        groups=groups,
    )
