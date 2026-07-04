"""MAP estimation: Sobol multistart, L-BFGS refinement, Laplace curvature.

The initializer works entirely in the latent standard-normal space, where the
prior is isotropic and Sobol points mapped through the normal quantile
function are genuine low-discrepancy prior draws. The Laplace covariance at
the best mode seeds the NUTS dense mass matrix, so warmup starts with an
approximately correct metric instead of discovering it from identity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

__all__ = ["MapResult", "MultistartConfig", "find_map"]


@dataclass(frozen=True)
class MultistartConfig:
    """Configuration for the deterministic multistart MAP search.

    Parameters
    ----------
    n_starts : int
        Number of Sobol prior draws scored with the collapsed posterior.
    n_refine : int
        Number of top-scoring starts refined with L-BFGS.
    max_iter : int
        L-BFGS iteration cap per start.
    dedupe_distance : float
        Latent-space distance below which two converged modes are merged.
    """

    n_starts: int = 128
    n_refine: int = 8
    max_iter: int = 200
    dedupe_distance: float = 1.0


@dataclass(frozen=True)
class MapResult:
    """Multistart MAP output.

    Attributes
    ----------
    modes : tuple
        ``(z, log_density)`` pairs of distinct converged modes, best first.
    cov : numpy.ndarray
        Laplace covariance at the best mode (falls back to the identity when
        the Hessian is not usable).
    timings : dict of str to float
        Stage wall times in seconds.
    """

    modes: tuple[tuple[np.ndarray, float], ...]
    cov: np.ndarray
    timings: dict[str, float]

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Return the best ``(z, log_density)`` mode."""
        return self.modes[0]


def find_map(
    log_density: Callable[[Any], Any],
    dim: int,
    config: MultistartConfig = MultistartConfig(),
    *,
    seed: int = 0,
) -> MapResult:
    """Run the Sobol multistart MAP search with Laplace curvature.

    Parameters
    ----------
    log_density : callable
        Jitted latent-space log density.
    dim : int
        Latent dimension.
    config : MultistartConfig
        Search configuration.
    seed : int
        Seed for the scrambled Sobol sequence.

    Returns
    -------
    MapResult
        Deduplicated modes (best first) and the Laplace covariance.
    """
    import jax
    import jax.numpy as jnp

    timings: dict[str, float] = {}

    start = time.perf_counter()
    sobol = qmc.Sobol(d=dim, scramble=True, rng=np.random.default_rng(seed))
    quantiles = sobol.random(config.n_starts)
    starts = norm.ppf(np.clip(quantiles, 1e-9, 1 - 1e-9))
    starts = np.vstack([np.zeros(dim), starts])
    scores = _score_batched(log_density, starts)
    timings["multistart_score_s"] = time.perf_counter() - start

    start = time.perf_counter()
    neg = jax.jit(jax.value_and_grad(lambda z: -log_density(z)))

    def objective(z: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = neg(jnp.asarray(z))
        return float(value), np.asarray(grad, dtype=np.float64)

    order = np.argsort(-scores)
    finite = [i for i in order if np.isfinite(scores[i])][: config.n_refine]
    refined: list[tuple[np.ndarray, float]] = []
    for index in finite:
        opt = minimize(
            objective,
            starts[index],
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": config.max_iter},
        )
        if np.all(np.isfinite(opt.x)) and np.isfinite(opt.fun):
            refined.append((np.asarray(opt.x, dtype=np.float64), -float(opt.fun)))
    if not refined:
        raise RuntimeError("multistart MAP search found no finite optimum.")
    timings["multistart_refine_s"] = time.perf_counter() - start

    refined.sort(key=lambda item: -item[1])
    modes: list[tuple[np.ndarray, float]] = []
    for z, value in refined:
        if all(np.linalg.norm(z - kept) >= config.dedupe_distance for kept, _ in modes):
            modes.append((z, value))

    start = time.perf_counter()
    cov = _laplace_cov(log_density, modes[0][0], dim)
    timings["laplace_s"] = time.perf_counter() - start

    return MapResult(modes=tuple(modes), cov=cov, timings=timings)


def _score_batched(
    log_density: Callable[[Any], Any], starts: np.ndarray, *, batch: int = 32
) -> np.ndarray:
    """Score start points with a vmapped log density in memory-safe chunks."""
    import jax
    import jax.numpy as jnp

    scorer = jax.jit(jax.vmap(log_density))
    scores = []
    for lo in range(0, starts.shape[0], batch):
        scores.append(np.asarray(scorer(jnp.asarray(starts[lo : lo + batch]))))
    return np.concatenate(scores)


def _laplace_cov(
    log_density: Callable[[Any], Any], z_map: np.ndarray, dim: int
) -> np.ndarray:
    """Return a positive-definite Laplace covariance, or the identity."""
    import jax
    import jax.numpy as jnp

    try:
        hess = np.asarray(jax.hessian(lambda z: -log_density(z))(jnp.asarray(z_map)))
        hess = 0.5 * (hess + hess.T)
        eigvals, eigvecs = np.linalg.eigh(hess)
        if not np.all(np.isfinite(eigvals)):
            return np.eye(dim)
        floor = max(1e-8, 1e-8 * float(eigvals.max()))
        eigvals = np.clip(eigvals, floor, None)
        return (eigvecs / eigvals) @ eigvecs.T
    except Exception:
        return np.eye(dim)
