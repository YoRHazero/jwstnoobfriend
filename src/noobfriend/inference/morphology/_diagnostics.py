"""Convergence diagnostics and predictive model comparison.

Diagnostics follow Vehtari et al. (2021): rank-normalized split R-hat and
bulk/tail effective sample sizes, computed over the *physical* parameters
only (never over pointwise log-likelihood arrays). Model comparison uses
PSIS-LOO (Vehtari, Gelman & Gabry 2017) on the conditional pointwise
log-likelihood.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from noobfriend.inference.morphology.results import (
    Comparison,
    PredictiveMetric,
    SamplerHealth,
)

__all__ = ["compare", "psis_loo", "sampler_health", "stacking_weights"]

RHAT_THRESHOLD = 1.01
ESS_THRESHOLD = 400.0


def sampler_health(params: dict[str, np.ndarray], divergences: int) -> SamplerHealth:
    """Compute convergence diagnostics over named physical parameters.

    Parameters
    ----------
    params : dict of str to numpy.ndarray
        Physical parameter draws, each of shape ``(chains, draws)``.
    divergences : int
        Total divergence count across chains.

    Returns
    -------
    SamplerHealth
        Rank-normalized diagnostics plus threshold-violation messages.
    """
    import arviz as az

    idata = az.from_dict(
        {"posterior": {name: np.asarray(v) for name, v in params.items()}}
    )
    rhat = _by_var(az.rhat(idata, method="rank"))
    ess_bulk = _by_var(az.ess(idata, method="bulk"))
    ess_tail = _by_var(az.ess(idata, method="tail"))
    max_rhat = max(rhat.values())
    min_bulk = min(ess_bulk.values())
    min_tail = min(ess_tail.values())

    messages: list[str] = []
    if max_rhat > RHAT_THRESHOLD:
        worst = max(rhat, key=rhat.get)
        messages.append(f"max rank-normalized R-hat {max_rhat:.3f} ({worst}).")
    if min_bulk < ESS_THRESHOLD:
        worst = min(ess_bulk, key=ess_bulk.get)
        messages.append(f"min bulk ESS {min_bulk:.0f} ({worst}).")
    if min_tail < ESS_THRESHOLD:
        worst = min(ess_tail, key=ess_tail.get)
        messages.append(f"min tail ESS {min_tail:.0f} ({worst}).")
    if divergences > 0:
        messages.append(f"{divergences} divergent transitions.")
    return SamplerHealth(
        max_rhat=max_rhat,
        min_ess_bulk=min_bulk,
        min_ess_tail=min_tail,
        divergences=divergences,
        messages=tuple(messages),
    )


def _by_var(tree: Any) -> dict[str, float]:
    """Flatten an ArviZ diagnostic result into ``{parameter: value}``."""
    return {
        str(name): float(np.asarray(values).max())
        for name, values in tree.data_vars.items()
    }


def psis_loo(log_likelihood: np.ndarray, *, name: str = "psis_loo") -> PredictiveMetric:
    """Compute PSIS-LOO from pointwise conditional log-likelihood draws.

    Parameters
    ----------
    log_likelihood : numpy.ndarray
        Shape ``(chains, draws, observations)``.
    name : str
        Metric name.

    Returns
    -------
    PredictiveMetric
        ELPD-LOO with validity gated on Pareto-k diagnostics.
    """
    import arviz as az

    log_likelihood = np.asarray(log_likelihood)
    if log_likelihood.ndim != 3:
        return PredictiveMetric(
            name=name,
            value=None,
            valid=False,
            reason="log_likelihood must have shape (chain, draw, observation).",
        )
    idata = az.from_dict(
        {
            "posterior": {"_dummy": np.zeros(log_likelihood.shape[:2])},
            "log_likelihood": {"obs": log_likelihood},
        }
    )
    loo = az.loo(idata, var_name="obs", pointwise=True)
    pareto_k = np.asarray(getattr(loo, "pareto_k", []), dtype=float)
    high_k = int(np.sum(pareto_k > 0.7)) if pareto_k.size else 0
    return PredictiveMetric(
        name=name,
        value=float(loo.elpd),
        valid=high_k == 0,
        reason=None if high_k == 0 else f"{high_k} observations have Pareto k > 0.7.",
        details={
            "elpd_loo": float(loo.elpd),
            "se": float(loo.se),
            "p_loo": float(loo.p),
            "high_pareto_k": float(high_k),
        },
    )


def compare(metrics: dict[str, PredictiveMetric]) -> Comparison:
    """Bundle per-model PSIS-LOO metrics into a comparison."""
    return Comparison(metrics=dict(metrics))


def stacking_weights(log_likelihood: np.ndarray) -> np.ndarray | None:
    """Compute per-chain stacking weights (Yao, Vehtari & Gelman 2022).

    Each chain is treated as one approximate posterior; its pointwise
    PSIS-LOO predictive density is computed from that chain's draws alone,
    and simplex weights maximizing the stacked LOO predictive density are
    solved by convex optimization. Chains stuck in configurations that
    predict the data poorly receive weight ~0, so the stacked mixture
    weighs posterior regimes by predictive quality rather than by how many
    chains happened to land in each.

    Parameters
    ----------
    log_likelihood : numpy.ndarray
        Pointwise conditional log-likelihood, shape
        ``(chains, draws, observations)``.

    Returns
    -------
    numpy.ndarray or None
        Simplex weights of shape ``(chains,)``, or ``None`` when the
        computation is unavailable (single chain or ArviZ failure).
    """
    import arviz as az
    from scipy.special import logsumexp

    log_likelihood = np.asarray(log_likelihood)
    n_chains = log_likelihood.shape[0]
    if n_chains < 2:
        return np.ones(max(n_chains, 1))
    try:
        elpd_rows = []
        for chain in range(n_chains):
            idata = az.from_dict(
                {
                    "posterior": {"_dummy": np.zeros((1, log_likelihood.shape[1]))},
                    "log_likelihood": {"obs": log_likelihood[chain : chain + 1]},
                }
            )
            loo = az.loo(idata, var_name="obs", pointwise=True)
            elpd_rows.append(np.asarray(loo.elpd_i))
    except Exception:
        return None
    elpd = np.stack(elpd_rows)  # (chains, obs)
    elpd = elpd - elpd.max(axis=0, keepdims=True)

    # Mixture-weight MLE via EM: monotone, deterministic, no optimizer knobs.
    weights = np.full(n_chains, 1.0 / n_chains)
    for _ in range(500):
        log_joint = elpd + np.log(np.clip(weights, 1e-300, None))[:, None]
        responsibilities = np.exp(log_joint - logsumexp(log_joint, axis=0))
        updated = responsibilities.mean(axis=1)
        if np.max(np.abs(updated - weights)) < 1e-10:
            weights = updated
            break
        weights = updated
    return weights / weights.sum()
