"""NUTS diagnostics and sampling summary for spectrum MCMC results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class MCMCDivergenceWarning(UserWarning):
    """Warning emitted when post-tuning NUTS divergences are present."""


@dataclass(frozen=True, slots=True)
class MCMCDiagnostics:
    """Chain-level and aggregate NUTS diagnostics."""

    divergences: int
    divergence_rate: float
    divergences_by_chain: tuple[int, ...]
    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    bfmi_by_chain: tuple[float, ...]
    min_bfmi: float
    max_tree_depth: int
    mean_tree_depth: float
    tree_depth_limit_count: int
    max_n_steps: int
    mean_n_steps: float
    mean_acceptance_rate: float


@dataclass(frozen=True, slots=True)
class MCMCSampling:
    """Elapsed sampling time and diagnostics for one NUTS run."""

    elapsed_seconds: float
    diagnostics: MCMCDiagnostics


def build_mcmc_diagnostics(
    idata: Any, variable_names: tuple[str, ...]
) -> MCMCDiagnostics:
    """Compute aggregate diagnostics for the model's free random variables."""
    try:
        import arviz as az
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Spectrum MCMC diagnostics require the 'mcmc' optional dependency."
        ) from error

    summary = az.summary(
        idata,
        var_names=list(variable_names),
        kind="diagnostics",
        round_to="none",
    )
    sample_stats = idata.sample_stats
    divergences = np.asarray(sample_stats["diverging"], dtype=bool)
    tree_depth = np.asarray(sample_stats["tree_depth"], dtype=int)
    n_steps = np.asarray(sample_stats["n_steps"], dtype=int)
    acceptance_rate = np.asarray(sample_stats["acceptance_rate"], dtype=float)
    energy = np.asarray(sample_stats["energy"], dtype=float)
    energy_variance = np.var(energy, axis=1, ddof=1)
    bfmi = np.divide(
        np.mean(np.diff(energy, axis=1) ** 2, axis=1),
        energy_variance,
        out=np.full_like(energy_variance, np.nan),
        where=energy_variance > 0.0,
    )
    if "reached_max_treedepth" in sample_stats:
        tree_depth_limit_count = int(
            np.asarray(sample_stats["reached_max_treedepth"], dtype=bool).sum()
        )
    else:
        tree_depth_limit_count = int(np.sum(tree_depth >= 10))
    return MCMCDiagnostics(
        divergences=int(divergences.sum()),
        divergence_rate=float(divergences.mean()),
        divergences_by_chain=tuple(int(value) for value in divergences.sum(axis=1)),
        max_rhat=float(summary["r_hat"].max()),
        min_ess_bulk=float(summary["ess_bulk"].min()),
        min_ess_tail=float(summary["ess_tail"].min()),
        bfmi_by_chain=tuple(float(value) for value in bfmi),
        min_bfmi=float(np.nanmin(bfmi)),
        max_tree_depth=int(tree_depth.max()),
        mean_tree_depth=float(tree_depth.mean()),
        tree_depth_limit_count=tree_depth_limit_count,
        max_n_steps=int(n_steps.max()),
        mean_n_steps=float(n_steps.mean()),
        mean_acceptance_rate=float(acceptance_rate.mean()),
    )
