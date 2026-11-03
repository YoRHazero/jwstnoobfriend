"""Sampling diagnostics and the fixed convergence verdict.

A run converged only if **all** of these hold (fixed thresholds):

- no divergent transition;
- max rank-normalized split R-hat < 1.01 over every non-constant physical
  parameter;
- bulk and tail ESS >= 400 for each of them;
- E-BFMI > 0.3 in every chain.

Anything else is not a result to report. If divergences remain, raise
``target_accept`` and *check ESS again*: smaller steps can silently starve
it. A model that will not converge after reparameterization and tuning is
best reported as not identifiable on these data.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

RHAT_MAX = 1.01
ESS_MIN = 400.0
BFMI_MIN = 0.3


class MCMCConvergenceWarning(UserWarning):
    """A sampling run failed at least one convergence criterion."""


@dataclass(frozen=True, slots=True)
class ParameterDiagnostics:
    """Convergence diagnostics of one physical parameter.

    Attributes
    ----------
    rhat
        Rank-normalized split R-hat.
    ess_bulk, ess_tail
        Effective sample sizes.
    """

    rhat: float
    ess_bulk: float
    ess_tail: float


@dataclass(frozen=True, slots=True)
class ConvergenceVerdict:
    """Outcome of the fixed convergence criteria.

    Attributes
    ----------
    converged
        Whether every criterion passed.
    failures
        One message per failed criterion.
    divergences
        Number of divergent transitions over all chains.
    max_rhat, min_ess_bulk, min_ess_tail
        Extremes over non-constant physical parameters.
    min_bfmi
        Smallest per-chain E-BFMI.
    """

    converged: bool
    failures: tuple[str, ...]
    divergences: int
    max_rhat: float
    min_ess_bulk: float
    min_ess_tail: float
    min_bfmi: float


@dataclass(frozen=True, slots=True)
class SamplingReport:
    """How a posterior was sampled and whether it converged.

    Attributes
    ----------
    options
        ``draws``, ``tune``, ``chains``, ``target_accept``, ``random_seed``,
        and the chain groups actually run side by side.
    elapsed_seconds
        Wall time of sampling.
    diagnostics
        Per-parameter diagnostics of the non-constant physical parameters.
    verdict
        The convergence verdict.
    reparameterizations
        The internal reparameterizations used, one description each.
    replaced_model
        Whether the model was swapped with ``with_numpyro_model``.
    """

    options: Mapping[str, Any]
    elapsed_seconds: float
    diagnostics: Mapping[str, ParameterDiagnostics]
    verdict: ConvergenceVerdict
    reparameterizations: tuple[str, ...]
    replaced_model: bool


def diagnose(idata: Any) -> tuple[dict[str, ParameterDiagnostics], ConvergenceVerdict]:
    """Compute diagnostics and the verdict from an InferenceData tree.

    Parameters
    ----------
    idata
        ArviZ ``DataTree`` with ``posterior`` and ``sample_stats`` groups.

    Returns
    -------
    diagnostics : dict
        Per-parameter diagnostics.
    verdict : ConvergenceVerdict
        The verdict.
    """
    import arviz as az

    posterior = idata["posterior"].dataset
    varying = [
        name
        for name, values in posterior.data_vars.items()
        if np.all(np.isfinite(values.values)) and np.ptp(values.values) > 0
    ]
    diagnostics: dict[str, ParameterDiagnostics] = {}
    if varying:
        rhat = az.rhat(idata, var_names=varying)
        bulk = az.ess(idata, var_names=varying, method="bulk")
        tail = az.ess(idata, var_names=varying, method="tail")
        for name in varying:
            diagnostics[name] = ParameterDiagnostics(
                rhat=_scalar(rhat, name),
                ess_bulk=_scalar(bulk, name),
                ess_tail=_scalar(tail, name),
            )
    stats = idata["sample_stats"].dataset
    divergences = int(np.sum(stats["diverging"].values)) if "diverging" in stats else 0
    bfmi = np.atleast_1d(np.asarray(_values(az.bfmi(idata), "energy"), dtype=float))
    max_rhat = max((d.rhat for d in diagnostics.values()), default=float("nan"))
    min_bulk = min((d.ess_bulk for d in diagnostics.values()), default=float("nan"))
    min_tail = min((d.ess_tail for d in diagnostics.values()), default=float("nan"))
    min_bfmi = float(np.min(bfmi))

    failures = []
    if divergences:
        failures.append(f"{divergences} divergent transitions (need 0)")
    worst = [name for name, d in diagnostics.items() if not d.rhat < RHAT_MAX]
    if worst:
        failures.append(f"R-hat >= {RHAT_MAX} for {_list(worst)} (max {max_rhat:.4f})")
    low_bulk = [name for name, d in diagnostics.items() if not d.ess_bulk >= ESS_MIN]
    if low_bulk:
        failures.append(
            f"bulk ESS < {ESS_MIN:g} for {_list(low_bulk)} (min {min_bulk:.0f})"
        )
    low_tail = [name for name, d in diagnostics.items() if not d.ess_tail >= ESS_MIN]
    if low_tail:
        failures.append(
            f"tail ESS < {ESS_MIN:g} for {_list(low_tail)} (min {min_tail:.0f})"
        )
    if not min_bfmi > BFMI_MIN:
        failures.append(f"E-BFMI {min_bfmi:.3f} <= {BFMI_MIN} in some chain")
    verdict = ConvergenceVerdict(
        converged=not failures,
        failures=tuple(failures),
        divergences=divergences,
        max_rhat=max_rhat,
        min_ess_bulk=min_bulk,
        min_ess_tail=min_tail,
        min_bfmi=min_bfmi,
    )
    return diagnostics, verdict


def warn_if_unconverged(verdict: ConvergenceVerdict) -> None:
    """Emit one :class:`MCMCConvergenceWarning` listing every failure."""
    if verdict.converged:
        return
    warnings.warn(
        "MCMC did not converge: "
        + "; ".join(verdict.failures)
        + ". Do not report these posteriors. Reparameterize or tighten priors; "
        "if raising target_accept, check ESS again.",
        MCMCConvergenceWarning,
        stacklevel=3,
    )


def frozen(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only copy of ``mapping``."""
    return MappingProxyType(dict(mapping))


def _values(tree: Any, name: str) -> Any:
    dataset = tree.dataset if hasattr(tree, "dataset") else tree
    return dataset[name].values


def _scalar(tree: Any, name: str) -> float:
    return float(np.asarray(_values(tree, name)))


def _list(names: list[str]) -> str:
    shown = ", ".join(names[:5])
    return shown if len(names) <= 5 else f"{shown} and {len(names) - 5} more"
