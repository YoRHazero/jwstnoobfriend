"""Predictive model-comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noobfriend.inference.morphology.result import FitResult, PredictiveMetric


@dataclass(frozen=True)
class LOOComparison:
    """PSIS-LOO values for a set of named fit results."""

    metrics: dict[str, PredictiveMetric]

    def best(self) -> str:
        """Return the valid model with the largest ELPD-LOO."""
        valid = {
            name: metric
            for name, metric in self.metrics.items()
            if metric.valid and metric.value is not None
        }
        if not valid:
            raise ValueError("no valid PSIS-LOO metric is available.")
        return max(valid, key=lambda name: valid[name].value or float("-inf"))


def psis_loo(result: FitResult, *, name: str = "psis_loo") -> PredictiveMetric:
    """Compute PSIS-LOO from recorded pointwise log likelihood."""
    if result.log_likelihood is None:
        return PredictiveMetric(
            name=name,
            value=None,
            valid=False,
            reason="FitResult.log_likelihood is missing.",
        )
    try:
        import arviz as az
    except Exception as exc:
        return PredictiveMetric(
            name=name,
            value=None,
            valid=False,
            reason=f"ArviZ is required for PSIS-LOO: {exc}",
        )

    log_likelihood = np.asarray(result.log_likelihood)
    if log_likelihood.ndim != 3:
        return PredictiveMetric(
            name=name,
            value=None,
            valid=False,
            reason="log_likelihood must have shape (chain, draw, observation).",
        )

    inference_data = az.from_dict(
        {
            "posterior": {"_dummy": np.zeros(log_likelihood.shape[:2])},
            "log_likelihood": {"obs": log_likelihood},
        }
    )
    loo = az.loo(inference_data, var_name="obs", pointwise=True)
    elpd = float(loo.elpd)
    p_loo = float(loo.p)
    pareto_k = np.asarray(getattr(loo, "pareto_k", []), dtype=float)
    high_k = int(np.sum(pareto_k > 0.7)) if pareto_k.size else 0
    return PredictiveMetric(
        name=name,
        value=elpd,
        valid=high_k == 0,
        reason=None if high_k == 0 else f"{high_k} observations have Pareto k > 0.7.",
        details={
            "elpd_loo": elpd,
            "se": float(loo.se),
            "p_loo": p_loo,
            "high_pareto_k": high_k,
        },
    )


def compare_psis_loo(results: dict[str, FitResult]) -> LOOComparison:
    """Compute PSIS-LOO for each named fit result."""
    return LOOComparison(
        metrics={name: psis_loo(result) for name, result in results.items()}
    )
