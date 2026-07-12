"""Pointwise PSIS-LOO and WAIC model criteria."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp


_LOG_LIKELIHOOD_VARIABLE = "observed_flux_normalized"
_WAIC_VARIANCE_WARNING_THRESHOLD = 0.4


@dataclass(frozen=True, slots=True, repr=False)
class MCMCPSISLOO:
    """PSIS-LOO expected log predictive density on log scale."""

    elpd: float
    standard_error: float
    effective_parameters: float
    pointwise: np.ndarray
    pareto_k: np.ndarray
    good_k: float
    warning: bool
    n_samples: int
    n_observations: int
    scale: str = "log"

    def __repr__(self) -> str:
        """Return a compact criterion representation."""
        return (
            f"MCMCPSISLOO(elpd={self.elpd:.8g}, standard_error={self.standard_error:.8g}, "
            f"effective_parameters={self.effective_parameters:.8g}, "
            f"max_pareto_k={float(np.max(self.pareto_k)):.8g}, warning={self.warning})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class MCMCWAIC:
    """WAIC expected log predictive density on log scale."""

    elpd: float
    standard_error: float
    effective_parameters: float
    pointwise: np.ndarray
    pointwise_variance: np.ndarray
    warning: bool
    n_samples: int
    n_observations: int
    scale: str = "log"

    def __repr__(self) -> str:
        """Return a compact criterion representation."""
        return (
            f"MCMCWAIC(elpd={self.elpd:.8g}, standard_error={self.standard_error:.8g}, "
            f"effective_parameters={self.effective_parameters:.8g}, warning={self.warning})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class MCMCCriteria:
    """Predictive information criteria computed from pointwise likelihoods."""

    psis_loo: MCMCPSISLOO
    waic: MCMCWAIC

    def __repr__(self) -> str:
        """Return a compact representation of both criteria."""
        return f"MCMCCriteria(psis_loo={self.psis_loo!r}, waic={self.waic!r})"

    def _repr_html_(self) -> str:
        """Return a notebook table of predictive criteria."""
        return f"""<table class="noob-mcmc-criteria">
  <thead><tr><th>Criterion</th><th>ELPD</th><th>SE</th><th>Effective parameters</th><th>Warning</th></tr></thead>
  <tbody>
    <tr><td>PSIS-LOO</td><td>{self.psis_loo.elpd:.8g}</td><td>{self.psis_loo.standard_error:.8g}</td><td>{self.psis_loo.effective_parameters:.8g}</td><td>{self.psis_loo.warning}</td></tr>
    <tr><td>WAIC</td><td>{self.waic.elpd:.8g}</td><td>{self.waic.standard_error:.8g}</td><td>{self.waic.effective_parameters:.8g}</td><td>{self.waic.warning}</td></tr>
  </tbody>
</table>"""


def build_mcmc_criteria(idata: Any) -> MCMCCriteria:
    """Compute pointwise PSIS-LOO and WAIC on a fixed log scale."""
    try:
        import arviz as az
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Spectrum MCMC criteria require the 'mcmc' optional dependency."
        ) from error

    loo = az.loo(
        idata,
        pointwise=True,
        var_name=_LOG_LIKELIHOOD_VARIABLE,
    )
    psis_loo = MCMCPSISLOO(
        elpd=float(loo.elpd),
        standard_error=float(loo.se),
        effective_parameters=float(loo.p),
        pointwise=_readonly(np.asarray(loo.elpd_i, dtype=float).reshape(-1)),
        pareto_k=_readonly(np.asarray(loo.pareto_k, dtype=float).reshape(-1)),
        good_k=float(loo.good_k),
        warning=bool(loo.warning),
        n_samples=int(loo.n_samples),
        n_observations=int(loo.n_data_points),
    )
    waic = _waic(idata)
    return MCMCCriteria(psis_loo=psis_loo, waic=waic)


def _waic(idata: Any) -> MCMCWAIC:
    log_likelihood = np.asarray(
        idata.log_likelihood[_LOG_LIKELIHOOD_VARIABLE],
        dtype=float,
    )
    if log_likelihood.ndim < 3:
        raise ValueError(
            "pointwise log likelihood must contain chain, draw, and observation dimensions."
        )
    n_samples = int(log_likelihood.shape[0] * log_likelihood.shape[1])
    flattened = log_likelihood.reshape(n_samples, -1)
    n_observations = int(flattened.shape[1])
    lppd = logsumexp(flattened, axis=0) - np.log(n_samples)
    pointwise_variance = np.var(flattened, axis=0, ddof=1)
    pointwise = lppd - pointwise_variance
    standard_error = (
        float(np.sqrt(n_observations * np.var(pointwise, ddof=1)))
        if n_observations > 1
        else float("nan")
    )
    return MCMCWAIC(
        elpd=float(np.sum(pointwise)),
        standard_error=standard_error,
        effective_parameters=float(np.sum(pointwise_variance)),
        pointwise=_readonly(pointwise),
        pointwise_variance=_readonly(pointwise_variance),
        warning=bool(np.any(pointwise_variance > _WAIC_VARIANCE_WARNING_THRESHOLD)),
        n_samples=n_samples,
        n_observations=n_observations,
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output
