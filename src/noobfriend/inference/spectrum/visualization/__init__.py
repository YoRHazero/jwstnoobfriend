"""Static visualization adapters for spectrum fit results."""

from noobfriend.inference.spectrum.visualization.diagnostics import (
    plot_mcmc_corner,
    plot_mcmc_pareto_k,
)
from noobfriend.inference.spectrum.visualization.fit import (
    plot_mcmc_fit,
    plot_mle_fit,
)

__all__ = [
    "plot_mcmc_corner",
    "plot_mcmc_fit",
    "plot_mcmc_pareto_k",
    "plot_mle_fit",
]
