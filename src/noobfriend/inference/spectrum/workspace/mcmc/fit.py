"""PyMC sampling workflow for prepared spectrum workspaces."""

from __future__ import annotations

import warnings
from time import perf_counter
from typing import TYPE_CHECKING

from noobfriend.inference.spectrum.workspace.mcmc.diagnostics import (
    MCMCDivergenceWarning,
)
from noobfriend.inference.spectrum.workspace.mcmc.options import MCMCOptions
from noobfriend.inference.spectrum.workspace.mcmc.result import (
    MCMCFitResult,
    build_mcmc_result,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.model import NoobSpectrumModel


def sample_spectrum_model(
    model: NoobSpectrumModel, *, options: MCMCOptions
) -> MCMCFitResult:
    """Sample an already-built spectrum model using automatic initial points."""
    try:
        import pymc as pm
    except ImportError as error:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Spectrum MCMC requires the 'mcmc' optional dependency."
        ) from error

    started = perf_counter()
    with model.pymc_model:
        idata = pm.sample(
            draws=options.draws,
            tune=options.tune,
            chains=options.chains,
            cores=options.cores,
            target_accept=options.target_accept,
            random_seed=options.random_seed,
            progressbar=options.progressbar,
            compute_convergence_checks=True,
            return_inferencedata=True,
        )
        elapsed = perf_counter() - started
        idata = pm.compute_log_likelihood(
            idata,
            model=model.pymc_model,
            progressbar=options.progressbar,
        )
    result = build_mcmc_result(
        model.workspace,
        idata,
        model._metadata,
        options,
        elapsed_seconds=elapsed,
    )
    diagnostics = result.sampling.diagnostics
    if diagnostics.divergences:
        warnings.warn(
            f"NUTS reported {diagnostics.divergences} post-tuning divergences; "
            "inspect the posterior geometry and consider target_accept=0.99.",
            MCMCDivergenceWarning,
            stacklevel=3,
        )
    return result
