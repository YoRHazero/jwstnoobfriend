"""Top-level composition of a spectrum MCMC result."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import TYPE_CHECKING, Any, Literal

from noobfriend.inference.spectrum.workspace.mcmc.criteria import (
    MCMCCriteria,
    build_mcmc_criteria,
)
from noobfriend.inference.spectrum.workspace.mcmc.diagnostics import (
    MCMCSampling,
    build_mcmc_diagnostics,
)
from noobfriend.inference.spectrum.workspace.mcmc.posterior import (
    MCMCPosterior,
    build_mcmc_posterior,
)
from noobfriend.inference.spectrum.workspace.mcmc.priors import MCMCPriors

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

    from noobfriend.inference.spectrum.data import NoobSpectrum
    from noobfriend.inference.spectrum.data.types import DispersionAxis
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace
    from noobfriend.inference.spectrum.workspace.mcmc.model import MCMCModelMetadata
    from noobfriend.inference.spectrum.workspace.mcmc.options import MCMCOptions


@dataclass(frozen=True, slots=True)
class MCMCInputs:
    """Input data, declarations, sampling options, and translated priors."""

    data: NoobSpectrum
    workspace: NoobFitWorkspace
    options: MCMCOptions
    priors: MCMCPriors


@dataclass(frozen=True, slots=True, repr=False)
class MCMCFitResult:
    """Posterior, provenance, sampling diagnostics, criteria, and raw inference data."""

    posterior: MCMCPosterior
    inputs: MCMCInputs
    sampling: MCMCSampling
    criteria: MCMCCriteria
    idata: Any

    def plot(
        self,
        *,
        hdi_probability: float = 0.94,
        posterior_draws: int | None = 1000,
        random_seed: int = 1729,
        show_residuals: bool = True,
        model_oversample: int = 8,
        flux_2d: ArrayLike | None = None,
        spatial: ArrayLike | None = None,
        dispersion: DispersionAxis | None = None,
        spatial_window: tuple[float, float] | None = None,
        component_colors: Mapping[str, str] | None = None,
        data_color: str = "#1A1A1A",
        continuum_color: str = "#7F7F7F",
        total_color: str = "#D62728",
        cmap: str = "magma",
        pmin: float = 1.0,
        pmax: float = 99.0,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        ylog: bool = False,
        size: int = 1000,
        title: str | None = None,
        legend_location: str = "best",
        view: Literal["observed", "emergent", "intrinsic"] = "observed",
    ) -> Figure:
        """Plot posterior model summaries and optional residuals in flux units.

        ``view`` selects the convolution layers of the component curves
        (observed = kernels + LSF, emergent = kernels only, intrinsic =
        bare profile); the total model and residuals stay observed.
        """
        from noobfriend.inference.spectrum.visualization import plot_mcmc_fit

        return plot_mcmc_fit(
            self,
            hdi_probability=hdi_probability,
            posterior_draws=posterior_draws,
            random_seed=random_seed,
            show_residuals=show_residuals,
            model_oversample=model_oversample,
            flux_2d=flux_2d,
            spatial=spatial,
            dispersion=dispersion,
            spatial_window=spatial_window,
            component_colors=component_colors,
            data_color=data_color,
            continuum_color=continuum_color,
            total_color=total_color,
            cmap=cmap,
            pmin=pmin,
            pmax=pmax,
            xlim=xlim,
            ylim=ylim,
            ylog=ylog,
            size=size,
            title=title,
            legend_location=legend_location,
            view=view,
        )

    def plot_corner(
        self,
        *,
        variables: Sequence[tuple[str, str]] | None = None,
        max_draws: int | None = 3000,
        random_seed: int = 1729,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Plot a corner view selected through component/parameter keys."""
        from noobfriend.inference.spectrum.visualization import plot_mcmc_corner

        return plot_mcmc_corner(
            self,
            variables=variables,
            max_draws=max_draws,
            random_seed=random_seed,
            figsize=figsize,
        )

    def plot_pareto_k(
        self,
        *,
        size: int = 900,
        title: str | None = None,
    ) -> Figure:
        """Plot pointwise PSIS-LOO Pareto-k values against wavelength."""
        from noobfriend.inference.spectrum.visualization import plot_mcmc_pareto_k

        return plot_mcmc_pareto_k(self, size=size, title=title)

    def __repr__(self) -> str:
        """Return a compact representation without expanding posterior arrays."""
        diagnostics = self.sampling.diagnostics
        return (
            f"MCMCFitResult(components={self.posterior.components!r}, "
            f"chains={self.inputs.options.chains}, draws={self.inputs.options.draws}, "
            f"divergences={diagnostics.divergences})"
        )

    def _repr_html_(self) -> str:
        """Return a compact notebook overview with valid component keys."""
        diagnostics = self.sampling.diagnostics
        return f"""<section class="noob-mcmc-result">
  <h2>MCMC fit result</h2>
  <p><strong>components:</strong> {escape(", ".join(self.posterior.components))}</p>
  <p>
    <strong>chains:</strong> {self.inputs.options.chains};
    <strong>draws:</strong> {self.inputs.options.draws};
    <strong>target accept:</strong> {self.inputs.options.target_accept:.6g};
    <strong>elapsed:</strong> {self.sampling.elapsed_seconds:.3f} s
  </p>
  <p>
    <strong>divergences:</strong> {diagnostics.divergences};
    <strong>max R-hat:</strong> {diagnostics.max_rhat:.6g};
    <strong>PSIS-LOO:</strong> {self.criteria.psis_loo.elpd:.6g};
    <strong>WAIC:</strong> {self.criteria.waic.elpd:.6g}
  </p>
</section>"""


def build_mcmc_result(
    workspace: NoobFitWorkspace,
    idata: Any,
    metadata: MCMCModelMetadata,
    options: MCMCOptions,
    *,
    elapsed_seconds: float,
) -> MCMCFitResult:
    """Assemble the stable top-level result from specialized builders."""
    return MCMCFitResult(
        posterior=build_mcmc_posterior(workspace, idata, metadata),
        inputs=MCMCInputs(
            data=workspace.spectrum,
            workspace=workspace,
            options=options,
            priors=metadata.priors,
        ),
        sampling=MCMCSampling(
            elapsed_seconds=elapsed_seconds,
            diagnostics=build_mcmc_diagnostics(idata, metadata.diagnostic_variables),
        ),
        criteria=build_mcmc_criteria(idata),
        idata=idata,
    )
