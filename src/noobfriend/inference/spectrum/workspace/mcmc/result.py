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
    from pathlib import Path

    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

    from noobfriend.inference.spectrum.data import NoobSpectrum, NoobSpectrumSet
    from noobfriend.inference.spectrum.data.types import DispersionAxis
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace
    from noobfriend.inference.spectrum.workspace.mcmc.frames import FrameFit
    from noobfriend.inference.spectrum.workspace.mcmc.lofo import FrameLOOResult
    from noobfriend.inference.spectrum.workspace.mcmc.model import MCMCModelMetadata
    from noobfriend.inference.spectrum.workspace.mcmc.options import MCMCOptions


@dataclass(frozen=True, slots=True)
class MCMCInputs:
    """Input data, declarations, sampling options, and translated priors.

    ``data`` is the single :class:`NoobSpectrum` for a single-frame fit or the
    :class:`NoobSpectrumSet` of frames for a joint fit.
    """

    data: NoobSpectrum | NoobSpectrumSet
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

        A joint multi-frame fit renders per-frame panels through
        :meth:`plot_frames`, forwarding the shared arguments (including
        ``show_residuals``, ``model_oversample``, ``view``,
        ``cmap``/``pmin``/``pmax``, and ``legend_location``); frames created
        with ``NoobSpectrum.from_2d`` get per-frame 2-D strips. The explicit
        2-D overlay arguments (``flux_2d``/``spatial``/``dispersion``/
        ``spatial_window``) are single-frame only and raise if supplied for a
        joint fit.
        """
        if self.inputs.workspace.n_frames > 1:
            if any(
                argument is not None
                for argument in (flux_2d, spatial, dispersion, spatial_window)
            ):
                raise ValueError(
                    "2-D overlays are single-frame only; a joint fit renders "
                    "per-frame panels. Call .plot_frames() or drop flux_2d, "
                    "spatial, dispersion, and spatial_window."
                )
            return self.plot_frames(
                hdi_probability=hdi_probability,
                posterior_draws=posterior_draws,
                random_seed=random_seed,
                show_residuals=show_residuals,
                model_oversample=model_oversample,
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

    def plot_frames(
        self,
        *,
        hdi_probability: float | None = 0.94,
        posterior_draws: int | None = 1000,
        random_seed: int = 1729,
        show_components: bool = True,
        show_chi_square: bool = True,
        show_residuals: bool = False,
        model_oversample: int = 8,
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
        """Plot the posterior model over each frame as stacked panels.

        Works for any fit; a single-frame fit yields one panel. Each frame's
        total model carries a pointwise ``hdi_probability`` band (its parameter
        uncertainty; pass ``None`` for median only), and each panel is annotated
        with its chi-square per pixel so inter-visit systematics are visible.
        ``show_residuals`` adds a residual panel per frame, ``view`` selects
        the convolution layers of the component curves (the total model and
        residuals stay observed), and frames created with
        ``NoobSpectrum.from_2d`` get their 2-D source strip scaled by
        ``cmap``/``pmin``/``pmax``.
        """
        from noobfriend.inference.spectrum.visualization import plot_mcmc_frames

        return plot_mcmc_frames(
            self,
            hdi_probability=hdi_probability,
            posterior_draws=posterior_draws,
            random_seed=random_seed,
            show_components=show_components,
            show_chi_square=show_chi_square,
            show_residuals=show_residuals,
            model_oversample=model_oversample,
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

    def frame_fits(
        self,
        *,
        hdi_probability: float | None = None,
        max_draws: int | None = 1000,
        random_seed: int = 1729,
    ) -> tuple[FrameFit, ...]:
        """Return the posterior model and chi-square for each frame.

        A single-frame fit yields one entry; a joint fit yields one entry per
        frame in workspace order. Comparing ``chi_square_per_pixel`` across
        frames exposes inter-visit systematics under the shared line model.
        With ``hdi_probability`` set, each frame also carries the total model's
        pointwise highest-density band (``model_lower``/``model_upper``).
        """
        from noobfriend.inference.spectrum.workspace.mcmc.frames import (
            build_frame_fits,
        )

        return build_frame_fits(
            self,
            hdi_probability=hdi_probability,
            max_draws=max_draws,
            random_seed=random_seed,
        )

    def frame_loo(
        self,
        *,
        max_draws: int | None = 1000,
        random_seed: int = 1729,
    ) -> FrameLOOResult:
        """Leave-one-frame-out PSIS cross-validation for a joint fit.

        Drops a whole exposure (not a pixel). A pooled continuum is marginalized
        analytically per frame so the estimate has no per-frame leak; a shared
        continuum has no per-frame nuisance and is evaluated directly. Lead with
        ``pareto_k`` and the ``elpd_i <= lpd_i`` self-check — an influential or
        inconsistent frame shows a high Pareto-k, while raw ``elpd_i`` merely
        scales with a frame's pixel count. Requires two or more frames and a
        pooled or shared continuum (``"independent"`` is not supported).
        """
        from noobfriend.inference.spectrum.workspace.mcmc.lofo import build_frame_loo

        return build_frame_loo(self, max_draws=max_draws, random_seed=random_seed)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save this result to a directory for later :meth:`load`.

        Writes ``idata.nc`` (the raw chains as arviz-compatible NetCDF),
        ``inputs.pkl`` (the pickled workspace and sampling options), and
        ``manifest.json`` (human-readable provenance plus the sampling
        diagnostics, so a run's validity can be checked without loading
        Python objects). The directory is created if needed; existing
        persistence files are refused unless ``overwrite`` is true.
        """
        from noobfriend.inference.spectrum.workspace.mcmc.persistence import (
            save_mcmc_result,
        )

        return save_mcmc_result(self, path, overwrite=overwrite)

    @classmethod
    def load(cls, path: str | Path) -> MCMCFitResult:
        """Rebuild a result saved by :meth:`save`.

        The posterior, diagnostics, and criteria are re-derived from the
        saved chains through the same builders that run after sampling, so
        the loaded result is always consistent with the chains on disk.
        Rebuilding recompiles the PyMC model to recover variable naming and
        therefore requires the ``mcmc`` optional dependency.
        """
        from noobfriend.inference.spectrum.workspace.mcmc.persistence import (
            load_mcmc_result,
        )

        return load_mcmc_result(path)

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
            data=_provenance_data(workspace),
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


def _provenance_data(
    workspace: NoobFitWorkspace,
) -> NoobSpectrum | NoobSpectrumSet:
    """Return the single spectrum or the joined frame set used for the fit."""
    if workspace.n_frames == 1:
        return workspace.spectrum
    from noobfriend.inference.spectrum.data import NoobSpectrumSet

    return NoobSpectrumSet(
        dict(zip(workspace.frame_ids, workspace.spectra, strict=True))
    )
