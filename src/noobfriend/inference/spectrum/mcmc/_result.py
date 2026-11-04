"""NoobMCMCResult: posterior access, checks, figures, persistence, R1 priors."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from noobfriend.inference.spectrum._xp import numpy_namespace
from noobfriend.inference.spectrum.mcmc._decompose import (
    NoobSpectrumDecomposition,
    decompose,
    median_noise,
    posterior_values,
)
from noobfriend.inference.spectrum.mcmc._posterior import (
    ComponentPosterior,
    ContinuumPosterior,
    FramePosterior,
    FramePosteriors,
    ParameterPosterior,
    PosteriorSource,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from noobfriend.inference.spectrum.mcmc._diagnostics import SamplingReport
    from noobfriend.inference.spectrum.workspace import ModelView, NoobFitWorkspace


class NoobMCMCResult:
    """Posterior of a spectrum fit.

    Read parameters like the declarations: ``r["Ha"].fwhm``,
    ``r["Ha"].laplace.fwhm``, ``r["continuum"].coefficient(1)``,
    ``r.frames["v1"].noise.rho``; a flat name such as ``r["Ha.fwhm"]`` works
    too. Every value is a :class:`ParameterPosterior`.

    Attributes
    ----------
    workspace : NoobFitWorkspace
        The fit plan.
    idata : xarray.DataTree
        ArviZ InferenceData. ``posterior`` holds only physical parameters
        under their flat names; internal sampling coordinates are in
        ``internal``; ``log_likelihood`` holds ``pixels`` (pointwise, exact
        conditionals for correlated noise). Use it directly with ArviZ (trace,
        rank, Pareto-k plots).
    sampling : SamplingReport
        Options, wall time, diagnostics, convergence verdict and
        reparameterizations.

    Examples
    --------
    Plots beyond ``decompose().plot()`` and ``plot_corner()`` come straight
    from ArviZ, since ``idata`` and ``loo`` are its native objects::

        import arviz as az

        az.plot_trace_dist(result.idata, var_names=["Ha.fwhm", "Ha.flux"])
        az.plot_rank(result.idata, var_names=["Ha.fwhm"])
        az.plot_khat(result.loo)              # pixel Pareto-k
        az.summary(result.idata)              # every physical parameter
    """

    def __init__(
        self, workspace: NoobFitWorkspace, idata: Any, sampling: SamplingReport
    ) -> None:
        self.workspace = workspace
        self.idata = idata
        self.sampling = sampling
        self._source = PosteriorSource(idata["posterior"].dataset)

    def __getitem__(
        self, name: str
    ) -> ComponentPosterior | ContinuumPosterior | ParameterPosterior:
        """Return a component by name, or one parameter by flat dotted name."""
        if "." in name:
            return self._source.get(name)
        if name == "continuum":
            names = tuple(
                f"continuum.{parameter}"
                for parameter in self.workspace.continuum.parameter_names
                if f"continuum.{parameter}" in self.workspace.parameters
            )
            return ContinuumPosterior(self._source, names, "as one shared value")
        handle = self.workspace[name]
        return ComponentPosterior(self._source, handle)  # type: ignore[arg-type]

    @cached_property
    def loo(self) -> Any:
        """PSIS-LOO over pixels (ArviZ ``ELPDData`` with pointwise Pareto-k)."""
        import arviz as az

        return az.loo(self.idata, pointwise=True, var_name="pixels")

    @cached_property
    def frames(self) -> FramePosteriors:
        """Per-frame parameters and whitened chi-square.

        There is no leave-one-frame-out LOO: PSIS on a frame's joint log
        likelihood cannot stand in for refitting without the frame, whose own
        continuum and noise parameters only that frame constrains. Compare
        models with the pixel-level :attr:`loo`; judge frame consistency with
        each frame's ``chi2``.
        """
        values = posterior_values(self, 1000)
        xp = numpy_namespace()
        frames = {}
        for frame_id in self.workspace.frame_ids:
            frame = self.workspace.spectrum.frames[frame_id]
            mask = self.workspace.fitted[frame_id]
            model = np.median(
                self.workspace.frame_model(xp, values, frame_id).total, axis=0
            )
            scale, rho = median_noise(values, frame_id)
            whitened = self.workspace.frame_noise[frame_id].whiten(
                xp, np.asarray(frame.flux)[mask] - model, scale, rho
            )
            frames[frame_id] = FramePosterior(
                self._source,
                self.workspace,
                frame_id,
                chi2=float(np.sum(np.asarray(whitened) ** 2)),
            )
        return FramePosteriors(frames)

    def summary(self) -> str:
        """Return plan, sampling verdict, posterior table, pixel LOO and frames as text."""
        from noobfriend.inference.spectrum.mcmc._summary import result_summary

        return result_summary(self)

    def decompose(
        self,
        view: ModelView = "observed",
        frames: Sequence[str] | None = None,
        hdi_prob: float = 0.94,
        draws: int = 1000,
        oversample: int = 8,
    ) -> NoobSpectrumDecomposition:
        """Posterior bands of the model curve and its components, per frame.

        This is the posterior of the library's mean model, without
        observation noise (not a posterior predictive). If the NumPyro model
        was rewritten to change the mean model, the change is not reflected;
        changes to priors or likelihood do not matter here.

        Parameters
        ----------
        view
            Components as ``"observed"`` (kernels and LSF),
            ``"emergent"`` (kernels only) or ``"intrinsic"`` (bare profile).
            Total, model, residual and chi-square are always observed, so
            emergent or intrinsic components do not add up to the total.
        frames
            Frame ids; every frame by default.
        hdi_prob
            Credible mass of the pointwise HDI bands.
        draws
            Number of draws, taken at evenly spaced positions of every chain
            (deterministic).
        oversample
            Grid points per native pixel for the smooth curves.

        Returns
        -------
        NoobSpectrumDecomposition
            Frame id to bands, residuals and chi-square; ``.plot()`` draws it.
        """
        return decompose(
            self,
            view=view,
            frames=frames,
            hdi_prob=hdi_prob,
            draws=draws,
            oversample=oversample,
        )

    def plot_corner(self, params: Sequence[str] | None = None) -> Figure:
        """Corner plot of posterior parameters.

        Parameters
        ----------
        params
            Flat names; every free line parameter by default.

        Returns
        -------
        matplotlib.figure.Figure
            The figure from ArviZ ``plot_pair``.
        """
        from noobfriend.inference.spectrum.mcmc._plot import plot_corner

        return plot_corner(self, params)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write the InferenceData to one netCDF file.

        Reload it with ``workspace.load(path)`` on the workspace rebuilt by
        the same script; ArviZ can also read the file on its own.

        Parameters
        ----------
        path
            Target file.
        overwrite
            Replace an existing file.

        Returns
        -------
        pathlib.Path
            The written file.
        """
        from noobfriend.inference.spectrum.mcmc._persist import save

        return save(self, path, overwrite=overwrite)

    def as_prior(self, names: Sequence[str]) -> dict[str, Any]:
        """Turn components into declarations sharing one joint prior.

        Every sampled parameter of the selected components (pooled means for
        a multi-frame continuum) gets a share of one multivariate normal,
        moment-matched to the posterior in unconstrained coordinates, so
        their correlations carry over. Pass the returned declarations to
        another ``prepare``: components left out are marginalized exactly,
        and setting a parameter again with its own method takes it out of
        the joint prior. Fitting the same pixels again is refused.

        Parameters
        ----------
        names
            Component names (lines, ``"continuum"``); every derived line's
            base must be included.

        Returns
        -------
        dict
            Name to the new ``NoobLine`` / ``NoobContinuum``.

        Warns
        -----
        NonNormalPriorWarning
            When a converted parameter is clearly non-normal in
            unconstrained coordinates.
        """
        from noobfriend.inference.spectrum.mcmc._as_prior import as_prior

        return as_prior(self, names)

    def __repr__(self) -> str:
        """Summarize the verdict."""
        verdict = "converged" if self.sampling.verdict.converged else "NOT converged"
        return f"NoobMCMCResult({self.workspace!r}, {verdict})"
