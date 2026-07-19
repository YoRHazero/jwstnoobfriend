"""Model-level contract built from a prepared spectrum workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.data import NoobSpectrum
    from noobfriend.inference.spectrum.line import (
        ComponentName,
        ContributionName,
        ProfileName,
    )
    from noobfriend.inference.spectrum.workspace import (
        ContinuumSpec,
        LineHandle,
        NoobFitWorkspace,
    )
    from noobfriend.inference.spectrum.workspace.mcmc import MCMCFitResult
    from noobfriend.inference.spectrum.workspace.mcmc.model import MCMCModelMetadata
    from noobfriend.inference.spectrum.workspace.mcmc.priors import MCMCPriors


@dataclass(frozen=True, slots=True)
class NoobSpectrumModel:
    """A built PyMC spectrum model that is ready for sampling."""

    workspace: NoobFitWorkspace
    pymc_model: Any
    _metadata: MCMCModelMetadata

    @classmethod
    def from_workspace(cls, workspace: NoobFitWorkspace) -> NoobSpectrumModel:
        """Compile a prepared workspace into one persistent PyMC model."""
        from noobfriend.inference.spectrum.workspace.mcmc.model import (
            build_workspace_model,
        )

        built = build_workspace_model(workspace)
        return cls(
            workspace=workspace,
            pymc_model=built.model,
            _metadata=built.metadata,
        )

    @property
    def spectrum(self) -> NoobSpectrum:
        """Prepared spectrum data of a single-frame model.

        Single-frame only: a model built from a multi-frame workspace raises
        ``RuntimeError``; use ``workspace.spectra`` for its frames.
        """
        return self.workspace.spectrum

    @property
    def handles(self) -> tuple[LineHandle, ...]:
        """Line handles in model order."""
        return self.workspace.handles

    @property
    def continuum(self) -> ContinuumSpec:
        """Continuum specification attached during prepare."""
        return self.workspace.continuum

    @property
    def continuum_parameter_names(self) -> tuple[str, ...]:
        """Continuum parameter names in design-matrix column order."""
        return self.continuum.parameter_names

    @property
    def component_names(self) -> tuple[ComponentName, ...]:
        """Component names used by this model, preserving first-seen order."""
        return _unique_ordered(tuple(handle.component for handle in self.handles))

    @property
    def contribution_names(self) -> tuple[ContributionName, ...]:
        """Contribution directions used by this model, preserving first-seen order."""
        return _unique_ordered(tuple(handle.contribution for handle in self.handles))

    @property
    def profile_names(self) -> tuple[ProfileName, ...]:
        """Profile families used by this model, preserving first-seen order."""
        return _unique_ordered(tuple(handle.profile for handle in self.handles))

    @property
    def priors(self) -> MCMCPriors:
        """Translated physical priors using the result component interface."""
        return self._metadata.priors

    @property
    def reparameterized_flux_pairs(self) -> tuple[tuple[str, str], ...]:
        """Emission/absorption source pairs using net/cancellation coordinates."""
        return self._metadata.reparameterized_flux_pairs

    @property
    def effective_fwhm_sources(self) -> tuple[str, ...]:
        """FWHM sources sampled internally in effective-width coordinates."""
        return self._metadata.effective_fwhm_sources

    def continuum_design(self, wavelength: ArrayLike | None = None) -> np.ndarray:
        """Return the continuum design matrix.

        When ``wavelength`` is omitted, only valid fitting pixels are used —
        single-frame only, since a joint model has one wavelength grid per
        frame (raises ``RuntimeError``). Passing ``wavelength`` explicitly
        works for any model.
        """
        wl = self.spectrum.valid_wavelength if wavelength is None else wavelength
        return self.continuum.design(wl)

    def sample(
        self,
        *,
        draws: int = 1000,
        tune: int = 2000,
        chains: int = 16,
        cores: int | None = None,
        target_accept: float = 0.95,
        random_seed: int = 1729,
        progressbar: bool = True,
    ) -> MCMCFitResult:
        """Sample this exact PyMC model and return physical posterior results."""
        from noobfriend.inference.spectrum.workspace.mcmc.fit import (
            sample_spectrum_model,
        )
        from noobfriend.inference.spectrum.workspace.mcmc.options import (
            build_mcmc_options,
        )

        options = build_mcmc_options(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=progressbar,
        )
        return sample_spectrum_model(self, options=options)


def _unique_ordered[T](values: tuple[T, ...]) -> tuple[T, ...]:
    seen: set[T] = set()
    output: list[T] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)
