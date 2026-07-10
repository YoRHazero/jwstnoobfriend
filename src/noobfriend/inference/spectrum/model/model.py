"""Model-level contract built from a prepared spectrum workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NoReturn

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.data import NoobSpectrum
    from noobfriend.inference.spectrum.line import ComponentName, ContributionName, ProfileName
    from noobfriend.inference.spectrum.workspace import ContinuumSpec, LineHandle, NoobFitWorkspace


@dataclass(frozen=True, slots=True)
class NoobSpectrumModel:
    """A fit-ready model specification resolved from one workspace."""

    workspace: NoobFitWorkspace

    @classmethod
    def from_workspace(cls, workspace: NoobFitWorkspace) -> NoobSpectrumModel:
        """Create a model contract from an already prepared workspace."""
        return cls(workspace=workspace)

    @property
    def spectrum(self) -> NoobSpectrum:
        """Prepared spectrum data used by this model."""
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

    def continuum_design(self, wavelength: ArrayLike | None = None) -> np.ndarray:
        """Return the continuum design matrix.

        When ``wavelength`` is omitted, only valid fitting pixels are used.
        """
        wl = self.spectrum.valid_wavelength if wavelength is None else wavelength
        return self.continuum.design(wl)

    def fit(self, *_args: object, **_kwargs: object) -> NoReturn:
        """Fit the model once a sampler backend is attached."""
        raise NotImplementedError("NoobSpectrumModel.fit is not implemented yet.")


def _unique_ordered[T](values: tuple[T, ...]) -> tuple[T, ...]:
    seen: set[T] = set()
    output: list[T] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return tuple(output)
