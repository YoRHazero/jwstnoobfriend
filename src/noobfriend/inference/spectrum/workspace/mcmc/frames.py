"""Per-frame model reconstruction and goodness-of-fit for joint MCMC results.

A joint fit shares its line parameters across frames while giving each frame its
own continuum. These tools evaluate the posterior-median model on every frame's
native (valid) wavelength grid, exposing per-frame residuals and a per-frame
chi-square. Comparing the per-frame chi-square is how inter-visit systematics
surface: a shared line model that fits one exposure but not another shows up as
an elevated chi-square on the discrepant frame.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from noobfriend.inference.spectrum.workspace.compiler import (
    contribution_sign,
    profile_template,
)

if TYPE_CHECKING:
    from noobfriend.inference.spectrum.workspace.mcmc.result import MCMCFitResult


@dataclass(frozen=True, slots=True, repr=False)
class FrameFit:
    """Posterior-median model of one frame on its native valid pixels.

    ``components`` maps each line id to that line's contribution drawn on the
    continuum (continuum plus the signed line), matching the plotting
    convention. ``chi_square`` sums the squared standardized residuals over the
    frame's valid pixels.
    """

    frame_id: str
    wavelength: np.ndarray
    data: np.ndarray
    error: np.ndarray
    continuum: np.ndarray
    components: Mapping[str, np.ndarray]
    model: np.ndarray
    chi_square: float
    n_pixels: int

    def __post_init__(self) -> None:
        """Freeze the component mapping while preserving workspace order."""
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))

    @property
    def chi_square_per_pixel(self) -> float:
        """Chi-square divided by the number of valid pixels."""
        return self.chi_square / self.n_pixels if self.n_pixels else float("nan")

    def __repr__(self) -> str:
        """Return a compact per-frame goodness-of-fit summary."""
        return (
            f"FrameFit(frame_id={self.frame_id!r}, n_pixels={self.n_pixels}, "
            f"chi_square={self.chi_square:.6g}, "
            f"chi_square_per_pixel={self.chi_square_per_pixel:.4g})"
        )


def build_frame_fits(result: MCMCFitResult) -> tuple[FrameFit, ...]:
    """Evaluate the posterior-median model per frame on native valid pixels.

    Parameters
    ----------
    result
        A sampled MCMC result, single-frame or joint.

    Returns
    -------
    tuple of FrameFit
        One entry per frame in workspace order. A single-frame fit yields one
        entry.
    """
    workspace = result.inputs.workspace
    posterior = result.posterior
    continuum_posterior = posterior["continuum"]
    parameter_names = workspace.continuum.parameter_names
    per_frame = workspace.n_frames > 1 and workspace.continuum.sharing != "shared"

    line_medians = [
        (
            handle,
            posterior[handle.id]["flux"].median,
            posterior[handle.id]["fwhm"].median,
            posterior[handle.id]["center"].median,
            tuple(
                (
                    kernel.kind,
                    posterior[handle.id][kernel.shape_name("fwhm")].median,
                    posterior[handle.id][kernel.shape_name("fraction")].median,
                )
                for kernel in handle.line.kernels
            ),
        )
        for handle in workspace.handles
    ]

    fits: list[FrameFit] = []
    for index, frame_id in enumerate(workspace.frame_ids):
        frame = workspace.spectra[index]
        wavelength = np.asarray(frame.valid_wavelength, dtype=float)
        data = np.asarray(frame.valid_flux, dtype=float)
        error = np.asarray(frame.valid_error, dtype=float)

        coefficients = np.asarray(
            [
                continuum_posterior[f"{name}[{frame_id}]" if per_frame else name].median
                for name in parameter_names
            ],
            dtype=float,
        )
        continuum = workspace.continuum.design(wavelength) @ coefficients

        total = continuum.copy()
        components: dict[str, np.ndarray] = {}
        for handle, flux, fwhm, center, kernels in line_medians:
            signed = (
                contribution_sign(handle)
                * flux
                * profile_template(
                    wavelength,
                    handle=handle,
                    center=center,
                    fwhm_kms=fwhm,
                    resolving_power=frame.resolving_power,
                    kernels=kernels,
                )
            )
            total = total + signed
            components[handle.id] = continuum + signed

        standardized = (data - total) / error
        fits.append(
            FrameFit(
                frame_id=frame_id,
                wavelength=_readonly(wavelength),
                data=_readonly(data),
                error=_readonly(error),
                continuum=_readonly(continuum),
                components=components,
                model=_readonly(total),
                chi_square=float(np.sum(standardized**2)),
                n_pixels=int(wavelength.size),
            )
        )
    return tuple(fits)


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.array(value, dtype=float, copy=True)
    output.setflags(write=False)
    return output
