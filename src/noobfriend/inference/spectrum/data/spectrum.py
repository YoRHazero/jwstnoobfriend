"""NoobSpectrum data preparation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from noobfriend.inference.spectrum.data.arrays import (
    _normalize_arrays,
    _normalize_redshift,
    _normalize_resolving_power,
    _normalize_unit,
)
from noobfriend.inference.spectrum.data.axis import _normalize_axis
from noobfriend.inference.spectrum.data.collapse import _collapse_to_1d
from noobfriend.inference.spectrum.data.source2d import Spectrum2DSource
from noobfriend.inference.spectrum.data.types import DispersionAxis, NoiseSource
from noobfriend.inference.spectrum.types import WaveUnit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from noobfriend.inference.spectrum.line import NoobLine
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace


@dataclass(frozen=True, slots=True, eq=False)
class NoobSpectrum:
    """Prepared 1-D spectrum data for line fitting.

    ``obs``, ``flux``, and ``error`` are normalized to immutable 1-D arrays.
    ``mask_excluded`` uses ``True`` for pixels excluded from fitting. Invalid
    numeric pixels are not rewritten; they are exposed through ``valid_mask`` so
    later fitting code can consistently drop them.
    """

    flux: ArrayLike
    error: ArrayLike
    obs: ArrayLike | None = field(default=None, kw_only=True)
    rest: ArrayLike | None = field(default=None, kw_only=True)
    z: float | None = field(default=None, kw_only=True)
    unit: WaveUnit = field(default="angstrom", kw_only=True)
    resolving_power: float | None = field(default=None, kw_only=True)
    mask_excluded: ArrayLike | None = field(default=None, kw_only=True)
    source_2d: Spectrum2DSource | None = field(default=None, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize spectrum arrays and metadata."""
        unit = _normalize_unit(self.unit)
        z = _normalize_redshift(self.z)
        resolving_power = _normalize_resolving_power(self.resolving_power)
        axis = _normalize_axis(obs=self.obs, rest=self.rest, z=z)
        arrays = _normalize_arrays(axis.obs, self.flux, self.error, self.mask_excluded)
        if self.source_2d is not None and self.source_2d.flux.shape[1] != arrays.wavelength.size:
            raise ValueError(
                "source_2d wavelength extent must match the prepared 1-D spectrum."
            )

        object.__setattr__(self, "flux", arrays.flux)
        object.__setattr__(self, "error", arrays.error)
        object.__setattr__(self, "obs", arrays.wavelength)
        object.__setattr__(self, "rest", axis.rest)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "resolving_power", resolving_power)
        object.__setattr__(self, "mask_excluded", arrays.mask_excluded)

    @classmethod
    def from_2d(
        cls,
        flux: ArrayLike,
        error: ArrayLike,
        *,
        obs: ArrayLike | None = None,
        rest: ArrayLike | None = None,
        z: float | None = None,
        collapse_window: tuple[int, int],
        dispersion: DispersionAxis = "row",
        unit: WaveUnit = "angstrom",
        resolving_power: float | None = None,
        mask_excluded: ArrayLike | None = None,
        noise: NoiseSource = "manual",
        error_boost: float = 1.0,
        source_mask: ArrayLike | None = None,
        max_lag: int = 8,
    ) -> NoobSpectrum:
        """Collapse 2-D data and retain its canonical source for visualization."""
        parsed_z = _normalize_redshift(z)
        axis = _normalize_axis(obs=obs, rest=rest, z=parsed_z)
        wl, fl, er = _collapse_to_1d(
            axis.obs,
            flux,
            error,
            collapse_window=collapse_window,
            dispersion=dispersion,
            noise=noise,
            error_boost=error_boost,
            source_mask=source_mask,
            max_lag=max_lag,
        )
        from noobfriend.inference.spectrum.data.source2d import (
            build_spectrum_2d_source,
        )

        source_2d = build_spectrum_2d_source(
            np.asarray(flux, dtype=float),
            np.asarray(error, dtype=float),
            collapse_window=collapse_window,
            dispersion=dispersion,
        )
        return cls(
            fl,
            er,
            obs=wl,
            rest=axis.rest,
            z=parsed_z,
            unit=unit,
            resolving_power=resolving_power,
            mask_excluded=mask_excluded,
            source_2d=source_2d,
        )

    @property
    def wavelength(self) -> np.ndarray:
        """Observed-frame wavelength, kept as a data-view alias."""
        if self.obs is None:
            raise RuntimeError("NoobSpectrum invariant violated: missing observed wavelength.")
        return self.obs

    @property
    def valid_mask(self) -> np.ndarray:
        """Pixels usable by fitting after finite/error/mask filtering."""
        valid = np.isfinite(self.obs) & np.isfinite(self.flux) & np.isfinite(self.error) & (self.error > 0)
        if self.mask_excluded is not None:
            valid &= ~self.mask_excluded
        valid.setflags(write=False)
        return valid

    @property
    def valid_wavelength(self) -> np.ndarray:
        """Wavelength values on valid pixels."""
        return self.obs[self.valid_mask]

    @property
    def valid_flux(self) -> np.ndarray:
        """Flux values on valid pixels."""
        return self.flux[self.valid_mask]

    @property
    def valid_error(self) -> np.ndarray:
        """1-sigma errors on valid pixels."""
        return self.error[self.valid_mask]

    def replace_mask(self, mask_excluded: ArrayLike | None) -> NoobSpectrum:
        """Return a copy with the data-quality mask replaced."""
        return replace(self, mask_excluded=mask_excluded)

    def exclude(self, mask_excluded: ArrayLike) -> NoobSpectrum:
        """Return a copy with additional excluded pixels OR-ed into the mask."""
        new_mask = np.asarray(mask_excluded, dtype=bool)
        if new_mask.shape != self.obs.shape:
            raise ValueError(f"mask_excluded shape {new_mask.shape} must match obs {self.obs.shape}.")
        if self.mask_excluded is not None:
            new_mask = np.asarray(self.mask_excluded, dtype=bool) | new_mask
        return replace(self, mask_excluded=new_mask)

    def clear_mask(self) -> NoobSpectrum:
        """Return a copy without a data-quality mask."""
        return replace(self, mask_excluded=None)

    def calibrate_error(
        self,
        *,
        mask: ArrayLike | None = None,
        degree: int = 3,
        nsigma: float = 3.0,
        niter: int = 5,
        min_pixels: int = 10,
    ) -> NoobSpectrum:
        """Return a copy whose errors match the continuum scatter."""
        from noobfriend.core.specutils import continuum_boost

        boost = continuum_boost(
            self.obs,
            self.flux,
            self.error,
            mask=mask,
            degree=degree,
            nsigma=nsigma,
            niter=niter,
            min_pixels=min_pixels,
        )
        return replace(self, error=self.error * boost)

    def prepare(
        self,
        lines: Sequence[NoobLine] | Mapping[str, NoobLine],
        *,
        continuum_order: int = 1,
        continuum_lambda_0: float | None = None,
    ) -> NoobFitWorkspace:
        """Resolve a line combination against this spectrum."""
        from noobfriend.inference.spectrum.workspace import NoobFitWorkspace

        return NoobFitWorkspace.build(
            self,
            lines,
            continuum_order=continuum_order,
            continuum_lambda_0=continuum_lambda_0,
        )

    def __repr__(self) -> str:
        """Concise spectrum summary."""
        n = self.obs.size
        lo = float(self.obs[0]) if n else float("nan")
        hi = float(self.obs[-1]) if n else float("nan")
        z = "None" if self.z is None else f"{self.z:g}"
        resolving_power = "None" if self.resolving_power is None else f"{self.resolving_power:g}"
        return (
            f"NoobSpectrum(n={n}, unit={self.unit!r}, z={z}, "
            f"range=[{lo:g}, {hi:g}], resolving_power={resolving_power})"
        )
