"""NoobSpectrum data preparation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from noobfriend.inference.spectrum.data._arrays import (
    _normalize_arrays,
    _normalize_redshift,
    _normalize_resolving_power,
    _normalize_unit,
)
from noobfriend.inference.spectrum.data._axis import _normalize_axis
from noobfriend.inference.spectrum.data._collapse import _collapse_to_1d
from noobfriend.inference.spectrum.data.source2d import Spectrum2DSource
from noobfriend.inference.spectrum.data._types import DispersionAxis, NoiseSource
from noobfriend.inference.spectrum._types import WaveUnit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from noobfriend.inference.spectrum.line import NoobLine, Prior
    from noobfriend.inference.spectrum.workspace import NoobFitWorkspace


@dataclass(frozen=True, slots=True, eq=False, init=False)
class NoobSpectrum:
    """Prepared 1-D spectrum data for line fitting.

    The constructor accepts any array-like input; the attributes below are the
    normalized form it stores. ``obs``, ``flux``, and ``error`` are immutable
    1-D float arrays of equal length, and ``obs`` is strictly increasing.
    ``mask_excluded`` uses ``True`` for pixels excluded from fitting. Invalid
    numeric pixels are not rewritten; they are exposed through ``valid_mask``
    so later fitting code can consistently drop them.

    Parameters
    ----------
    flux, error
        Flux density and its 1-sigma uncertainty, one value per pixel.
    obs
        Observed-frame wavelength in ``unit``. Optional only when ``rest`` and
        ``z`` are given, in which case it is derived from them.
    rest
        Rest-frame wavelength in ``unit``.
    z
        Source redshift, strictly positive when provided.
    unit
        Wavelength unit shared by ``obs`` and ``rest``.
    resolving_power
        Spectral resolving power, positive when provided.
    mask_excluded
        Boolean mask, ``True`` for pixels excluded from fitting.
    source_2d
        Canonical 2-D source retained by :meth:`from_2d`; its wavelength
        extent must match the prepared 1-D spectrum.

    Attributes
    ----------
    flux, error, obs : numpy.ndarray
        Read-only normalized arrays.
    rest, mask_excluded : numpy.ndarray or None
        Read-only normalized arrays, ``None`` when not supplied.

    Raises
    ------
    ValueError
        If the arrays disagree in shape, ``obs`` is not strictly increasing,
        no valid unmasked pixel remains, or the wavelength inputs are mutually
        inconsistent.
    """

    flux: np.ndarray
    error: np.ndarray
    obs: np.ndarray
    rest: np.ndarray | None
    z: float | None
    unit: WaveUnit
    resolving_power: float | None
    mask_excluded: np.ndarray | None
    source_2d: Spectrum2DSource | None

    def __init__(
        self,
        flux: ArrayLike,
        error: ArrayLike,
        *,
        obs: ArrayLike | None = None,
        rest: ArrayLike | None = None,
        z: float | None = None,
        unit: WaveUnit = "angstrom",
        resolving_power: float | None = None,
        mask_excluded: ArrayLike | None = None,
        source_2d: Spectrum2DSource | None = None,
    ) -> None:
        """Validate and normalize spectrum arrays and metadata."""
        parsed_unit = _normalize_unit(unit)
        parsed_z = _normalize_redshift(z)
        parsed_resolving_power = _normalize_resolving_power(resolving_power)
        axis = _normalize_axis(obs=obs, rest=rest, z=parsed_z)
        arrays = _normalize_arrays(axis.obs, flux, error, mask_excluded)
        if source_2d is not None and source_2d.flux.shape[1] != arrays.wavelength.size:
            raise ValueError(
                "source_2d wavelength extent must match the prepared 1-D spectrum."
            )

        object.__setattr__(self, "flux", arrays.flux)
        object.__setattr__(self, "error", arrays.error)
        object.__setattr__(self, "obs", arrays.wavelength)
        object.__setattr__(self, "rest", axis.rest)
        object.__setattr__(self, "z", parsed_z)
        object.__setattr__(self, "unit", parsed_unit)
        object.__setattr__(self, "resolving_power", parsed_resolving_power)
        object.__setattr__(self, "mask_excluded", arrays.mask_excluded)
        object.__setattr__(self, "source_2d", source_2d)

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
        return self.obs

    @property
    def valid_mask(self) -> np.ndarray:
        """Pixels usable by fitting after finite/error/mask filtering."""
        valid = (
            np.isfinite(self.obs)
            & np.isfinite(self.flux)
            & np.isfinite(self.error)
            & (self.error > 0)
        )
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
            raise ValueError(
                f"mask_excluded shape {new_mask.shape} must match obs {self.obs.shape}."
            )
        if self.mask_excluded is not None:
            new_mask = self.mask_excluded | new_mask
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
        continuum_priors: Mapping[str, Prior] | None = None,
    ) -> NoobFitWorkspace:
        """Resolve a line combination against this spectrum.

        Parameters
        ----------
        lines
            Line declarations, as a sequence or a mapping of id to line.
        continuum_order
            Polynomial order of the continuum.
        continuum_lambda_0
            Continuum anchor wavelength; defaults to the first line's center.
        continuum_priors
            Priors for individual continuum coefficients, keyed by name
            (``"c"``, ``"k1"``, ...) and given in the spectrum's own flux
            units. Coefficients left out keep the sampling backend's
            data-scaled default.
        """
        from noobfriend.inference.spectrum.workspace import NoobFitWorkspace

        return NoobFitWorkspace.build(
            self,
            lines,
            continuum_order=continuum_order,
            continuum_lambda_0=continuum_lambda_0,
            continuum_priors=continuum_priors,
        )

    def __repr__(self) -> str:
        """Concise spectrum summary."""
        n = self.obs.size
        lo = float(self.obs[0]) if n else float("nan")
        hi = float(self.obs[-1]) if n else float("nan")
        z = "None" if self.z is None else f"{self.z:g}"
        resolving_power = (
            "None" if self.resolving_power is None else f"{self.resolving_power:g}"
        )
        return (
            f"NoobSpectrum(n={n}, unit={self.unit!r}, z={z}, "
            f"range=[{lo:g}, {hi:g}], resolving_power={resolving_power})"
        )
