"""``NoobSpectrum``: the 1-D spectrum plus the physics needed to fit it.

A :class:`NoobSpectrum` carries the data (wavelength / flux / error) together
with the observation facts a line fit needs but a bare array cannot supply: the
systemic redshift ``z``, the wavelength ``wave_unit`` (mandatory here -- it is
consumed by equivalent widths and filter-curve convolution, unlike the
deliberately unit-agnostic plotters), and an optional spectral resolving power
``R`` that floors line widths. Build it from an already-extracted 1-D spectrum
with :meth:`from_1d`, or collapse a 2-D spectrum over a cross-dispersion window
with :meth:`from_2d`.

:meth:`setup` takes a list of :class:`~noobfriend.inference.spectrum._line.NoobLine`
declarations and compiles them against this spectrum into a
:class:`~noobfriend.inference.spectrum._setup.LineFitSetup`.

This module is internal; :class:`NoobSpectrum` is re-exported from
:mod:`noobfriend.inference.spectrum`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike

from noobfriend.inference.spectrum._units import WaveUnit, check_unit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from noobfriend.inference.spectrum._line import NoobLine
    from noobfriend.inference.spectrum._setup import LineFitSetup


@dataclass(frozen=True, eq=False, repr=False)
class NoobSpectrum:
    """A 1-D spectrum with the redshift, unit, and resolution to fit it.

    Built via :meth:`from_1d` / :meth:`from_2d`; not constructed directly.

    Attributes
    ----------
    wavelength, flux, error : numpy.ndarray
        Equal-length 1-D arrays. ``wavelength`` is in ``wave_unit``; ``error`` is
        the 1-sigma uncertainty on ``flux``.
    z : float
        Systemic redshift; line centres default to ``(1 + z) * rest_wavelength``.
    wave_unit : WaveUnit
        Wavelength unit (``"A"``, ``"nm"``, or ``"um"``); a line's rest wavelength
        is converted into this frame at setup, and all fitted quantities come out
        in it.
    R : float or None
        Spectral resolving power ``lambda / dlambda``, used to floor line widths.
        ``None`` leaves widths floored only by the template bounds.
    """

    wavelength: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    z: float
    wave_unit: WaveUnit
    R: float | None = None

    @classmethod
    def from_1d(
        cls,
        wavelength: ArrayLike,
        flux: ArrayLike,
        error: ArrayLike,
        *,
        z: float,
        wave_unit: WaveUnit,
        R: float | None = None,
    ) -> NoobSpectrum:
        """Build from an already-extracted 1-D spectrum.

        Parameters
        ----------
        wavelength, flux, error : array_like
            Equal-length 1-D arrays.
        z : float
            Systemic redshift (``> -1``).
        wave_unit : WaveUnit
            Wavelength unit of ``wavelength`` (``"A"``, ``"nm"``, or ``"um"``).
        R : float or None, optional
            Spectral resolving power.

        Returns
        -------
        NoobSpectrum

        Raises
        ------
        ValueError
            If the arrays are not 1-D and equal length, ``z <= -1``, ``wave_unit``
            is empty, or ``R`` is non-positive.
        """
        wl = np.asarray(wavelength, dtype=float)
        fl = np.asarray(flux, dtype=float)
        er = np.asarray(error, dtype=float)
        if wl.ndim != 1 or fl.ndim != 1 or er.ndim != 1:
            raise ValueError("wavelength, flux, error must each be 1-D.")
        if not (wl.shape == fl.shape == er.shape):
            raise ValueError(
                f"wavelength {wl.shape}, flux {fl.shape}, error {er.shape} "
                "must have equal length."
            )
        return cls(*_validated(wl, fl, er, z, wave_unit, R))

    @classmethod
    def from_2d(
        cls,
        wavelength: ArrayLike,
        flux: ArrayLike,
        error: ArrayLike,
        *,
        collapse_window: tuple[int, int],
        dispersion: Literal["row", "column"] = "row",
        z: float,
        wave_unit: WaveUnit,
        R: float | None = None,
        correlation_source: Literal["continuum", "background", "manual"] = "manual",
        correlation_correction: float = 1.0,
        max_lag: int = 8,
    ) -> NoobSpectrum:
        """Collapse a 2-D spectrum over a cross-dispersion window into 1-D.

        Sums ``flux`` over the cross-dispersion rows in ``collapse_window``
        (flux-conserving) and propagates ``error`` in quadrature, then inflates
        the 1-D error by a correlated-noise ``boost`` whose source is chosen by
        ``correlation_source``. The default (``"manual"`` with a factor of
        ``1.0``) leaves the naive quadrature error untouched.

        Parameters
        ----------
        wavelength : array_like
            1-D wavelength along the dispersion axis (length matching the 2-D
            dispersion extent).
        flux, error : array_like
            2-D arrays of equal shape.
        collapse_window : tuple of int
            ``(lo, hi)`` half-open index range on the cross-dispersion axis.
        dispersion : {"row", "column"}, default "row"
            Dispersion direction: ``"row"`` disperses along axis 1, ``"column"``
            along axis 0 (matching the grism convention).
        z, wave_unit, R : see :meth:`from_1d`.
        correlation_source : {"manual", "continuum", "background"}, default "manual"
            Where the error-inflation boost comes from: ``"manual"`` uses
            ``correlation_correction`` verbatim; ``"continuum"`` measures the
            line-free scatter of the collapsed 1-D spectrum; ``"background"``
            estimates the noise autocovariance from the 2-D source-free
            background (see :mod:`noobfriend.core.specutils`).
        correlation_correction : float, default 1.0
            The manual boost; used (and only used) when
            ``correlation_source="manual"``.
        max_lag : int, default 8
            Lag-window half-width for the ``"background"`` autocovariance.

        Returns
        -------
        NoobSpectrum

        Raises
        ------
        ValueError
            If ``dispersion``/``correlation_source`` is invalid, the 2-D arrays
            mismatch, ``wavelength`` does not match the dispersion extent,
            ``collapse_window`` is out of range, ``correlation_correction`` is
            non-positive, or it is set off ``"manual"``.
        """
        if dispersion not in ("row", "column"):
            raise ValueError(
                f"dispersion must be 'row' or 'column', got {dispersion!r}."
            )
        fl = np.asarray(flux, dtype=float)
        er = np.asarray(error, dtype=float)
        wl = np.asarray(wavelength, dtype=float)
        if fl.ndim != 2 or er.ndim != 2:
            raise ValueError("flux and error must be 2-D for from_2d.")
        if fl.shape != er.shape:
            raise ValueError(f"flux {fl.shape} and error {er.shape} must match.")
        dispersion_axis = 1 if dispersion == "row" else 0
        cross_axis = 1 - dispersion_axis
        if wl.ndim != 1 or wl.shape[0] != fl.shape[dispersion_axis]:
            raise ValueError(
                f"wavelength length {wl.shape} must match the dispersion extent "
                f"{fl.shape[dispersion_axis]}."
            )
        lo, hi = int(collapse_window[0]), int(collapse_window[1])
        if not 0 <= lo < hi <= fl.shape[cross_axis]:
            raise ValueError(
                f"collapse_window {collapse_window} out of range for "
                f"cross-dispersion extent {fl.shape[cross_axis]}."
            )
        if correlation_source not in ("continuum", "background", "manual"):
            raise ValueError(
                f"correlation_source must be 'continuum', 'background', or "
                f"'manual', got {correlation_source!r}."
            )
        if correlation_source != "manual" and correlation_correction != 1.0:
            raise ValueError(
                "correlation_correction is only used with "
                "correlation_source='manual'; do not set both."
            )
        if correlation_correction <= 0:
            raise ValueError(
                f"correlation_correction must be positive, got {correlation_correction}."
            )

        from noobfriend.core.specutils import collapse

        flux1d, error1d = collapse(
            fl, er, window=(lo, hi), dispersion_axis=dispersion_axis
        )
        boost = _collapse_boost(
            correlation_source,
            correlation_correction,
            fl,
            er,
            wl,
            flux1d,
            error1d,
            collapse_window=(lo, hi),
            dispersion_axis=dispersion_axis,
            max_lag=max_lag,
        )
        return cls(*_validated(wl, flux1d, error1d * boost, z, wave_unit, R))

    def calibrate_error(self, *, mask: ArrayLike | None = None) -> NoobSpectrum:
        """Rescale the error to match the line-free continuum scatter.

        Returns a copy whose ``error`` is multiplied by the
        :func:`~noobfriend.core.specutils.continuum_boost` -- the ratio
        of the spectrum's own line-free scatter to its quoted error. Useful when
        the supplied error array is mis-calibrated (e.g. correlated noise), and
        applicable to any spectrum, not just a 2-D collapse.

        Parameters
        ----------
        mask : array_like of bool, optional
            ``True`` on pixels eligible as continuum (line-free); defaults to a
            sigma-clip of all finite pixels.

        Returns
        -------
        NoobSpectrum
            A copy with the rescaled error.
        """
        from noobfriend.core.specutils import continuum_boost

        boost = continuum_boost(self.wavelength, self.flux, self.error, mask=mask)
        return replace(self, error=self.error * boost)

    def setup(self, lines: Sequence[NoobLine]) -> LineFitSetup:
        """Compile a list of line declarations against this spectrum.

        Parameters
        ----------
        lines : sequence of NoobLine
            The model: root lines and their derived companions.

        Returns
        -------
        LineFitSetup
            The compiled model -- generated ids, resolved per-axis ties, and
            validation -- ready to inspect before fitting.
        """
        from noobfriend.inference.spectrum._setup import build_setup

        return build_setup(self, lines)

    def __repr__(self) -> str:
        """Concise summary of size, redshift, unit, and wavelength span."""
        n = self.wavelength.size
        lo = float(np.nanmin(self.wavelength)) if n else float("nan")
        hi = float(np.nanmax(self.wavelength)) if n else float("nan")
        r = "None" if self.R is None else f"{self.R:g}"
        return (
            f"NoobSpectrum(n={n}, z={self.z:g}, unit={self.wave_unit!r}, "
            f"range=[{lo:g}, {hi:g}], R={r})"
        )


def _validated(
    wl: np.ndarray,
    fl: np.ndarray,
    er: np.ndarray,
    z: float,
    wave_unit: WaveUnit,
    R: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, WaveUnit, float | None]:
    """Validate the shared scalar metadata and return the constructor tuple.

    Raises
    ------
    ValueError
        If ``z <= -1``, ``wave_unit`` is not a valid unit, or ``R`` is
        non-positive.
    """
    if z <= -1:
        raise ValueError(f"z must be > -1, got {z}.")
    check_unit(wave_unit, "wave_unit")
    if R is not None and R <= 0:
        raise ValueError(f"R must be positive, got {R}.")
    return wl, fl, er, float(z), wave_unit, (None if R is None else float(R))


def _collapse_boost(
    source: str,
    manual: float,
    flux2d: np.ndarray,
    error2d: np.ndarray,
    wavelength: np.ndarray,
    flux1d: np.ndarray,
    error1d: np.ndarray,
    *,
    collapse_window: tuple[int, int],
    dispersion_axis: int,
    max_lag: int,
) -> float:
    """Resolve the correlated-noise boost for a 2-D collapse by source."""
    if source == "manual":
        return float(manual)
    from noobfriend.core.specutils import continuum_boost, cross_dispersion_boost

    if source == "continuum":
        return continuum_boost(wavelength, flux1d, error1d)
    return cross_dispersion_boost(
        flux2d,
        error2d,
        window=collapse_window,
        dispersion_axis=dispersion_axis,
        max_lag=max_lag,
    )
