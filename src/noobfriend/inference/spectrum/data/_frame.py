"""One frame of a spectrum: native pixels, mask, resolution and noise."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from noobfriend.inference.spectrum._xp import readonly
from noobfriend.inference.spectrum.noise import Noise

#: ``(wavelength, R)`` table, linearly interpolated in the data's unit.
type ResolvingPowerTable = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True, eq=False)
class Source2D:
    """The 2-D spectrum a frame was collapsed from.

    Arrays are stored as ``(spatial, dispersion)`` whatever the input
    orientation was.

    Attributes
    ----------
    flux, error
        2-D arrays, rows along the cross-dispersion axis.
    spatial
        Cross-dispersion coordinate of each row (pixel index by default).
    collapse_window
        ``(lo, hi)`` half-open row range summed into the 1-D spectrum.
    dispersion_axis
        The dispersion axis of the arrays as originally passed (``0`` or
        ``1``); 2-D masks given later use that orientation too.
    """

    flux: np.ndarray
    error: np.ndarray
    spatial: np.ndarray
    collapse_window: tuple[int, int]
    dispersion_axis: int

    def canonical(self, array: ArrayLike) -> np.ndarray:
        """Reorient a 2-D array given in the input orientation to ``(spatial, dispersion)``."""
        values = np.asarray(array)
        return values.T if self.dispersion_axis == 0 else values


@dataclass(frozen=True, slots=True, eq=False)
class SpectrumFrame:
    """One frame of a :class:`NoobSpectrum`: read-only, built only by the spectrum.

    The full native pixel sequence is kept; invalid or masked pixels are only
    excluded through :attr:`usable`, never removed, so pixel adjacency stays
    true for correlated noise.

    Attributes
    ----------
    id
        Frame id (``"frame_0"`` for a single spectrum, the ``join`` key
        otherwise).
    obs
        Observed-frame wavelengths, strictly increasing, in the spectrum's
        unit.
    flux, error
        Flux density and its 1-sigma error per pixel.
    mask
        ``True`` on pixels excluded from fitting (constructor mask plus every
        ``exclude`` interval).
    resolving_power
        ``None``, a constant ``R``, or a ``(wavelength, R)`` table.
    noise
        The frame's :class:`~noobfriend.inference.spectrum.noise.Noise` model.
    source_2d
        The 2-D source for frames built by ``from_2d``, else ``None``.
    """

    id: str
    obs: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    mask: np.ndarray
    resolving_power: float | ResolvingPowerTable | None
    noise: Noise
    source_2d: Source2D | None

    @property
    def size(self) -> int:
        """Number of native pixels."""
        return int(self.obs.size)

    @property
    def valid(self) -> np.ndarray:
        """``True`` where flux and error are finite and the error positive."""
        valid = np.isfinite(self.flux) & np.isfinite(self.error) & (self.error > 0)
        valid.setflags(write=False)
        return valid

    @property
    def usable(self) -> np.ndarray:
        """``True`` on valid pixels that are not masked."""
        usable = self.valid & ~self.mask
        usable.setflags(write=False)
        return usable

    def resolving_power_at(self, wavelength: float) -> float | None:
        """Return ``R`` at one observed wavelength.

        Parameters
        ----------
        wavelength
            Observed wavelength in the spectrum's unit.

        Returns
        -------
        float or None
            The constant ``R``, the table interpolated linearly, or ``None``
            when the frame has no resolving power.

        Raises
        ------
        ValueError
            If ``wavelength`` lies outside a table's range.
        """
        if self.resolving_power is None or isinstance(self.resolving_power, float):
            return self.resolving_power
        table_wavelength, table_r = self.resolving_power
        if not table_wavelength[0] <= wavelength <= table_wavelength[-1]:
            raise ValueError(
                f"frame {self.id!r}: wavelength {wavelength:g} is outside its "
                f"resolving-power table [{table_wavelength[0]:g}, "
                f"{table_wavelength[-1]:g}]."
            )
        return float(np.interp(wavelength, table_wavelength, table_r))


def normalize_resolving_power(value: object) -> float | ResolvingPowerTable | None:
    """Validate a resolving power: ``None``, a positive number, or a table.

    Parameters
    ----------
    value
        ``None``, a number, or a ``(wavelength, R)`` pair of 1-D arrays.

    Returns
    -------
    float, tuple of numpy.ndarray, or None
        The normalized value; tables become read-only arrays.

    Raises
    ------
    TypeError
        If ``value`` is a callable or otherwise unsupported.
    ValueError
        If the number or table entries are not positive and finite, or the
        table wavelengths are not strictly increasing.
    """
    if value is None:
        return None
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"resolving_power must be positive; got {number!r}.")
        return number
    if callable(value) or not isinstance(value, Sequence) or len(value) != 2:
        raise TypeError(
            "resolving_power must be None, a number, or a (wavelength, R) table; "
            "functions are not accepted."
        )
    wavelength = readonly(value[0])
    r = readonly(value[1])
    if wavelength.ndim != 1 or wavelength.shape != r.shape or wavelength.size < 2:
        raise ValueError(
            "a resolving_power table needs two equal 1-D arrays of length >= 2."
        )
    if not np.all(np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0):
        raise ValueError(
            "resolving_power table wavelengths must be strictly increasing."
        )
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        raise ValueError("resolving_power table values must be positive.")
    return wavelength, r


def build_frame(
    frame_id: str,
    obs: ArrayLike,
    flux: ArrayLike,
    error: ArrayLike,
    *,
    mask: ArrayLike | None,
    resolving_power: object,
    source_2d: Source2D | None = None,
) -> SpectrumFrame:
    """Validate arrays and build a white-noise frame.

    Raises
    ------
    ValueError
        If shapes disagree, ``obs`` is not finite and strictly increasing, or
        no usable pixel remains.
    """
    wavelength = readonly(obs)
    flux_array = readonly(flux)
    error_array = readonly(error)
    if wavelength.ndim != 1:
        raise ValueError(f"obs must be 1-D; got shape {wavelength.shape}.")
    if flux_array.shape != wavelength.shape or error_array.shape != wavelength.shape:
        raise ValueError(
            f"obs {wavelength.shape}, flux {flux_array.shape} and error "
            f"{error_array.shape} must have equal shapes."
        )
    if not np.all(np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0):
        raise ValueError("obs must be finite and strictly increasing.")
    if mask is None:
        mask_array = readonly(np.zeros(wavelength.size), dtype=bool)
    else:
        mask_array = readonly(mask, dtype=bool)
        if mask_array.shape != wavelength.shape:
            raise ValueError(
                f"mask {mask_array.shape} must match obs {wavelength.shape}."
            )
    frame = SpectrumFrame(
        id=frame_id,
        obs=wavelength,
        flux=flux_array,
        error=error_array,
        mask=mask_array,
        resolving_power=normalize_resolving_power(resolving_power),
        noise=Noise(),
        source_2d=source_2d,
    )
    if not frame.usable.any():
        raise ValueError("the spectrum has no valid, unmasked pixel.")
    return frame


def check_frame_id(frame_id: object) -> str:
    """Validate a frame id (a non-empty string without ``.``)."""
    if not isinstance(frame_id, str) or not frame_id or "." in frame_id:
        raise ValueError(
            f"frame ids must be non-empty strings without '.'; got {frame_id!r}."
        )
    return frame_id
