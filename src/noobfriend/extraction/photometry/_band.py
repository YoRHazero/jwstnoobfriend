"""Normalized internal representation of one photometry band.

Callers pass each band as a plain mapping (``"data"`` / ``"wcs"`` plus optional
extras); :func:`normalize_band` validates that mapping once and freezes it into
a :class:`Band`. Keeping the public surface dictionary-based means producers
such as :func:`noobfriend.core.io.load_grizli_cutout` do not have to import or
construct any photometry type.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Band:
    """One photometry band on its own native pixel grid.

    Attributes
    ----------
    name : str
        Band name.
    data : numpy.ndarray
        2-D science image on the band's native grid.
    wcs : Any
        World-coordinate system exposing the noobfriend transform protocol.
    error : numpy.ndarray or None
        1-sigma error image, or ``None``.
    wavelength : float or None
        Effective wavelength in microns, or ``None``.
    wavelength_error : tuple[float, float] or None
        Left/right wavelength uncertainty, or ``None``.
    flux_scale_mjy : float or None
        Multiplicative scale from one raw per-pixel value to mJy, or ``None``.
    flux_unit : str or None
        Original image unit or calibration source behind the scale.
    label_map : numpy.ndarray or None
        Segmentation labels restricting aperture growth, or ``None``.
    label_allowed : tuple[int, ...] or None
        Labels allowed for aperture growth, or ``None``.
    """

    name: str
    data: np.ndarray
    wcs: Any
    error: np.ndarray | None
    wavelength: float | None
    wavelength_error: tuple[float, float] | None
    flux_scale_mjy: float | None
    flux_unit: str | None
    label_map: np.ndarray | None
    label_allowed: tuple[int, ...] | None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the native ``(rows, cols)`` image shape."""
        return self.data.shape  # type: ignore[return-value]


def normalize_band(name: str, spec: Mapping[str, Any]) -> Band:
    """Validate one public band spec and freeze it into a :class:`Band`.

    Parameters
    ----------
    name : str
        Band name.
    spec : mapping
        Plain mapping with required ``"data"`` and ``"wcs"`` entries, plus
        optional ``"error"``, ``"wavelength"``, ``"wavelength_error"``,
        ``"flux_scale_mjy"``, ``"flux_unit"``, ``"label_map"``, and
        ``"label_allowed"``. ``wavelength_error`` must be a tuple of length 1
        or 2; ``(value,)`` is read as symmetric.

    Returns
    -------
    Band

    Raises
    ------
    ValueError
        If a required key is missing or any field is malformed.
    """
    if "data" not in spec:
        raise ValueError(f"band {name!r} is missing required key 'data'.")
    if "wcs" not in spec:
        raise ValueError(f"band {name!r} is missing required key 'wcs'.")

    data = np.asarray(spec["data"])
    if data.ndim != 2:
        raise ValueError(f"band {name!r} data must be 2-D; got shape {data.shape}.")

    error = None if spec.get("error") is None else np.asarray(spec["error"])
    if error is not None and error.shape != data.shape:
        raise ValueError(
            f"band {name!r} error shape {error.shape} does not match data "
            f"shape {data.shape}."
        )

    label_map = None if spec.get("label_map") is None else np.asarray(spec["label_map"])
    if label_map is not None and label_map.shape != data.shape:
        raise ValueError(
            f"band {name!r} label_map shape {label_map.shape} does not match "
            f"data shape {data.shape}."
        )

    label_allowed = spec.get("label_allowed")
    label_allowed_out = (
        None if label_allowed is None else tuple(int(label) for label in label_allowed)
    )

    wavelength = None if spec.get("wavelength") is None else float(spec["wavelength"])
    wavelength_error = _normalize_wavelength_error(name, spec.get("wavelength_error"))
    if wavelength_error is not None and wavelength is None:
        raise ValueError(f"band {name!r} provides wavelength_error but no wavelength.")

    flux_scale_mjy = (
        None if spec.get("flux_scale_mjy") is None else float(spec["flux_scale_mjy"])
    )
    if flux_scale_mjy is not None and not np.isfinite(flux_scale_mjy):
        raise ValueError(f"band {name!r} flux_scale_mjy must be finite.")
    flux_unit = None if spec.get("flux_unit") is None else str(spec["flux_unit"])

    return Band(
        name=str(name),
        data=data,
        wcs=spec["wcs"],
        error=error,
        wavelength=wavelength,
        wavelength_error=wavelength_error,
        flux_scale_mjy=flux_scale_mjy,
        flux_unit=flux_unit,
        label_map=label_map,
        label_allowed=label_allowed_out,
    )


def _normalize_wavelength_error(band: str, value: object) -> tuple[float, float] | None:
    """Normalize a wavelength-error tuple to ``(left, right)``."""
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise ValueError(
            f"band {band!r} wavelength_error must be a tuple of length 1 or 2."
        )
    if len(value) == 1:
        left = right = float(value[0])
    elif len(value) == 2:
        left, right = float(value[0]), float(value[1])
    else:
        raise ValueError(
            f"band {band!r} wavelength_error must have length 1 or 2; got {len(value)}."
        )
    if not (np.isfinite(left) and np.isfinite(right) and left >= 0 and right >= 0):
        raise ValueError(
            f"band {band!r} wavelength_error values must be finite and non-negative."
        )
    return (left, right)
