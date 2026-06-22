"""Normalized internal representation of one morphology band.

Callers pass each band as a plain mapping (``"data"`` / ``"error"`` plus optional
``"wcs"`` / ``"psf"`` / ``"mask"``); :func:`normalize_band` validates that mapping
once and freezes it into a :class:`Band`. Keeping the public surface
dictionary-based means a :class:`~noobfriend.inference.morphology.NoobImage` is
built from plain dicts, with no morphology type to import or construct -- the same
convention as :mod:`noobfriend.extraction.photometry`.

This module is internal. The :class:`BandSpec` TypedDict is re-exported from
:mod:`noobfriend.inference.morphology` as a static-checking aid; :class:`Band`
itself is not part of the public surface.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, TypedDict

import numpy as np


class BandSpec(TypedDict, total=False):
    """One morphology band as a plain dict, with statically-checked keys.

    Because this is a :class:`~typing.TypedDict`, a dict literal passed to
    :meth:`~noobfriend.inference.morphology.NoobImage.from_bands` is checked by
    editors for key names and value types -- no import or construction needed.
    ``data`` and ``error`` are required (enforced at runtime by
    :func:`normalize_band`); ``wcs`` is optional -- omit it to have a tangent-plane
    WCS synthesised from the target position and the cutout size.

    Keys
    ----
    data : numpy.ndarray
        2-D science image on the band's native grid (required).
    error : numpy.ndarray
        2-D 1-sigma error image, same shape as ``data`` (required).
    wcs : Any
        WCS exposing the noobfriend transform protocol; omit to synthesise one.
    psf : numpy.ndarray or None
        Oversampled PSF kernel for this band; omit to fall back to a theoretical
        PSF at fit time.
    mask : numpy.ndarray or None
        Boolean fit mask, same shape as ``data``, ``True`` where a pixel is
        *excluded* from the fit (valid data that is not a source being modelled);
        omit to have it auto-derived at fit time.
    """

    data: np.ndarray
    error: np.ndarray
    wcs: Any
    psf: np.ndarray | None
    mask: np.ndarray | None


#: Keys :func:`normalize_band` accepts; anything else is treated as a typo.
_BAND_KEYS: frozenset[str] = frozenset(BandSpec.__annotations__)


@dataclass(frozen=True)
class Band:
    """One morphology band on its own native pixel grid.

    Built by :func:`normalize_band`; not constructed directly by callers.

    Attributes
    ----------
    name : str
        Band name (the filter key it was given under).
    data, error : numpy.ndarray
        2-D science image and its 1-sigma error, equal shape, on the native grid.
    wcs : Any
        World-coordinate system exposing the noobfriend transform protocol.
    psf : numpy.ndarray or None
        Oversampled PSF kernel, or ``None`` (resolve a theoretical one at fit time).
    mask : numpy.ndarray or None
        Boolean fit mask (``True`` = excluded), or ``None`` (auto-derive at fit
        time). Invalid pixels (non-finite data or ``error <= 0``) are dropped from
        the likelihood independently of this mask.
    """

    name: str
    data: np.ndarray
    error: np.ndarray
    wcs: Any
    psf: np.ndarray | None
    mask: np.ndarray | None

    @property
    def shape(self) -> tuple[int, int]:
        """Return the native ``(rows, cols)`` image shape."""
        return self.data.shape  # type: ignore[return-value]

    def with_mask(self, mask: np.ndarray | None) -> Band:
        """Return a copy with the fit mask replaced.

        Parameters
        ----------
        mask : numpy.ndarray or None
            New boolean mask (``True`` = excluded), same shape as ``data``, or
            ``None`` to clear it (revert to auto-derivation at fit time).

        Returns
        -------
        Band

        Raises
        ------
        ValueError
            If ``mask`` is given but its shape does not match ``data``.
        """
        return replace(self, mask=_validated_mask(self.name, mask, self.data.shape))


def normalize_band(name: str, spec: Any) -> Band:
    """Validate one public band spec and freeze it into a :class:`Band`.

    WCS synthesis is deliberately *not* done here -- it needs the target position
    and cutout size, which live on :class:`~noobfriend.inference.morphology.NoobImage`;
    a missing ``"wcs"`` is left as ``None`` for the image layer to fill.

    Parameters
    ----------
    name : str
        Band name.
    spec : mapping
        A :class:`BandSpec`-shaped mapping: required ``"data"`` and ``"error"``
        plus optional ``"wcs"``, ``"psf"``, ``"mask"``. Unknown keys are rejected.

    Returns
    -------
    Band
        A frozen band, possibly with ``wcs=None`` (to be synthesised upstream).

    Raises
    ------
    ValueError
        If an unknown key is present, ``"data"`` or ``"error"`` is missing or
        malformed, or ``"error"`` / ``"mask"`` shapes do not match ``"data"``.
    """
    if not isinstance(spec, dict):
        raise ValueError(
            f"band {name!r} spec must be a mapping; got {type(spec).__name__}."
        )
    unknown = set(spec) - _BAND_KEYS
    if unknown:
        raise ValueError(
            f"band {name!r} has unknown key(s) {sorted(unknown)}; "
            f"allowed keys are {sorted(_BAND_KEYS)}."
        )
    if "data" not in spec:
        raise ValueError(f"band {name!r} is missing required key 'data'.")
    if spec.get("error") is None:
        raise ValueError(f"band {name!r} is missing required key 'error'.")

    data = np.asarray(spec["data"], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"band {name!r} data must be 2-D; got shape {data.shape}.")

    error = np.asarray(spec["error"], dtype=float)
    if error.shape != data.shape:
        raise ValueError(
            f"band {name!r} error shape {error.shape} does not match data "
            f"shape {data.shape}."
        )

    psf = spec.get("psf")
    if psf is not None:
        psf = np.asarray(psf, dtype=float)
        if psf.ndim != 2:
            raise ValueError(f"band {name!r} psf must be 2-D; got shape {psf.shape}.")

    mask = _validated_mask(name, spec.get("mask"), data.shape)

    return Band(
        name=str(name), data=data, error=error, wcs=spec.get("wcs"), psf=psf, mask=mask
    )


def _validated_mask(name: str, mask: Any, shape: tuple[int, int]) -> np.ndarray | None:
    """Validate an optional boolean fit mask against the data shape."""
    if mask is None:
        return None
    out = np.asarray(mask, dtype=bool)
    if out.shape != shape:
        raise ValueError(
            f"band {name!r} mask shape {out.shape} does not match data shape {shape}."
        )
    return out
