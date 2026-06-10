"""Single-band aperture mask growth backed by :mod:`noobase.aperture`.

The noobase primitive accepts ``(row, col)`` seed pixels and strict native
``float64`` arrays. This module adapts that to noobfriend's user-facing
``(x, y)`` pixel convention, applies the growth-side rule that ``0.0`` is
invalid, and optionally restricts growth with a segmentation label map. The
mask it returns is binary and lives on the band's own native grid; combining
masks across bands is the caller's job (see
:mod:`noobfriend.extraction.photometry._coverage`).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from noobase.aperture import grow_mask

from noobfriend.extraction.photometry._array import native_float64, valid_data_mask

StopReason = Literal["snr_below", "gradient_flip", "filled"]


@dataclass(frozen=True)
class ApertureMask:
    """A source aperture grown on a single image.

    Attributes
    ----------
    mask : numpy.ndarray
        Boolean source mask on the band's native image grid.
    stop_reason : {"snr_below", "gradient_flip", "filled"}
        Termination reason reported by noobase.
    n_iterations : int
        Number of heap iterations reported by noobase.
    seed_xy : tuple[int, int]
        Rounded ``(x, y)`` seed pixel used for this band.
    """

    mask: np.ndarray
    stop_reason: StopReason
    n_iterations: int
    seed_xy: tuple[int, int]


def _round_xy(seed_xy: tuple[float, float]) -> tuple[int, int]:
    """Round ``(x, y)`` to integer pixel indices."""
    return int(np.round(seed_xy[0])), int(np.round(seed_xy[1]))


def _validate_seed(seed_xy: tuple[int, int], shape: tuple[int, int]) -> None:
    """Validate that a rounded ``(x, y)`` seed lies inside ``shape``."""
    x, y = seed_xy
    n_rows, n_cols = shape
    if x < 0 or x >= n_cols or y < 0 or y >= n_rows:
        raise ValueError(
            f"seed_xy {seed_xy!r} is outside image bounds "
            f"(width={n_cols}, height={n_rows})."
        )


def grow_aperture_mask(
    data: np.ndarray,
    *,
    seed_xy: tuple[float, float],
    error: np.ndarray | None = None,
    label_map: np.ndarray | None = None,
    label_allowed: Sequence[int] | None = None,
    allow_background: bool = True,
    grow_kwargs: Mapping[str, Any] | None = None,
) -> ApertureMask:
    """Grow one aperture mask from a band-local ``(x, y)`` seed.

    Parameters
    ----------
    data : numpy.ndarray
        2-D image. ``NaN`` and strict ``0.0`` pixels are treated as unavailable.
    seed_xy : tuple[float, float]
        Seed in image pixel coordinates, ordered as ``(x, y)``.
    error : numpy.ndarray, optional
        1-sigma error image for noobase's SNR stop.
    label_map : numpy.ndarray, optional
        Segmentation labels. When provided without ``label_allowed``, growth is
        restricted to the seed pixel's label (plus the background label, unless
        ``allow_background`` is ``False``).
    label_allowed : sequence of int, optional
        Labels from ``label_map`` allowed for aperture growth.
    allow_background : bool, default True
        When a ``label_map`` is given, also admit the background label ``0`` to
        the allowed set, so the aperture may grow off the source's own segment
        into surrounding sky (it is still blocked by *other* sources' labels)
        and is halted by the SNR/gradient stop rather than the segment edge.
        Set ``False`` to hard-confine growth to ``label_allowed`` / the seed's
        own label. No effect when ``label_map`` is ``None``.
    grow_kwargs : mapping, optional
        Additional keyword arguments forwarded to
        :func:`noobase.aperture.grow_mask`.

    Returns
    -------
    ApertureMask
        The grown aperture mask and noobase termination metadata.

    Raises
    ------
    ValueError
        If the image is not 2-D, shapes are inconsistent, the seed lies outside
        the image, or the seed is invalid / disallowed.
    """
    image = native_float64(data)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")

    seed_int = _round_xy(seed_xy)
    _validate_seed(seed_int, image.shape)
    seed_x, seed_y = seed_int

    allowed = valid_data_mask(image)
    if label_map is not None:
        labels = np.asarray(label_map)
        if labels.shape != image.shape:
            raise ValueError(
                f"label_map shape {labels.shape} does not match data shape "
                f"{image.shape}."
            )
        allowed_labels = (
            {int(labels[seed_y, seed_x])}
            if label_allowed is None
            else {int(label) for label in label_allowed}
        )
        if allow_background:
            allowed_labels.add(0)
        allowed &= np.isin(labels, list(allowed_labels))

    if not bool(allowed[seed_y, seed_x]):
        raise ValueError(
            f"seed_xy {seed_xy!r} rounds to {seed_int!r}, which is invalid "
            "or outside the allowed label region."
        )

    kwargs = dict(grow_kwargs or {})
    snr_disabled = "snr_threshold" in kwargs and kwargs["snr_threshold"] is None
    error_in = None
    if error is not None and not snr_disabled:
        error_in = native_float64(error)
        if error_in.shape != image.shape:
            raise ValueError(
                f"error shape {error_in.shape} does not match data shape {image.shape}."
            )
        error_in = np.where(allowed, error_in, np.nan)

    image_in = np.where(allowed, image, np.nan)
    binary_labels = np.ascontiguousarray(allowed.astype(np.int32))
    result = grow_mask(
        image_in,
        [(seed_y, seed_x)],
        err=error_in,
        label_map=binary_labels,
        label_allowed=[1],
        **kwargs,
    )
    return ApertureMask(
        mask=np.asarray(result.mask, dtype=bool),
        stop_reason=cast(StopReason, result.stop_reason),
        n_iterations=int(result.n_iterations),
        seed_xy=seed_int,
    )
