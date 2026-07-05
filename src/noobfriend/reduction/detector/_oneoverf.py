"""Custom 1/f (correlated read) noise removal for NIRCam rate frames.

Pure, array-space implementation of the rate-level destriping step inserted
before flat-fielding in the stage-2 reduction. NIRCam detectors imprint a
slowly varying, read-correlated noise pattern ("1/f striping") plus a
per-amplifier pedestal; both are *additive* in the un-flat-fielded count-rate
image, so they are removed here by subtracting a source-masked robust offset
along rows and/or columns, optionally independently per amplifier column block.

The function takes and returns plain ``(data, err, dq)`` arrays and never
touches a datamodel, FITS header or the :mod:`jwst` package -- the calling
reduction script unwraps the datamodel, calls this, and writes the corrected
``data`` back. ``err`` and ``dq`` pass through unchanged: 1/f removal is an
additive-pattern subtraction that neither rescales the per-pixel noise nor
reflags pixels.
"""

import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from noobfriend.reduction.detector._masking import source_exclusion

_Floats = NDArray[np.floating]
_Ints = NDArray[np.integer]


def subtract_oneoverf(
    data: _Floats,
    err: _Floats,
    dq: _Ints,
    *,
    mask: NDArray[np.bool_] | None = None,
    by: Literal["row", "column", "both"] = "both",
    amp_aware: bool = True,
    n_amps: int = 4,
    nsigma: float = 3.0,
    dilate: int = 3,
    dq_bad_bits: int = 1,
    min_samples: int = 5,
) -> tuple[_Floats, _Floats, _Ints]:
    """Subtract the 1/f striping and amplifier pedestal from a rate frame.

    A robust (median) offset is estimated from the *background* pixels -- those
    that are finite, not flagged in ``dq``, and not covered by a source -- and
    subtracted along rows and/or columns. With ``amp_aware`` the estimate is made
    independently within each amplifier column block, which also removes the
    per-amplifier pedestal. Pixels excluded from the estimate are still corrected
    (the offset is subtracted from the whole block), so source flux is preserved.

    Parameters
    ----------
    data : numpy.ndarray
        2-D count-rate image to correct (e.g. a ``_rate`` ``SCI`` array). Not
        modified in place; a corrected copy is returned.
    err, dq : numpy.ndarray
        The matching error and data-quality arrays. They are returned unchanged
        and are part of the signature only so every reduction step shares the
        same ``(data, err, dq)`` contract; ``dq`` is read to mask bad pixels.
    mask : numpy.ndarray of bool, optional
        Pixels to *exclude* from the background estimate (``True`` = exclude),
        e.g. a source or grism-trace mask supplied by the caller. When ``None``
        (default), a source mask is derived from ``data`` with
        :func:`~noobfriend.core.imgutils.segment` and dilated by ``dilate``.
    by : {"row", "column", "both"}, default "both"
        Direction(s) along which to subtract the offset. ``"both"`` subtracts the
        row offset first, then re-estimates and subtracts the column offset; it is
        the default because real NIRCam frames carry striping in *both* directions
        (a single direction leaves a residual orthogonal grid). Note that genuine
        *diagonal* striping -- 1/f correlated along the read order -- is removed by
        neither and needs a read-order-aware step.
    amp_aware : bool, default True
        When ``True``, split the columns into ``n_amps`` equal blocks and correct
        each independently (removes the per-amplifier pedestal); when ``False``,
        treat the full width as one block.
    n_amps : int, default 4
        Number of amplifier column blocks (4 for a NIRCam ``FULL`` frame). The
        image width must be divisible by ``n_amps`` when ``amp_aware`` is True.
    nsigma : float, default 3.0
        Detection level (robust sigmas above the median) for the automatic source
        mask; used only when ``mask`` is ``None``.
    dilate : int, default 3
        Number of binary-dilation iterations grown around the automatic source
        mask to catch faint wings; used only when ``mask`` is ``None``.
    dq_bad_bits : int, default 1
        Bitmask of ``dq`` flags marking pixels to exclude from the estimate. The
        default ``1`` is the JWST ``DO_NOT_USE`` bit; pass ``0`` to ignore ``dq``.
    min_samples : int, default 5
        Minimum number of background pixels a row/column block must contain for
        its offset to be estimated; below this the offset is left at ``0`` (no
        correction) to avoid subtracting a noise-dominated value.

    Returns
    -------
    data : numpy.ndarray
        The corrected count-rate image (a new array).
    err, dq : numpy.ndarray
        The inputs ``err`` and ``dq``, unchanged.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D, ``by`` is not one of the accepted values, or the
        width is not divisible by ``n_amps`` while ``amp_aware`` is True.
    """
    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    if by not in ("row", "column", "both"):
        raise ValueError(f"by must be 'row', 'column' or 'both'; got {by!r}.")

    excluded = source_exclusion(
        image, dq, mask=mask, nsigma=nsigma, dilate=dilate, dq_bad_bits=dq_bad_bits
    )

    n_blocks = n_amps if amp_aware else 1
    width = image.shape[1]
    if amp_aware and width % n_blocks != 0:
        raise ValueError(
            f"width {width} is not divisible by n_amps {n_amps}; "
            "pass amp_aware=False or a matching n_amps."
        )

    corrected = image.copy()
    block_width = width // n_blocks
    for b in range(n_blocks):
        cols = slice(b * block_width, (b + 1) * block_width)
        block_excl = excluded[:, cols]
        if by in ("row", "both"):
            vals = np.where(block_excl, np.nan, corrected[:, cols])
            corrected[:, cols] -= _safe_median(vals, axis=1, min_samples=min_samples)[
                :, None
            ]
        if by in ("column", "both"):
            vals = np.where(block_excl, np.nan, corrected[:, cols])
            corrected[:, cols] -= _safe_median(vals, axis=0, min_samples=min_samples)[
                None, :
            ]

    return corrected, err, dq


def _safe_median(values: _Floats, *, axis: int, min_samples: int) -> _Floats:
    """Robust ``nanmedian`` along ``axis``, falling back to ``0`` when sparse.

    Slices with fewer than ``min_samples`` finite values yield ``0`` rather than
    a noise-dominated (or ``NaN``) offset, so a row/column dominated by sources or
    bad pixels is simply left uncorrected.
    """
    count = np.isfinite(values).sum(axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(values, axis=axis)
    return np.where(count >= min_samples, np.nan_to_num(median, nan=0.0), 0.0)
