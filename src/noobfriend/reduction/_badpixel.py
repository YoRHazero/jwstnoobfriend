"""Custom outlier / bad-pixel flagging for reduced frames.

Flags isolated outlier pixels (hot, warm, cold, residual cosmic-ray or snowball
-halo pixels) that the CRDS bad-pixel mask and the detector1 jump step miss.
Working only in the *background* (sources are protected), a pixel is flagged when
it deviates from a 3x3 median-filtered version by more than ``nsigma`` robust
sigmas. The flag is OR-ed into ``dq``; the science pixels of real sources are
left untouched.

Pure ``(data, err, dq)`` arrays in and out: ``data`` is unchanged unless
``set_nan`` is requested, ``err`` passes through, and only ``dq`` gains flags.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

from noobfriend.reduction._masking import source_exclusion

_Floats = NDArray[np.floating]
_Ints = NDArray[np.integer]


def flag_outlier_pixels(
    data: _Floats,
    err: _Floats,
    dq: _Ints,
    *,
    mask: NDArray[np.bool_] | None = None,
    nsigma: float = 5.0,
    dq_flag: int = 1,
    set_nan: bool = False,
    nsigma_source: float = 3.0,
    dilate: int = 5,
    dq_bad_bits: int = 1,
) -> tuple[_Floats, _Ints, _Ints]:
    """Flag background pixels deviating strongly from their local median.

    Parameters
    ----------
    data : numpy.ndarray
        2-D frame to scan.
    err, dq : numpy.ndarray
        Matching error and data-quality arrays; ``err`` passes through and ``dq``
        is read (to skip already-flagged pixels) and returned with new flags.
    mask : numpy.ndarray of bool, optional
        Pixels to protect / exclude from detection (``True`` = protect). When
        ``None`` a source mask is derived from ``data`` so real sources are not
        flagged.
    nsigma : float, default 5.0
        Detection level: a background pixel is flagged when its deviation from the
        3x3 median exceeds ``nsigma`` times the robust sigma of those deviations.
    dq_flag : int, default 1
        Bit OR-ed into ``dq`` for flagged pixels (``1`` = JWST ``DO_NOT_USE``).
    set_nan : bool, default False
        When ``True``, also set flagged pixels in the returned ``data`` to ``NaN``.
    nsigma_source, dilate, dq_bad_bits
        Forwarded to the protecting source mask (see
        :func:`~noobfriend.reduction._masking.source_exclusion`); used only when
        ``mask`` is ``None``.

    Returns
    -------
    data : numpy.ndarray
        The input frame, unchanged unless ``set_nan`` flagged pixels to ``NaN``.
    err : numpy.ndarray
        The input ``err``, unchanged.
    dq : numpy.ndarray
        A copy of ``dq`` with ``dq_flag`` OR-ed in at flagged pixels.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D.
    """
    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")

    protected = source_exclusion(
        image,
        dq,
        mask=mask,
        nsigma=nsigma_source,
        dilate=dilate,
        dq_bad_bits=dq_bad_bits,
    )
    residual = image - ndi.median_filter(image, size=3, mode="nearest")
    background = residual[~protected & np.isfinite(residual)]
    if background.size < 2:
        return image, err, np.asarray(dq)
    sigma = 1.4826 * float(np.median(np.abs(background - np.median(background))))

    outlier = (
        np.isfinite(residual) & ~protected & (np.abs(residual) > nsigma * sigma)
        if sigma > 0
        else np.zeros(image.shape, dtype=bool)
    )
    new_dq = np.asarray(dq).copy()
    new_dq[outlier] |= dq_flag
    out = np.where(outlier, np.nan, image) if set_nan else image
    return out, err, new_dq
