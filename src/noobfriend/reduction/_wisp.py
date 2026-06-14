"""Custom wisp (scattered-light) subtraction for NIRCam SW frames.

Certain NIRCam short-wavelength detectors (A3, A4, B3, B4) show fixed scattered
-light "wisp" features near ~2 um that the official pipeline does not remove. A
per-detector/filter wisp *template* (a normalised image of the feature, supplied
by the caller) is scaled to the frame and subtracted. The scale is fit as the
slope of a robust linear regression of the frame on the template over the
background pixels, so the flat sky (the intercept) is left for the background
step and only the template-shaped component is removed.

Pure ``(data, err, dq)`` arrays in and out; ``err`` / ``dq`` pass through.
"""

import numpy as np
from numpy.typing import NDArray

from noobfriend.reduction._masking import source_exclusion

_Floats = NDArray[np.floating]
_Ints = NDArray[np.integer]


def subtract_wisp(
    data: _Floats,
    err: _Floats,
    dq: _Ints,
    *,
    template: _Floats,
    scale: float | None = None,
    mask: NDArray[np.bool_] | None = None,
    nsigma: float = 3.0,
    dilate: int = 5,
    dq_bad_bits: int = 1,
) -> tuple[_Floats, _Floats, _Ints]:
    """Subtract a scaled wisp ``template`` from a NIRCam SW frame.

    Parameters
    ----------
    data : numpy.ndarray
        2-D frame to correct. Not modified in place; a corrected copy is returned.
    err, dq : numpy.ndarray
        Matching error and data-quality arrays, returned unchanged; ``dq`` is read
        to mask bad pixels when fitting the scale.
    template : numpy.ndarray
        Wisp template, same shape as ``data`` -- a (typically normalised) image of
        the scattered-light feature for this detector / filter.
    scale : float, optional
        Amplitude to subtract (``data - scale * template``). When ``None``
        (default) it is fit as the slope of a robust linear regression of ``data``
        on ``template`` over the background pixels, clipped to be non-negative.
    mask : numpy.ndarray of bool, optional
        Pixels to exclude from the scale fit (``True`` = exclude). When ``None``
        a source mask is derived from ``data`` and grown by ``dilate``.
    nsigma, dilate, dq_bad_bits
        Forwarded to the source mask (see
        :func:`~noobfriend.reduction._masking.source_exclusion`); used only when
        ``scale`` is ``None``.

    Returns
    -------
    data : numpy.ndarray
        The wisp-subtracted frame (a new array).
    err, dq : numpy.ndarray
        The inputs ``err`` and ``dq``, unchanged.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D or ``template`` does not match its shape.
    """
    image = np.asarray(data, dtype=float)
    wisp = np.asarray(template, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    if wisp.shape != image.shape:
        raise ValueError(
            f"template shape {wisp.shape} does not match data shape {image.shape}."
        )

    if scale is None:
        excluded = source_exclusion(
            image, dq, mask=mask, nsigma=nsigma, dilate=dilate, dq_bad_bits=dq_bad_bits
        )
        scale = _fit_scale(image, wisp, excluded)
    return image - scale * wisp, err, dq


def _fit_scale(
    image: _Floats, template: _Floats, excluded: NDArray[np.bool_]
) -> float:
    """Return the non-negative regression slope of ``image`` on ``template``.

    Fit over the background (non-excluded, finite) pixels with both arrays
    mean-centred, so the constant sky cancels and the slope measures the
    template-shaped (wisp) amplitude.
    """
    good = ~excluded & np.isfinite(image) & np.isfinite(template)
    if good.sum() < 2:
        return 0.0
    d = image[good]
    t = template[good]
    t_centered = t - t.mean()
    denom = float((t_centered * t_centered).sum())
    if denom <= 0.0:
        return 0.0
    slope = float((t_centered * (d - d.mean())).sum() / denom)
    return max(slope, 0.0)
