"""Per-detector fixed-pattern ("crosshatch") template for NIRCam imaging frames.

NIRCam detectors imprint a faint *detector-locked* fine-structure pattern -- a
diagonal crosshatch plus axial banding -- that the per-row/column 1/f step
(:func:`~noobfriend.reduction.subtract_oneoverf`) cannot remove: the diagonal
has ~zero net row/column projection. Stacking many dithered exposures of one
detector in detector coordinates averages the (moving) sky away and reinforces
this fixed component; a split-half test confirms it reproduces (so it is a
genuine template, not per-exposure 1/f noise), and it survives flat-fielding
nearly unchanged (so it is *additive* in count rate and belongs at the rate
stage, before flat).

Four pure-array helpers, jwst/navigation-free, used by an imaging runner:

- :func:`fixed_pattern_residual` -- per frame (pass 1): detrend (remove the smooth
  sky with the same mesh background as :func:`~noobfriend.reduction.subtract_background`)
  and source-mask, leaving the fine fixed structure at full resolution. Unlike the
  smooth sky template the structure cannot be downsampled, so the residual is kept
  full-size.
- :func:`combine_fixed_pattern` -- after pass 1: cross-frame median of the residuals
  into one zero-mean, full-resolution template per detector.
- :func:`fit_pattern_amplitude` / :func:`subtract_fixed_pattern` -- per frame
  (pass 2): fit the per-frame amplitude (the pattern reproduces only partially,
  so a fixed-amplitude subtraction over-subtracts) and subtract it. Apply this
  *before* ``subtract_oneoverf`` and before flat-fielding.

The GRISM path reuses the same cross-frame median combiner but builds a different
per-frame residual: it keeps the full detector-fixed low-frequency gradient plus
fine pattern, masks traces, and removes only the per-frame scalar median. Its
subtraction fits ``alpha * template + beta`` over trace-free pixels and subtracts
only ``alpha * template``; the fitted ``beta`` is a nuisance level, not a detector
correction.
"""

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from noobfriend.reduction._background import _mesh_background
from noobfriend.reduction._masking import source_exclusion

_Floats = NDArray[np.floating]
_Ints = NDArray[np.integer]
_Bool = NDArray[np.bool_]


def fixed_pattern_residual(
    data: _Floats,
    dq: _Ints,
    *,
    mask: _Bool | None = None,
    box_size: int = 128,
    filter_size: int = 3,
    min_valid_fraction: float = 0.1,
    nsigma: float = 3.0,
    dilate: int = 3,
    dq_bad_bits: int = 1,
) -> _Floats:
    """Return one frame's detrended, source-masked residual for template building.

    A smooth 2-D background (the box-median mesh of
    :func:`~noobfriend.reduction.subtract_background`) is removed so only the fine
    fixed structure remains, and source / flagged / non-finite pixels are set to
    ``NaN`` so they are skipped by the later cross-frame median. The result is kept
    at full resolution -- the crosshatch is fine structure that downsampling would
    destroy.

    Parameters
    ----------
    data : numpy.ndarray
        2-D rate frame (pre-1/f, pre-flat ``_rate`` ``SCI`` array, in DN/s).
    dq : numpy.ndarray
        Matching data-quality array, read to mask bad pixels.
    mask : numpy.ndarray of bool, optional
        Pixels to exclude (``True`` = exclude). When ``None`` (default) a source
        mask is derived from ``data`` and grown by ``dilate``.
    box_size : int, default 128
        Side length (pixels) of the mesh-background boxes; large enough that the
        fine pattern survives the detrend, small enough to follow the smooth sky.
    filter_size : int, default 3
        Side length (in boxes) of the median filter smoothing the background grid.
    min_valid_fraction : float, default 0.1
        Minimum background fraction for a box's median to be trusted; sparser
        boxes are inpainted from their neighbours.
    nsigma : float, default 3.0
        Source-detection level for the automatic mask (used only when ``mask`` is
        ``None``). Kept above the sub-sigma pattern so the stripes are not masked.
    dilate : int, default 3
        Binary-dilation iterations grown around the automatic source mask.
    dq_bad_bits : int, default 1
        Bitmask of ``dq`` flags marking pixels to exclude (``1`` = ``DO_NOT_USE``).

    Returns
    -------
    numpy.ndarray
        The full-resolution residual; excluded pixels are ``NaN``.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D or ``box_size`` is not positive.
    """
    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    if box_size < 1:
        raise ValueError(f"box_size must be >= 1; got {box_size}.")

    excluded = source_exclusion(
        image, dq, mask=mask, nsigma=nsigma, dilate=dilate, dq_bad_bits=dq_bad_bits
    )
    background = _mesh_background(
        image, excluded, box_size, filter_size, min_valid_fraction
    )
    residual = image - background
    residual[excluded] = np.nan
    return residual


def grism_fixed_pattern_frame(
    data: _Floats,
    dq: _Ints,
    *,
    mask: _Bool | None = None,
    subtract_median: bool = True,
    dq_bad_bits: int = 1,
) -> _Floats:
    """Return one GRISM frame for unified fixed-pattern template building.

    Unlike :func:`fixed_pattern_residual`, this intentionally does *not* remove a
    mesh/row/column/low-pass background. For GRISM the smooth detector-locked
    gradient and the fine diagonal pattern are treated as one additive detector
    template; only trace/source/bad/non-finite pixels are set to ``NaN`` and the
    frame's global median is removed so pointing-dependent DC sky level does not
    enter the detector template.

    Parameters
    ----------
    data, dq : numpy.ndarray
        2-D rate frame and matching data-quality array.
    mask : numpy.ndarray of bool, optional
        Pixels to exclude from the template (``True`` = exclude), usually from
        :func:`~noobfriend.reduction.grism_trace_mask`.
    subtract_median : bool, default True
        Remove the masked-frame scalar median before combining frames.
    dq_bad_bits : int, default 1
        Bitmask of ``dq`` flags marking pixels to exclude (``1`` = ``DO_NOT_USE``).

    Returns
    -------
    numpy.ndarray
        A full-resolution frame with excluded pixels set to ``NaN``.
    """
    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    excluded = _explicit_exclusion(image, dq, mask=mask, dq_bad_bits=dq_bad_bits)
    frame = image.copy()
    frame[excluded] = np.nan
    if subtract_median:
        finite = np.isfinite(frame)
        if finite.any():
            frame -= np.nanmedian(frame)
    return frame


def combine_fixed_pattern(
    residuals: Sequence[_Floats], *, min_coverage: float = 0.2, tile_rows: int = 256
) -> _Floats:
    """Combine per-frame residuals into one zero-mean fixed-pattern template.

    The residuals (from :func:`fixed_pattern_residual`, all the same shape) are
    median-combined across frames -- the cross-frame median rejecting the dithered
    sources and outliers the per-frame masking missed. Pixels covered by fewer
    than ``min_coverage`` of the frames are set to ``0`` (no correction there
    rather than a noise-dominated value), and the template is shifted to zero
    median over its covered pixels so subtracting it does not move the background.

    The median is taken in blocks of ``tile_rows`` rows in ``float32``, so peak
    working memory is ``n_frames * tile_rows * width * 4`` bytes, not the whole
    ``n_frames * height * width`` stack at once. For stacks many times the FRESCO
    depth, pass the residuals as on-disk memory maps
    (``numpy.load(path, mmap_mode="r")``) so only the active block is resident and
    memory stays bounded regardless of frame count.

    Parameters
    ----------
    residuals : sequence of numpy.ndarray
        Per-frame residuals of one detector, ``NaN`` where masked, all the same
        2-D shape. May be memory-mapped arrays.
    min_coverage : float, default 0.2
        Minimum fraction of frames that must contribute a finite value to a pixel
        for its template value to be trusted.
    tile_rows : int, default 256
        Number of image rows combined per block -- the memory/speed knob. Smaller
        bounds memory tighter; it does not change the result.

    Returns
    -------
    numpy.ndarray
        The full-resolution, zero-mean ``float32`` fixed-pattern template (``0``
        where coverage is insufficient).

    Raises
    ------
    ValueError
        If ``residuals`` is empty, ``tile_rows`` is not positive, or the residuals
        are not all the same 2-D shape.
    """
    n = len(residuals)
    if n == 0:
        raise ValueError("residuals is empty; need at least one residual frame.")
    if tile_rows < 1:
        raise ValueError(f"tile_rows must be >= 1; got {tile_rows}.")
    shape = np.asarray(residuals[0]).shape
    if len(shape) != 2 or any(np.asarray(r).shape != shape for r in residuals):
        raise ValueError("all residuals must share the same 2-D shape.")
    height, width = shape
    template = np.zeros(shape, dtype=np.float32)
    coverage = np.zeros(shape, dtype=np.float32)
    for y0 in range(0, height, tile_rows):
        y1 = min(y0 + tile_rows, height)
        block = np.empty((n, y1 - y0, width), dtype=np.float32)
        for i, residual in enumerate(residuals):
            block[i] = np.asarray(residual[y0:y1], dtype=np.float32)
        cov = np.isfinite(block).mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN pixels
            median = np.nanmedian(block, axis=0)
        coverage[y0:y1] = cov
        template[y0:y1] = np.where(
            cov >= min_coverage, np.nan_to_num(median, nan=0.0), 0.0
        )
    covered = coverage >= min_coverage
    if covered.any():
        template[covered] -= np.float32(np.median(template[covered]))
    return template


def fit_pattern_amplitude(
    data: _Floats,
    dq: _Ints,
    template: _Floats,
    *,
    mask: _Bool | None = None,
    nsigma: float = 3.0,
    dilate: int = 3,
    dq_bad_bits: int = 1,
) -> float:
    """Return the per-frame amplitude ``a`` minimising ``data - a * template``.

    The least-squares scale is fit over the background pixels (sources, flagged
    and non-finite pixels excluded) with both arrays mean-subtracted first, so a
    residual sky level does not bias the projection. The fitted ``a`` is below one
    when the template (built from many frames) over-represents this frame's
    realisation -- subtracting ``a * template`` then avoids over-subtracting.

    Parameters
    ----------
    data : numpy.ndarray
        2-D frame the template is fit to (same rate stage the template was built at).
    dq : numpy.ndarray
        Matching data-quality array, read to mask bad pixels.
    template : numpy.ndarray
        The fixed-pattern template from :func:`combine_fixed_pattern`, same shape.
    mask : numpy.ndarray of bool, optional
        Extra pixels to exclude from the fit (``True`` = exclude); combined with
        the automatic source mask.
    nsigma : float, default 3.0
        Source-detection level for the automatic mask.
    dilate : int, default 3
        Binary-dilation iterations grown around the automatic source mask.
    dq_bad_bits : int, default 1
        Bitmask of ``dq`` flags marking pixels to exclude (``1`` = ``DO_NOT_USE``).

    Returns
    -------
    float
        The amplitude, or ``0.0`` when the template carries no power over the fit
        pixels (then no correction is applied).
    """
    image = np.asarray(data, dtype=float)
    tmpl = np.asarray(template, dtype=float)
    excluded = source_exclusion(
        image, dq, mask=mask, nsigma=nsigma, dilate=dilate, dq_bad_bits=dq_bad_bits
    )
    good = ~excluded & np.isfinite(image) & np.isfinite(tmpl)
    if good.sum() < 2:
        return 0.0
    d = image[good] - image[good].mean()
    t = tmpl[good] - tmpl[good].mean()
    denom = float(t @ t)
    if denom <= 0:
        return 0.0
    return float(d @ t / denom)


def fit_pattern_amplitude_offset(
    data: _Floats,
    dq: _Ints,
    template: _Floats,
    *,
    mask: _Bool | None = None,
    dq_bad_bits: int = 1,
) -> tuple[float, float]:
    """Fit ``data ~= alpha * template + beta`` over explicitly unmasked pixels.

    This is the GRISM fit used after building a unified low+high-frequency
    detector template. The scalar ``beta`` absorbs the exposure's sky level so a
    pointing-dependent DC background does not bias the template amplitude; callers
    subtract only ``alpha * template``.

    Returns
    -------
    tuple[float, float]
        ``(alpha, beta)``. ``alpha`` is ``0`` when too few pixels or no template
        power are available.
    """
    image = np.asarray(data, dtype=float)
    tmpl = np.asarray(template, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    if image.shape != tmpl.shape:
        raise ValueError(
            f"template shape {tmpl.shape} does not match data shape {image.shape}."
        )
    excluded = _explicit_exclusion(
        image, dq, mask=mask, dq_bad_bits=dq_bad_bits
    ) | ~np.isfinite(tmpl)
    good = ~excluded
    if good.sum() < 2:
        finite = image[np.isfinite(image)]
        beta = float(np.median(finite)) if finite.size else 0.0
        return 0.0, beta

    d = image[good]
    t = tmpl[good]
    d_mean = float(d.mean())
    t_mean = float(t.mean())
    d0 = d - d_mean
    t0 = t - t_mean
    denom = float(t0 @ t0)
    if denom <= 0:
        return 0.0, d_mean
    alpha = float(d0 @ t0 / denom)
    beta = d_mean - alpha * t_mean
    return alpha, beta


def subtract_fixed_pattern(
    data: _Floats,
    err: _Floats,
    dq: _Ints,
    template: _Floats,
    *,
    mask: _Bool | None = None,
    alpha: float | None = None,
    nsigma: float = 3.0,
    dilate: int = 3,
    dq_bad_bits: int = 1,
) -> tuple[_Floats, _Floats, _Ints]:
    """Subtract the (per-frame amplitude-fit) fixed-pattern template from a frame.

    The detector-fixed crosshatch is additive in count rate, so this runs at the
    rate stage -- *before* :func:`~noobfriend.reduction.subtract_oneoverf` and
    before flat-fielding. ``err`` / ``dq`` pass through: subtracting a
    deterministic template adds no noise and reflags no pixels.

    Parameters
    ----------
    data : numpy.ndarray
        2-D rate frame to correct. Not modified in place; a corrected copy is
        returned.
    err, dq : numpy.ndarray
        Matching error and data-quality arrays, returned unchanged; ``dq`` is read
        to mask bad pixels for the amplitude fit.
    template : numpy.ndarray
        The detector's fixed-pattern template from :func:`combine_fixed_pattern`,
        the same shape as ``data``.
    mask : numpy.ndarray of bool, optional
        Extra pixels to exclude from the amplitude fit (``True`` = exclude).
    alpha : float, optional
        Amplitude to subtract. When ``None`` (default) it is fit per frame with
        :func:`fit_pattern_amplitude`; pass a float to use a fixed scale instead.
    nsigma : float, default 3.0
        Source-detection level for the automatic fit mask.
    dilate : int, default 3
        Binary-dilation iterations grown around the automatic source mask.
    dq_bad_bits : int, default 1
        Bitmask of ``dq`` flags marking pixels to exclude from the fit.

    Returns
    -------
    data : numpy.ndarray
        The pattern-subtracted frame (a new array).
    err, dq : numpy.ndarray
        The inputs ``err`` and ``dq``, unchanged.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D or its shape does not match ``template``.
    """
    image = np.asarray(data, dtype=float)
    tmpl = np.asarray(template, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    if image.shape != tmpl.shape:
        raise ValueError(
            f"template shape {tmpl.shape} does not match data shape {image.shape}."
        )
    if alpha is None:
        amplitude = fit_pattern_amplitude(
            image,
            dq,
            tmpl,
            mask=mask,
            nsigma=nsigma,
            dilate=dilate,
            dq_bad_bits=dq_bad_bits,
        )
    else:
        amplitude = float(alpha)
    return image - amplitude * np.nan_to_num(tmpl, nan=0.0), err, dq


def subtract_grism_fixed_pattern(
    data: _Floats,
    err: _Floats,
    dq: _Ints,
    template: _Floats,
    *,
    mask: _Bool | None = None,
    alpha: float | None = None,
    dq_bad_bits: int = 1,
) -> tuple[_Floats, _Floats, _Ints]:
    """Subtract a unified GRISM fixed-pattern template from one frame.

    The amplitude is fit with a nuisance offset over the supplied trace-free mask
    (or fixed by ``alpha``), then only the scaled detector template is removed.
    ``err`` and ``dq`` pass through unchanged because the operation subtracts a
    deterministic additive model.
    """
    image = np.asarray(data, dtype=float)
    tmpl = np.asarray(template, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    if image.shape != tmpl.shape:
        raise ValueError(
            f"template shape {tmpl.shape} does not match data shape {image.shape}."
        )
    if alpha is None:
        amplitude, _ = fit_pattern_amplitude_offset(
            image, dq, tmpl, mask=mask, dq_bad_bits=dq_bad_bits
        )
    else:
        amplitude = float(alpha)
    return image - amplitude * np.nan_to_num(tmpl, nan=0.0), err, dq


def _explicit_exclusion(
    image: _Floats,
    dq: _Ints,
    *,
    mask: _Bool | None,
    dq_bad_bits: int,
) -> _Bool:
    """Return explicit non-finite / DQ / caller mask exclusions."""
    if np.asarray(dq).shape != image.shape:
        raise ValueError(f"dq shape {np.asarray(dq).shape} does not match data shape.")
    excluded = ~np.isfinite(image)
    if dq_bad_bits:
        excluded |= (np.asarray(dq) & dq_bad_bits) != 0
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != image.shape:
            raise ValueError(
                f"mask shape {mask_array.shape} does not match data shape {image.shape}."
            )
        excluded |= mask_array
    return excluded
