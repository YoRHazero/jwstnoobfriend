"""Per-module empirical sky-residual template for NIRCam WFSS frames.

The jwst WFSS ``BackgroundStep`` scales one fixed ``WFSSBKG`` template by a single
factor, which leaves a fixed, detector-locked residual ("bowl") -- shown to be
field-independent (a per-module template correlates r>0.95 across GOODS-S and
GOODS-N). Rather than fit a noisy per-frame 2-D background, this builds **one
template per (module, dispersion direction)** by stacking the trace-masked
residuals of all that group's frames, then subtracts it per frame with a scalar
fit.

Three pure-array helpers, jwst/navigation-free, used by the grism runner:

- :func:`sky_residual_grid` -- per frame (pass 1): downsample the trace-masked
  residual to a small grid (storable on disk; downsampling is lossless for the
  smooth template -- r=0.998 vs full-res at a 512 grid -- and denoises it).
- :func:`combine_sky_template` -- after pass 1: cross-frame median of the grids
  into a full-resolution template.
- :func:`fit_template_scalar` / :func:`subtract_sky_template` -- per frame
  (pass 2): scalar-fit and subtract the template.
"""

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce
from skimage.restoration import inpaint_biharmonic
from skimage.transform import resize

_Floats = NDArray[np.floating]
_Ints = NDArray[np.integer]
_Bool = NDArray[np.bool_]


def sky_residual_grid(data: _Floats, mask: _Bool, *, factor: int = 4) -> _Floats:
    """Return the trace-masked residual downsampled by ``factor`` (block medians).

    The masked (``True``) and non-finite pixels are excluded, and each
    ``factor`` x ``factor`` block is reduced by ``nanmedian`` -- a first,
    per-frame source rejection. The result is small (a 512 x 512 grid for the
    default ``factor=4`` on a 2048 frame, ~1 MB) so it can be streamed to disk
    during pass 1 without holding the full-resolution stack in memory.

    Parameters
    ----------
    data : numpy.ndarray
        2-D post-flat, master-sky-subtracted frame.
    mask : numpy.ndarray of bool
        Trace / source / bad-pixel exclusion mask (``True`` = exclude), e.g. from
        :func:`~noobfriend.reduction.grism_trace_mask`.
    factor : int, default 4
        Block side length for downsampling. The frame size should be divisible by
        it (and ideally a power of two, to keep amplifier boundaries grid-aligned).

    Returns
    -------
    numpy.ndarray
        The downsampled residual grid; fully-masked blocks are ``NaN``.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D or ``factor`` is not positive.
    """
    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"data must be 2-D; got shape {image.shape}.")
    if factor < 1:
        raise ValueError(f"factor must be >= 1; got {factor}.")
    masked = np.where(np.asarray(mask, dtype=bool) | ~np.isfinite(image), np.nan, image)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN blocks
        return block_reduce(masked, (factor, factor), func=np.nanmedian)


def combine_sky_template(
    grids: Sequence[_Floats],
    shape: tuple[int, int],
    *,
    smooth_sigma: float = 24.0,
) -> _Floats:
    """Combine per-frame residual grids into a full-resolution sky template.

    The grids (from :func:`sky_residual_grid`, all the same shape) are
    median-combined across frames -- the cross-frame median rejecting dithered
    sources and outliers the per-frame masking missed -- any still-empty cells
    are biharmonic-inpainted, and the result is bilinearly upsampled to ``shape``
    and Gaussian-smoothed.

    Parameters
    ----------
    grids : sequence of numpy.ndarray
        Per-frame residual grids of one (module, direction) group.
    shape : tuple of int
        Full-frame ``(ny, nx)`` to upsample the template to.
    smooth_sigma : float, default 24.0
        Gaussian smoothing sigma (full-resolution pixels) applied to the upsampled
        template; ``0`` disables smoothing.

    Returns
    -------
    numpy.ndarray
        The full-resolution sky-residual template.

    Raises
    ------
    ValueError
        If ``grids`` is empty.
    """
    if len(grids) == 0:
        raise ValueError("grids is empty; need at least one residual grid.")
    stack = np.stack([np.asarray(g, dtype=float) for g in grids])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN cells
        combined = np.nanmedian(stack, axis=0)
    bad = ~np.isfinite(combined)
    if bad.all():
        combined = np.zeros_like(combined)
    elif bad.any():
        combined = inpaint_biharmonic(np.nan_to_num(combined, nan=0.0), bad)
    full = resize(
        combined, shape, order=1, mode="edge", anti_aliasing=False, preserve_range=True
    )
    return gaussian_filter(full, smooth_sigma) if smooth_sigma > 0 else full


def fit_template_scalar(
    data: _Floats, template: _Floats, *, mask: _Bool | None = None
) -> float:
    """Return the least-squares scale ``s`` minimising ``data - s * template``.

    Fit over the finite background pixels (``mask`` excluded); the master-sky
    residual amplitude varies frame to frame with sky brightness, so a per-frame
    scalar is preferable to subtracting the template as-is.

    Parameters
    ----------
    data : numpy.ndarray
        2-D frame the template is fit to.
    template : numpy.ndarray
        The sky-residual template, same shape as ``data``.
    mask : numpy.ndarray of bool, optional
        Pixels to exclude from the fit (``True`` = exclude), e.g. the trace mask.

    Returns
    -------
    float
        The scale factor, or ``1.0`` when the template carries no power.
    """
    good = np.isfinite(data) & np.isfinite(template)
    if mask is not None:
        good &= ~np.asarray(mask, dtype=bool)
    t = np.asarray(template, dtype=float)[good]
    denom = float(np.sum(t * t))
    if denom <= 0:
        return 1.0
    return float(np.sum(np.asarray(data, dtype=float)[good] * t) / denom)


def subtract_sky_template(
    data: _Floats,
    err: _Floats,
    dq: _Ints,
    template: _Floats,
    *,
    mask: _Bool | None = None,
    scalar: bool | float = True,
) -> tuple[_Floats, _Floats, _Ints]:
    """Subtract the (scalar-fit) sky-residual template from a frame.

    Parameters
    ----------
    data : numpy.ndarray
        2-D frame to correct. Not modified in place; a corrected copy is returned.
    err, dq : numpy.ndarray
        Matching error and data-quality arrays, returned unchanged; the smooth
        template's own uncertainty is negligible and no pixels are reflagged.
    template : numpy.ndarray
        The (module, direction) sky-residual template from
        :func:`combine_sky_template`.
    mask : numpy.ndarray of bool, optional
        Pixels excluded from the scalar fit (``True`` = exclude).
    scalar : bool or float, default True
        ``True`` fits a per-frame scale (:func:`fit_template_scalar`); a float
        uses that scale directly; ``False`` subtracts the template as-is.

    Returns
    -------
    data : numpy.ndarray
        The template-subtracted frame (a new array).
    err, dq : numpy.ndarray
        The inputs ``err`` and ``dq``, unchanged.
    """
    image = np.asarray(data, dtype=float)
    if scalar is True:
        factor = fit_template_scalar(image, template, mask=mask)
    elif scalar is False:
        factor = 1.0
    else:
        factor = float(scalar)
    return image - factor * np.asarray(template, dtype=float), err, dq
