"""Invert the grism dispersion: a dispersed pixel -> candidate source positions.

The forward problem (a source at direct-image position ``(x0, y0)`` disperses to
a trace of pixels, one per wavelength) is single-valued. The inverse problem
this module solves -- *given a single dispersed pixel, which source positions
could have produced it?* -- is not: source position ``(x0, y0)`` and wavelength
together (3 unknowns) map to a 2-D dispersed pixel, leaving a 1-parameter
family. So a line detected at one grism pixel is consistent not with a point but
with a **locus**: the curve of ``(x0, y0)`` whose order-m trace passes through
that pixel, one point per wavelength in the order's valid domain.

The locus is found without any catalog, grism config file, or ``jwst`` install
-- the dispersion polynomials are already baked into the GWCS. For each
wavelength we invert the forward map by a fixed-point iteration (the dispersion
depends only weakly on source position, so it contracts in ~3 steps). Because
the solution is smooth and nearly straight within the valid domain, only a
handful of wavelengths are solved exactly (the *anchors*); the dense locus is a
cubic spline through them, which is far cheaper than solving every sample and is
fully vectorised over the anchors.

Associating the locus with an actual source (e.g. intersecting it with a
direct-image segmentation map) is deliberately out of scope; :meth:`SourceLocus.pixels`
hands back the integer pixels the locus crosses for exactly that downstream step.

This module is internal; :class:`SourceLocus`, :func:`source_locus`, and
:func:`source_loci` are re-exported from ``noobfriend.extraction.grism``.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from noobfriend.core.wcs import TransformWCS
from scipy.interpolate import CubicSpline

from noobfriend.extraction._wcs import (
    grism_trace_transform,
    world_detector_transforms,
)
from noobfriend.extraction.grism._wavelength import read_wavelength_domain

#: Forward trace map ``f(x0, y0, wavelength) -> (x_grism, y_grism)``, as returned
#: by :func:`~noobfriend.extraction._wcs.grism_trace_transform`.
Trace = Callable[[Any, Any, Any], tuple[Any, Any]]

#: Trace-parameter window bounding the default locus. The fit domain is
#: ``t in [0, 1]``, but near the band edges ``dlambda/dt -> 0`` and the model's
#: wavelength inverse turns wiggly, so a cubic spline cannot follow the locus
#: there (and more anchors do not help). Empirically the high edge degrades
#: past ``t ~ 0.93``; trimming to this interior keeps the spline sub-pixel
#: everywhere, and excludes wavelengths beyond the filter's useful throughput
#: anyway. User-supplied ranges are validated against this window.
_T_LO: float = 0.05
_T_HI: float = 0.90


@dataclass(frozen=True)
class SourceLocus:
    """Source positions whose trace passes through one dispersed pixel.

    The degeneracy curve for a feature (e.g. an emission line) detected at a
    single grism pixel: each point is a candidate direct-image source position,
    valid if the line is at the corresponding :attr:`wavelength`. Coordinates
    are detector pixels of the grism exposure's own frame (where ``detector``
    is the undispersed/direct astrometry), so they index that exposure's
    direct-image grid directly; map them through ``detector -> world`` for any
    other frame.

    Attributes
    ----------
    x_line, y_line : float
        The dispersed pixel this locus was computed for (grism detector frame).
    order : int
        Spectral order assumed for the inversion.
    x0, y0 : numpy.ndarray
        Candidate source positions, shape ``(N,)``, in grism-exposure detector
        pixels, ordered by ascending :attr:`wavelength`.
    wavelength : numpy.ndarray
        Wavelength (microns), shape ``(N,)``, implied at each candidate -- i.e.
        the line wavelength for which that source would disperse onto
        ``(x_line, y_line)``.
    wavelength_range : tuple[float, float]
        The order's valid wavelength domain ``(lo, hi)`` in microns that bounds
        the locus (before the small endpoint inset).
    """

    x_line: float
    y_line: float
    order: int
    x0: np.ndarray
    y0: np.ndarray
    wavelength: np.ndarray
    wavelength_range: tuple[float, float]

    def pixels(
        self, *, bounds: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the unique integer pixels the locus passes through.

        Rounds the sampled locus to integer detector pixels and removes
        duplicates, giving the candidate-source pixel set ready to intersect
        with a segmentation map. The result is a *set* (lexicographically
        sorted), not the ordered path.

        Parameters
        ----------
        bounds : tuple[int, int] or None, optional
            Array shape ``(height, width)`` to clip to; pixels outside it are
            dropped. ``None`` (default) returns all pixels, including any off
            the detector.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(y, x)`` integer pixel index arrays, suitable for
            ``segmap[y, x]``.
        """
        iy = np.rint(self.y0).astype(np.intp)
        ix = np.rint(self.x0).astype(np.intp)
        coords = np.unique(np.stack((iy, ix), axis=1), axis=0)
        iy, ix = coords[:, 0], coords[:, 1]
        if bounds is not None:
            height, width = bounds
            keep = (iy >= 0) & (iy < height) & (ix >= 0) & (ix < width)
            iy, ix = iy[keep], ix[keep]
        return iy, ix


def _disperse(
    trace: Trace, x0: np.ndarray, y0: np.ndarray, lam: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the forward trace element-wise over equal-length inputs.

    Calls ``trace`` once on the three 1-D arrays and, when the underlying model
    broadcasts them into a 2-D outer-product grid, recovers the element-wise
    result from its diagonal.

    Parameters
    ----------
    trace : Trace
        Forward trace map ``f(x0, y0, wavelength) -> (x_grism, y_grism)``.
    x0, y0, lam : numpy.ndarray
        Equal-length 1-D arrays of source x, source y, and wavelength (microns).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(x_grism, y_grism)``, each 1-D of the input length.
    """
    xg, yg = trace(x0, y0, lam)
    xg = np.asarray(xg, dtype=float)
    yg = np.asarray(yg, dtype=float)
    if xg.ndim == 2:
        xg = np.diagonal(xg)
        yg = np.diagonal(yg)
    return xg, yg


def _solve_anchors(
    trace: Trace,
    x_line: float,
    y_line: float,
    lam: np.ndarray,
    n_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the forward map at each wavelength by fixed-point iteration.

    For each ``lambda`` finds the source ``(x0, y0)`` whose trace passes through
    ``(x_line, y_line)``, via ``x0 <- x_line - dx``, ``y0 <- y_line - dy`` where
    ``(dx, dy)`` is the current trace displacement. The dispersion's dependence
    on source position is weak, so the map contracts quickly (a few iterations
    reach sub-millipixel).

    Parameters
    ----------
    trace : Trace
        Forward trace map ``f(x0, y0, wavelength) -> (x_grism, y_grism)``.
    x_line, y_line : float
        The dispersed pixel to invert.
    lam : numpy.ndarray
        Wavelengths (microns) at which to solve, shape ``(N,)``.
    n_iter : int
        Number of fixed-point iterations.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(x0, y0)`` source positions, each shape ``(N,)``.
    """
    x0 = np.full(lam.shape, float(x_line))
    y0 = np.full(lam.shape, float(y_line))
    for _ in range(n_iter):
        xg, yg = _disperse(trace, x0, y0, lam)
        x0 = x0 + (x_line - xg)
        y0 = y0 + (y_line - yg)
    return x0, y0


def _build_locus(
    trace: Trace,
    x_line: float,
    y_line: float,
    order: int,
    wavelength_range: tuple[float, float],
    *,
    n_anchors: int,
    n_iter: int,
    step: float,
) -> SourceLocus:
    """Build a :class:`SourceLocus` from a bound trace map and a wavelength range.

    The WCS-free core: solves ``n_anchors`` exact anchors across the wavelength
    range, fits a cubic spline through them, and samples it densely enough that
    consecutive points are at most ``step`` pixels apart. The caller is
    responsible for passing a range over which the locus is well-behaved (see
    :func:`source_locus`); no trimming happens here.

    Parameters
    ----------
    trace : Trace
        Forward trace map already bound to the chosen order.
    x_line, y_line : float
        The dispersed pixel to invert.
    order : int
        Spectral order (recorded on the result).
    wavelength_range : tuple[float, float]
        Wavelength interval ``(lo, hi)`` in microns to span.
    n_anchors : int
        Number of exactly-solved anchor wavelengths (``>= 4`` for the cubic).
    n_iter : int
        Fixed-point iterations per anchor.
    step : float
        Maximum spacing (pixels) between consecutive sampled locus points.

    Returns
    -------
    SourceLocus
        The sampled degeneracy locus.

    Raises
    ------
    ValueError
        If ``n_anchors < 4``, ``n_iter < 1``, or ``step <= 0``.
    """
    if n_anchors < 4:
        raise ValueError(
            f"n_anchors must be >= 4 for cubic interpolation, got {n_anchors}."
        )
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}.")
    if step <= 0.0:
        raise ValueError(f"step must be > 0, got {step}.")

    lo, hi = sorted(float(v) for v in wavelength_range)
    lam_anchors = np.linspace(lo, hi, n_anchors)

    x0a, y0a = _solve_anchors(trace, x_line, y_line, lam_anchors, n_iter)

    spline_x = CubicSpline(lam_anchors, x0a)
    spline_y = CubicSpline(lam_anchors, y0a)

    arc_length = float(np.hypot(np.diff(x0a), np.diff(y0a)).sum())
    n_samples = max(n_anchors, int(np.ceil(arc_length / step)) + 1)
    lam_dense = np.linspace(lam_anchors[0], lam_anchors[-1], n_samples)

    return SourceLocus(
        x_line=float(x_line),
        y_line=float(y_line),
        order=int(order),
        x0=spline_x(lam_dense),
        y0=spline_y(lam_dense),
        wavelength=lam_dense,
        wavelength_range=(lo, hi),
    )


def _resolve_range(
    wavelength_range: tuple[float, float] | None,
    safe_range: tuple[float, float],
) -> tuple[float, float]:
    """Resolve the locus wavelength range against the trustworthy ``t``-window.

    Parameters
    ----------
    wavelength_range : tuple[float, float] or None
        Caller-requested range in microns, or ``None`` to use the window.
    safe_range : tuple[float, float]
        The ``(lo, hi)`` wavelengths bounding the well-behaved trace interior,
        from :func:`~noobfriend.extraction.grism.read_wavelength_domain` with
        ``t_range = (_T_LO, _T_HI)``.

    Returns
    -------
    tuple[float, float]
        The range to span, sorted ascending: ``safe_range`` when no request, or
        the validated request.

    Raises
    ------
    ValueError
        If ``wavelength_range`` extends beyond ``safe_range`` (where the spline
        is unreliable).
    """
    lo_s, hi_s = safe_range
    if wavelength_range is None:
        return (lo_s, hi_s)
    lo, hi = sorted(float(v) for v in wavelength_range)
    tol = 1e-9 * (hi_s - lo_s)
    if lo < lo_s - tol or hi > hi_s + tol:
        raise ValueError(
            f"wavelength_range ({lo:.4f}, {hi:.4f}) um extends beyond the "
            f"trustworthy window ({lo_s:.4f}, {hi_s:.4f}) um (trace parameter "
            f"t in [{_T_LO}, {_T_HI}]), where the locus spline is unreliable. "
            f"Clamp the range to the window, or omit it to use the window."
        )
    return (lo, hi)


def source_locus(
    wcs: TransformWCS,
    x_line: float,
    y_line: float,
    *,
    order: int = 1,
    wavelength_range: tuple[float, float] | None = None,
    n_anchors: int = 10,
    n_iter: int = 3,
    step: float = 1.0,
) -> SourceLocus:
    """Find the source-position locus consistent with a dispersed pixel.

    Inverts the grism dispersion: returns the curve of direct-image source
    positions whose order-``order`` trace passes through ``(x_line, y_line)``,
    one point per wavelength. Everything is read from the GWCS -- no catalog,
    grism config, or ``jwst`` install required.

    The locus is bounded by a trustworthy interior of the trace -- trace
    parameter ``t in [0.05, 0.90]`` -- rather than the full fit domain
    ``t in [0, 1]``: near the band edges the wavelength inverse turns unreliable
    and the spline cannot follow the locus (those wavelengths are also beyond
    the filter's useful throughput).

    Parameters
    ----------
    wcs : TransformWCS
        A JWST WFSS grism WCS (one exposing a ``grism_detector`` frame).
    x_line, y_line : float
        The dispersed pixel where the feature (e.g. emission line) sits, in the
        grism exposure's detector pixels.
    order : int, optional
        Spectral order to invert, by default ``1``.
    wavelength_range : tuple[float, float] or None, optional
        Wavelength interval ``(lo, hi)`` in microns to span. ``None`` (default)
        uses the trustworthy window read from the dispersion model at this
        pixel. A supplied range is **validated** to lie within that window and
        raises ``ValueError`` otherwise (e.g. passing a filter bandpass whose
        edge exceeds the window).
    n_anchors : int, optional
        Number of wavelengths solved exactly before spline interpolation, by
        default ``10`` (sub-pixel everywhere across the detector). Must be
        ``>= 4``.
    n_iter : int, optional
        Fixed-point iterations per anchor, by default ``3`` (reaches
        ~micropixel; 2 already reaches ~millipixel).
    step : float, optional
        Maximum spacing (pixels) between consecutive sampled locus points, by
        default ``1.0`` (a gap-free pixel trail for :meth:`SourceLocus.pixels`).

    Returns
    -------
    SourceLocus
        The sampled degeneracy locus.

    Raises
    ------
    ValueError
        If ``wavelength_range`` extends beyond the trustworthy window.

    See Also
    --------
    source_loci : Same inversion for many pixels of one WCS at once.

    Notes
    -----
    The window depends only weakly on the source's direct-image position, so it
    is read at the dispersed pixel itself (a close proxy).
    """
    _, detector_to_world = world_detector_transforms(wcs)
    ra, dec = detector_to_world(x_line, y_line)
    safe_range = read_wavelength_domain(
        wcs, float(ra), float(dec), order=order, t_range=(_T_LO, _T_HI)
    )
    wr = _resolve_range(wavelength_range, safe_range)

    trace = grism_trace_transform(wcs, order=order)
    return _build_locus(
        trace,
        float(x_line),
        float(y_line),
        order,
        wr,
        n_anchors=n_anchors,
        n_iter=n_iter,
        step=step,
    )


def source_loci(
    wcs: TransformWCS,
    x_line: Sequence[float] | np.ndarray,
    y_line: Sequence[float] | np.ndarray,
    *,
    order: int = 1,
    wavelength_range: tuple[float, float] | None = None,
    n_anchors: int = 10,
    n_iter: int = 3,
    step: float = 1.0,
) -> list[SourceLocus]:
    """Find source-position loci for many dispersed pixels of one WCS.

    Batched :func:`source_locus`: the dispersion model is read once and a single
    trustworthy window is resolved once for the whole set (evaluated at the mean
    pixel; the window's source-position dependence is weak). Intended for
    sweeping all detections in one grism frame.

    Parameters
    ----------
    wcs : TransformWCS
        A JWST WFSS grism WCS (one exposing a ``grism_detector`` frame).
    x_line, y_line : Sequence[float] or numpy.ndarray
        Dispersed pixel coordinates, same length, in grism detector pixels.
    order : int, optional
        Spectral order to invert, by default ``1``.
    wavelength_range : tuple[float, float] or None, optional
        Shared wavelength interval in microns. ``None`` (default) uses the
        trustworthy window. A supplied range is validated against it (at the
        mean pixel). See :func:`source_locus`.
    n_anchors, n_iter, step : optional
        Per-locus parameters, forwarded to :func:`source_locus`.

    Returns
    -------
    list[SourceLocus]
        One locus per input pixel, in input order.

    Raises
    ------
    ValueError
        If ``x_line`` and ``y_line`` differ in shape, or ``wavelength_range``
        extends beyond the trustworthy window.
    """
    xs = np.atleast_1d(np.asarray(x_line, dtype=float))
    ys = np.atleast_1d(np.asarray(y_line, dtype=float))
    if xs.shape != ys.shape:
        raise ValueError(
            f"x_line and y_line must have the same shape, got {xs.shape} and "
            f"{ys.shape}."
        )

    _, detector_to_world = world_detector_transforms(wcs)
    ra, dec = detector_to_world(float(xs.mean()), float(ys.mean()))
    safe_range = read_wavelength_domain(
        wcs, float(ra), float(dec), order=order, t_range=(_T_LO, _T_HI)
    )
    wr = _resolve_range(wavelength_range, safe_range)

    trace = grism_trace_transform(wcs, order=order)
    return [
        _build_locus(
            trace,
            float(xi),
            float(yi),
            order,
            wr,
            n_anchors=n_anchors,
            n_iter=n_iter,
            step=step,
        )
        for xi, yi in zip(xs, ys)
    ]
