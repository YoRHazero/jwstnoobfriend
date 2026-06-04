"""Tests for the grism source-position locus (dispersion inversion) core.

These cover the WCS-free maths of :mod:`noobfriend.extraction.grism._locus`:
the fixed-point anchor solve, the diagonal forward-evaluation helper, the
spline build, and :meth:`SourceLocus.pixels`. They inject a synthetic trace
map with a closed-form linear dispersion (so the true locus is exactly linear
in wavelength and the cubic spline reproduces it), and never touch ``gwcs`` or
``example_data``. The WCS-driven wrappers ``source_locus`` / ``source_loci``
are thin adapters around this core and are exercised only end-to-end elsewhere.
"""

from collections.abc import Callable

import numpy as np
import pytest

from noobfriend.extraction.grism import SourceLocus
from noobfriend.extraction.grism._locus import (
    _build_locus,
    _disperse,
    _resolve_range,
    _solve_anchors,
)

Trace = Callable[..., tuple[np.ndarray, np.ndarray]]


def _linear_trace(
    *,
    ax: float = 220.0,
    ay: float = -3.0,
    lam_ref: float = 4.0,
    eps_x: float = 0.03,
    eps_y: float = 0.02,
) -> Trace:
    """Make a GRISMR-like trace: linear in wavelength with weak field dependence.

    ``xg = x0 (1 + eps_x) + ax (lam - lam_ref)`` and similarly for ``yg``. The
    weak ``x0``/``y0`` terms make the fixed point need more than one iteration
    (contraction factor ``eps``), while linearity keeps the inverse locus
    exactly linear in wavelength.
    """

    def trace(
        x0: np.ndarray, y0: np.ndarray, lam: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        x0 = np.asarray(x0, dtype=float)
        y0 = np.asarray(y0, dtype=float)
        lam = np.asarray(lam, dtype=float)
        xg = x0 * (1.0 + eps_x) + ax * (lam - lam_ref)
        yg = y0 * (1.0 + eps_y) + ay * (lam - lam_ref)
        return xg, yg

    return trace


def _round_trip_residual(loc: SourceLocus, trace: Trace) -> float:
    """Max pixel error when the locus is dispersed back to the seed pixel."""
    xg, yg = trace(loc.x0, loc.y0, loc.wavelength)
    return float(np.max(np.hypot(xg - loc.x_line, yg - loc.y_line)))


# --------------------------------------------------------------------------- #
# _disperse: element-wise eval, incl. the 2-D outer-product diagonal case
# --------------------------------------------------------------------------- #


def test_disperse_passes_through_1d_output():
    trace = _linear_trace()
    x0 = np.array([100.0, 200.0, 300.0])
    y0 = np.array([10.0, 20.0, 30.0])
    lam = np.array([3.9, 4.0, 4.1])
    xg, yg = _disperse(trace, x0, y0, lam)
    exp_x, exp_y = trace(x0, y0, lam)
    np.testing.assert_allclose(xg, exp_x)
    np.testing.assert_allclose(yg, exp_y)


def test_disperse_recovers_diagonal_from_2d_grid():
    # Mimic the NIRCam model broadcasting equal-length inputs outer-product
    # style: element-wise answer lives on the diagonal, off-diagonal is junk.
    def trace_2d(x0, y0, lam):
        x0 = np.asarray(x0, float)
        y0 = np.asarray(y0, float)
        lam = np.asarray(lam, float)
        ex = x0 + lam
        ey = y0 - lam
        n = ex.size
        gx = np.full((n, n), 999.0)
        gy = np.full((n, n), -999.0)
        np.fill_diagonal(gx, ex)
        np.fill_diagonal(gy, ey)
        return gx, gy

    x0 = np.array([1.0, 2.0, 3.0])
    y0 = np.array([4.0, 5.0, 6.0])
    lam = np.array([0.1, 0.2, 0.3])
    xg, yg = _disperse(trace_2d, x0, y0, lam)
    np.testing.assert_allclose(xg, x0 + lam)
    np.testing.assert_allclose(yg, y0 - lam)


# --------------------------------------------------------------------------- #
# _solve_anchors: fixed-point inversion
# --------------------------------------------------------------------------- #


def test_solve_anchors_round_trip():
    trace = _linear_trace()
    lam = np.linspace(3.9, 4.9, 6)
    x0, y0 = _solve_anchors(trace, 700.0, 1300.0, lam, n_iter=10)
    xg, yg = trace(x0, y0, lam)
    np.testing.assert_allclose(xg, 700.0, atol=1e-8)
    np.testing.assert_allclose(yg, 1300.0, atol=1e-8)


def test_solve_anchors_converges_with_more_iterations():
    trace = _linear_trace(eps_x=0.1, eps_y=0.1)  # slower contraction
    lam = np.linspace(3.9, 4.9, 6)

    def residual(n_iter: int) -> float:
        x0, y0 = _solve_anchors(trace, 700.0, 1300.0, lam, n_iter=n_iter)
        xg, yg = trace(x0, y0, lam)
        return float(np.max(np.hypot(xg - 700.0, yg - 1300.0)))

    res = [residual(n) for n in (1, 2, 3, 5)]
    assert res[0] > res[1] > res[2] > res[3]


# --------------------------------------------------------------------------- #
# _build_locus: anchors -> spline -> dense sampling
# --------------------------------------------------------------------------- #


def test_build_locus_dense_round_trip_is_exact_for_linear_trace():
    trace = _linear_trace()
    loc = _build_locus(
        trace, 700.0, 1300.0, 1, (3.85, 4.95), n_anchors=10, n_iter=10, step=1.0
    )
    # Linear dispersion -> linear locus -> cubic spline reproduces it exactly.
    assert _round_trip_residual(loc, trace) < 1e-6


def test_build_locus_is_ascending_in_wavelength_and_consistent_shapes():
    trace = _linear_trace()
    loc = _build_locus(
        trace, 512.0, 800.0, 1, (3.85, 4.95), n_anchors=10, n_iter=5, step=1.0
    )
    assert loc.x0.shape == loc.y0.shape == loc.wavelength.shape
    assert np.all(np.diff(loc.wavelength) > 0)
    assert loc.wavelength_range == (3.85, 4.95)
    assert loc.order == 1


def test_build_locus_sampling_respects_step():
    trace = _linear_trace()
    step = 2.0
    loc = _build_locus(
        trace, 700.0, 1300.0, 1, (3.85, 4.95), n_anchors=10, n_iter=5, step=step
    )
    gaps = np.hypot(np.diff(loc.x0), np.diff(loc.y0))
    assert gaps.max() <= step * (1.0 + 1e-6)


def test_build_locus_sorts_reversed_wavelength_range():
    trace = _linear_trace()
    loc = _build_locus(
        trace, 700.0, 1300.0, 1, (4.95, 3.85), n_anchors=10, n_iter=5, step=1.0
    )
    assert loc.wavelength_range == (3.85, 4.95)
    assert np.all(np.diff(loc.wavelength) > 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_anchors": 3, "n_iter": 3, "step": 1.0},
        {"n_anchors": 10, "n_iter": 0, "step": 1.0},
        {"n_anchors": 10, "n_iter": 3, "step": 0.0},
        {"n_anchors": 10, "n_iter": 3, "step": -1.0},
    ],
)
def test_build_locus_validates_parameters(kwargs):
    trace = _linear_trace()
    with pytest.raises(ValueError):
        _build_locus(trace, 700.0, 1300.0, 1, (3.85, 4.95), **kwargs)


# --------------------------------------------------------------------------- #
# _resolve_range: default to the safe window, validate supplied ranges
# --------------------------------------------------------------------------- #


def test_resolve_range_none_returns_safe_window():
    assert _resolve_range(None, (3.88, 5.01)) == (3.88, 5.01)


def test_resolve_range_within_window_is_sorted():
    assert _resolve_range((4.5, 4.0), (3.88, 5.01)) == (4.0, 4.5)


def test_resolve_range_at_boundary_is_accepted():
    assert _resolve_range((3.88, 5.01), (3.88, 5.01)) == (3.88, 5.01)


@pytest.mark.parametrize("bad", [(3.5, 4.5), (4.0, 5.2), (3.5, 5.2)])
def test_resolve_range_outside_window_raises(bad):
    with pytest.raises(ValueError):
        _resolve_range(bad, (3.88, 5.01))


# --------------------------------------------------------------------------- #
# SourceLocus.pixels: round, dedup, clip
# --------------------------------------------------------------------------- #


def _locus_with(x0: np.ndarray, y0: np.ndarray) -> SourceLocus:
    return SourceLocus(
        x_line=5.0,
        y_line=5.0,
        order=1,
        x0=np.asarray(x0, dtype=float),
        y0=np.asarray(y0, dtype=float),
        wavelength=np.linspace(4.0, 4.5, len(x0)),
        wavelength_range=(3.9, 5.0),
    )


def test_pixels_rounds_and_dedups():
    loc = _locus_with(
        x0=np.array([1.2, 1.4, 2.6, 2.6, -1.0]),
        y0=np.array([3.1, 3.0, 7.8, 7.8, 0.4]),
    )
    iy, ix = loc.pixels()
    got = set(zip(iy.tolist(), ix.tolist()))
    assert got == {(0, -1), (3, 1), (8, 3)}
    assert iy.dtype == np.intp and ix.dtype == np.intp


def test_pixels_clips_to_bounds():
    loc = _locus_with(
        x0=np.array([1.2, 1.4, 2.6, 2.6, -1.0]),
        y0=np.array([3.1, 3.0, 7.8, 7.8, 0.4]),
    )
    iy, ix = loc.pixels(bounds=(9, 9))
    got = set(zip(iy.tolist(), ix.tolist()))
    assert got == {(3, 1), (8, 3)}  # (0, -1) dropped: x < 0


def test_pixels_keeps_offdetector_without_bounds():
    loc = _locus_with(x0=np.array([-3.0, 2000.0]), y0=np.array([-2.0, 50.0]))
    iy, ix = loc.pixels()
    got = set(zip(iy.tolist(), ix.tolist()))
    assert got == {(-2, -3), (50, 2000)}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
