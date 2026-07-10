"""Tests for the DIY scalar sky matcher (synthetic frames only, no real FITS)."""

import numpy as np

from noobfriend.core.wcs import from_fits_wcs
from noobfriend.reduction.mosaic import SkyMatcher, field_grid, frame_sky
from noobfriend.reduction.mosaic._sky import _solve_offsets

from ._helpers import corners as _corners
from ._helpers import tan_wcs

RA0, DEC0 = 150.0, 2.0
SCALE = 1.0  # arcsec/pix (coarse; sky needs no resolution)


def _tan(ra0, dec0, shape):
    return tan_wcs(ra0, dec0, shape, SCALE)


def test_frame_sky_recovers_level_ignoring_sources():
    rng = np.random.default_rng(0)
    data = 5.0 + rng.normal(0, 0.1, (128, 128))
    data[40:45, 40:45] += 500.0  # a bright source
    dq = np.zeros((128, 128), dtype=np.int32)
    dq[0, 0] = 1  # a DO_NOT_USE pixel
    level, mad = frame_sky(data, dq)
    assert abs(level - 5.0) < 0.05
    assert 0.05 < mad < 0.2


def test_frame_sky_all_bad_is_nan():
    data = np.ones((8, 8))
    dq = np.ones((8, 8), dtype=np.int32)  # every pixel DO_NOT_USE
    level, mad = frame_sky(data, dq)
    assert np.isnan(level) and np.isnan(mad)


def test_solve_offsets_recovers_pairwise_differences():
    # three frames with true offsets; pairwise diffs = offset_i - offset_j
    true = np.array([0.0, 0.3, 0.7])
    pairs = [
        (0, 1, true[0] - true[1]),
        (1, 2, true[1] - true[2]),
        (0, 2, true[0] - true[2]),
    ]
    got = _solve_offsets(pairs, 3)
    # match-down gauge: lowest is 0, spacing preserved
    assert np.allclose(got - got.min(), true - true.min(), atol=1e-9)


def test_solve_offsets_no_pairs_is_zero():
    assert np.array_equal(_solve_offsets([], 4), np.zeros(4))


def test_skymatcher_recovers_injected_offsets():
    shape = (80, 80)
    # three overlapping frames, each a constant sky plus a known offset
    centres = [
        (RA0, DEC0),
        (RA0 + 20 / 3600.0 / np.cos(np.radians(DEC0)), DEC0),
        (RA0, DEC0 + 20 / 3600.0),
    ]
    injected = [0.0, 0.4, 0.9]
    wcss = [_tan(ra, dec, shape) for ra, dec in centres]
    corners = [_corners(w, shape) for w in wcss]
    field = field_grid(corners, SCALE, rotation=0.0)

    matcher = SkyMatcher(field)
    for k, (w, off) in enumerate(zip(wcss, injected)):
        data = np.full(shape, 3.0 + off)
        dq = np.zeros(shape, dtype=np.int32)
        matcher.add(f"f{k}", data, dq, from_fits_wcs(w))

    offsets = matcher.match()
    got = np.array([offsets[f"f{k}"] for k in range(3)])
    truth = np.array(injected)
    assert np.allclose(got - got.min(), truth - truth.min(), atol=0.02)


def test_skymatcher_uniform_field_is_a_no_op():
    shape = (80, 80)
    centres = [(RA0, DEC0), (RA0 + 20 / 3600.0 / np.cos(np.radians(DEC0)), DEC0)]
    wcss = [_tan(ra, dec, shape) for ra, dec in centres]
    corners = [_corners(w, shape) for w in wcss]
    field = field_grid(corners, SCALE, rotation=0.0)

    matcher = SkyMatcher(field)
    for k, w in enumerate(wcss):
        matcher.add(
            f"f{k}", np.full(shape, 7.0), np.zeros(shape, np.int32), from_fits_wcs(w)
        )

    offsets = matcher.match()
    assert all(abs(v) < 1e-6 for v in offsets.values())


def test_skymatcher_disjoint_frames_get_zero():
    shape = (40, 40)
    # two frames a degree apart -> no overlap -> no match equations -> zero
    wcss = [_tan(RA0, DEC0, shape), _tan(RA0 + 1.0, DEC0 + 1.0, shape)]
    corners = [_corners(w, shape) for w in wcss]
    field = field_grid(corners, SCALE, rotation=0.0)

    matcher = SkyMatcher(field)
    for k, w in enumerate(wcss):
        matcher.add(
            f"f{k}",
            np.full(shape, 2.0 + k),
            np.zeros(shape, np.int32),
            from_fits_wcs(w),
        )

    assert matcher.match() == {"f0": 0.0, "f1": 0.0}
