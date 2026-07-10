"""Tests for the field-median outlier engine (reduction.mosaic._outlier).

Synthetic frames on tiny TAN grids only -- no real FITS data. The flagging math
is checked against the stcal contract (thresholds, slope guard, 3x3 grow) and
the median/blot roundtrip against frames that agree on the sky except for an
injected cosmic ray.
"""

import numpy as np
import pytest
from astropy.wcs import WCS

from noobfriend.reduction.mosaic import (
    OUTLIER_DQ,
    FieldMedian,
    blot_to_frame,
    field_grid,
    flag_outliers,
)
from noobfriend.reduction.mosaic._outlier import DO_NOT_USE, _abs_deriv

_SCALE = 0.06  # arcsec/px for the synthetic frames
_RA0, _DEC0 = 150.0, 2.0


class _FrameWCS:
    """Minimal gwcs-protocol adapter over an astropy TAN WCS."""

    def __init__(self, wcs: WCS) -> None:
        self._wcs = wcs

    def get_transform(self, from_frame: str, to_frame: str):
        if (from_frame, to_frame) == ("world", "detector"):
            return self._wcs.world_to_pixel_values
        if (from_frame, to_frame) == ("detector", "world"):
            return self._wcs.pixel_to_world_values
        raise ValueError(f"Unsupported transform {from_frame!r} -> {to_frame!r}.")


def _frame_wcs(dx_px: float = 0.0, dy_px: float = 0.0) -> _FrameWCS:
    """Build a TAN WCS near the field centre, dithered by ``(dx_px, dy_px)``."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [_RA0, _DEC0]
    wcs.wcs.crpix = [16.5 + dx_px, 16.5 + dy_px]
    scale = _SCALE / 3600.0
    wcs.wcs.cd = np.array([[-scale, 0.0], [0.0, scale]])
    return _FrameWCS(wcs)


def _corners(frame: _FrameWCS, shape: tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    d2w = frame.get_transform("detector", "world")
    x = np.array([0.0, nx - 1.0, nx - 1.0, 0.0])
    y = np.array([0.0, 0.0, ny - 1.0, ny - 1.0])
    ra, dec = d2w(x, y)
    return np.column_stack([ra, dec])


def _sky_scene(frame: _FrameWCS, shape: tuple[int, int]) -> np.ndarray:
    """Evaluate a smooth scene in *world* coordinates (identical on the sky)."""
    ny, nx = shape
    x, y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    ra, dec = frame.get_transform("detector", "world")(x, y)
    return 10.0 + 5e4 * (np.asarray(ra) - _RA0) + 8e4 * (np.asarray(dec) - _DEC0)


# -- flag_outliers (stcal-contract replica) ------------------------------------


def test_flag_outliers_flags_a_spike_and_grows_one_ring() -> None:
    blot = np.full((16, 16), 10.0)
    err = np.ones_like(blot)
    data = blot.copy()
    data[8, 8] += 100.0  # far above snr1
    data[8, 9] += 4.5  # between snr2 and snr1: only flaggable via the grow

    mask = flag_outliers(data, err, blot)

    assert mask[8, 8]
    assert mask[8, 9]  # neighbour of a strong CR, above snr2 -> grown in
    assert mask.sum() == 2


def test_flag_outliers_isolated_marginal_pixel_stays() -> None:
    blot = np.full((16, 16), 10.0)
    err = np.ones_like(blot)
    data = blot.copy()
    data[4, 4] += 4.5  # above snr2 but below snr1, no strong neighbour

    assert not flag_outliers(data, err, blot).any()


def test_flag_outliers_slope_guard_protects_bright_edges() -> None:
    # A sharp source edge: data == blot exactly, huge local slope. A small
    # registration-like offset (below scale1 * slope) must not flag.
    blot = np.zeros((16, 16))
    blot[:, 8:] = 100.0
    err = np.full_like(blot, 0.1)
    data = blot * 1.0
    data[:, 8] += 20.0  # large vs err, small vs scale1 * |slope| = 120

    assert not flag_outliers(data, err, blot).any()


def test_flag_outliers_never_flags_without_median_coverage() -> None:
    blot = np.full((8, 8), np.nan)
    data = np.full((8, 8), 1000.0)
    err = np.ones_like(data)

    assert not flag_outliers(data, err, blot).any()


def test_abs_deriv_is_local_max_neighbour_difference() -> None:
    a = np.zeros((5, 5))
    a[2, 2] = 7.0
    d = _abs_deriv(a)
    assert d[2, 2] == 7.0
    assert d[1, 2] == d[3, 2] == d[2, 1] == d[2, 3] == 7.0
    assert d[0, 0] == 0.0


# -- FieldMedian + blot roundtrip ----------------------------------------------


@pytest.fixture
def dithered_frames() -> tuple[list[_FrameWCS], list[np.ndarray], tuple[int, int]]:
    shape = (32, 32)
    frames = [_frame_wcs(0.0, 0.0), _frame_wcs(2.0, 1.0), _frame_wcs(-1.0, 2.0)]
    scenes = [_sky_scene(f, shape) for f in frames]
    return frames, scenes, shape


def test_median_blot_flags_only_the_injected_cr(dithered_frames) -> None:
    frames, scenes, shape = dithered_frames
    data = [s.copy() for s in scenes]
    data[0][16, 16] += 500.0  # cosmic ray in frame 0 only
    dq = [np.zeros(shape, dtype=np.uint32) for _ in frames]
    err = np.full(shape, 0.5)

    grid = field_grid([_corners(f, shape) for f in frames], _SCALE, rotation=0.0)
    stack = FieldMedian(grid, [f"exp{i}" for i in range(3)])
    for i, frame in enumerate(frames):
        stack.add(f"exp{i}", data[i], dq[i], frame, coarse_step=None)
    median = stack.median()

    blot = blot_to_frame(median, grid, frames[0], shape, coarse_step=None)
    inner = np.s_[4:-4, 4:-4]
    assert np.allclose(blot[inner], scenes[0][inner], atol=0.2)  # CR is gone

    mask = flag_outliers(data[0], err, blot)
    assert mask[16, 16]
    assert mask[inner].sum() <= 9  # the CR and at most its grown ring

    dq[0] |= mask * np.uint32(OUTLIER_DQ)
    assert dq[0][16, 16] == OUTLIER_DQ


def test_clean_frames_produce_no_flags(dithered_frames) -> None:
    frames, scenes, shape = dithered_frames
    dq = np.zeros(shape, dtype=np.uint32)
    err = np.full(shape, 0.5)

    grid = field_grid([_corners(f, shape) for f in frames], _SCALE, rotation=0.0)
    stack = FieldMedian(grid, [f"exp{i}" for i in range(3)])
    for i, frame in enumerate(frames):
        stack.add(f"exp{i}", scenes[i], dq, frame, coarse_step=None)
    median = stack.median()

    for i, frame in enumerate(frames):
        blot = blot_to_frame(median, grid, frame, shape, coarse_step=None)
        inner = np.s_[4:-4, 4:-4]
        assert not flag_outliers(scenes[i], err, blot)[inner].any()


def test_do_not_use_pixels_are_excluded_from_the_median(dithered_frames) -> None:
    frames, scenes, shape = dithered_frames
    data = [s.copy() for s in scenes]
    dq = [np.zeros(shape, dtype=np.uint32) for _ in frames]
    # A hot pixel present in every frame at the same detector position would
    # normally poison the median -- masking it via DQ must keep the median clean.
    data[0][10, 10] += 900.0
    dq[0][10, 10] = DO_NOT_USE

    grid = field_grid([_corners(f, shape) for f in frames], _SCALE, rotation=0.0)
    stack = FieldMedian(grid, [f"exp{i}" for i in range(3)])
    for i, frame in enumerate(frames):
        stack.add(f"exp{i}", data[i], dq[i], frame, coarse_step=None)
    median = stack.median()

    blot = blot_to_frame(median, grid, frames[0], shape, coarse_step=None)
    assert abs(blot[10, 10] - scenes[0][10, 10]) < 0.5


def test_memmap_stack_matches_in_memory(dithered_frames, tmp_path) -> None:
    frames, scenes, shape = dithered_frames
    dq = np.zeros(shape, dtype=np.uint32)

    grid = field_grid([_corners(f, shape) for f in frames], _SCALE, rotation=0.0)
    ram = FieldMedian(grid, ["a", "b", "c"])
    disk = FieldMedian(grid, ["a", "b", "c"], work_dir=tmp_path / "outlier")
    for stack in (ram, disk):
        for i, (key, frame) in enumerate(zip(["a", "b", "c"], frames)):
            stack.add(key, scenes[i], dq, frame, coarse_step=None)

    assert (tmp_path / "outlier" / "median_stack.dat").exists()
    np.testing.assert_array_equal(ram.median(), disk.median())
    disk.cleanup()
    assert not (tmp_path / "outlier" / "median_stack.dat").exists()


def test_unknown_layer_raises(dithered_frames) -> None:
    frames, scenes, shape = dithered_frames
    grid = field_grid([_corners(f, shape) for f in frames], _SCALE, rotation=0.0)
    stack = FieldMedian(grid, ["a"])
    with pytest.raises(KeyError):
        stack.add("missing", scenes[0], np.zeros(shape, dtype=np.uint32), frames[0])


def test_empty_layers_rejected(dithered_frames) -> None:
    frames, _, shape = dithered_frames
    grid = field_grid([_corners(f, shape) for f in frames], _SCALE, rotation=0.0)
    with pytest.raises(ValueError, match="at least one layer"):
        FieldMedian(grid, [])
