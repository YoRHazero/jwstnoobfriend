"""Tests for the 2-D background step ``reduction.subtract_background``.

Synthetic frames: sky noise plus compact sources, on top of a known smooth
background (a gentle tilted plane plus a broad Gaussian glow). The step must
remove the smooth background while preserving source flux, and pass ``err`` /
``dq`` through. Assertions are on the frame *interior*: like any mesh background
(e.g. photutils ``Background2D``), the outermost half-box border can only be
extrapolated, so it is excluded.
"""

import numpy as np

from noobfriend.reduction import subtract_background

# box=16 on a 256 frame gives a 16x16 mesh -- the same grid resolution as the
# default box_size=128 on a 2048 NIRCam frame, so the default filter_size=3 is
# exercised representatively.
_SIZE = 256
_BOX = 16


def _frame(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``(sources_only, with_background, source_mask)`` synthetic frames."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    sky = rng.normal(0.0, 1.0, size=(_SIZE, _SIZE))

    source = np.zeros((_SIZE, _SIZE), dtype=bool)
    for cy, cx in ((60, 80), (180, 200), (120, 40), (200, 150)):
        source[cy - 3 : cy + 4, cx - 3 : cx + 4] = True
    truth = sky.copy()
    truth[source] += 400.0  # bright compact sources

    background = (
        20.0
        + 0.01 * xx
        + 0.006 * yy
        + 60.0 * np.exp(-(((xx - 120) ** 2 + (yy - 130) ** 2) / (2 * 80.0**2)))
    )
    return truth, truth + background, source


def _interior(arr: np.ndarray) -> np.ndarray:
    return arr[_BOX:-_BOX, _BOX:-_BOX]


def test_removes_smooth_background_preserving_sources() -> None:
    truth, dirty, source = _frame()
    err = np.ones_like(dirty)
    dq = np.zeros(dirty.shape, dtype=np.int32)

    out, out_err, out_dq = subtract_background(dirty, err, dq, box_size=_BOX)

    bg = _interior(~source)
    out_in = _interior(out)
    dirty_in = _interior(dirty)
    assert np.std(dirty_in[bg]) > 10.0
    assert np.std(out_in[bg]) < 2.0
    assert abs(np.mean(out_in[bg])) < 1.0

    # Source flux above the local background is preserved.
    flux_truth = truth[source].sum() - np.median(truth[~source]) * source.sum()
    flux_out = out[source].sum() - np.median(out[~source]) * source.sum()
    assert np.isclose(flux_out, flux_truth, rtol=0.05)

    assert out_err is err
    assert out_dq is dq


def test_external_mask_path() -> None:
    _, dirty, source = _frame(seed=2)
    err = np.ones_like(dirty)
    dq = np.zeros(dirty.shape, dtype=np.int32)

    out, _, _ = subtract_background(dirty, err, dq, box_size=_BOX, mask=source)
    assert np.std(_interior(out)[_interior(~source)]) < 2.0


def test_fully_masked_region_inpainted_from_neighbours() -> None:
    # A block larger than a box, masked entirely over background-only pixels: its
    # boxes fall below min_valid_fraction and must be inpainted from neighbours,
    # recovering the smooth background rather than leaving a hole or trusting a
    # few-pixel median.
    # A smooth plane (no peak hidden under the block); biharmonic inpainting
    # recovers a linear surface near-exactly from the surrounding boxes.
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    bg = 20.0 + 0.01 * xx + 0.006 * yy
    dirty = bg + rng.normal(0.0, 1.0, size=(_SIZE, _SIZE))
    err = np.ones_like(dirty)
    dq = np.zeros(dirty.shape, dtype=np.int32)
    block = np.zeros(dirty.shape, dtype=bool)
    block[100:148, 100:148] = True  # 3x3 boxes, fully masked

    out, _, _ = subtract_background(dirty, err, dq, box_size=_BOX, mask=block)

    inner = out[112:136, 112:136]  # interior of the masked block
    assert abs(np.mean(inner)) < 2.0
    assert np.std(inner) < 2.5


def test_rejects_non_2d_and_bad_box() -> None:
    err = np.ones((8, 8))
    dq = np.zeros((8, 8), dtype=np.int32)
    try:
        subtract_background(np.zeros((2, 8, 8)), err, dq)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("3-D data should raise ValueError")

    try:
        subtract_background(np.zeros((8, 8)), err, dq, box_size=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("box_size < 1 should raise ValueError")
