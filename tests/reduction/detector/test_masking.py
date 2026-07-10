"""Tests for the wing-tier source exclusion in ``reduction.detector._masking``.

Synthetic frames only: a big bright disk (above the ``big_px`` area gate) must
receive a wing exclusion scaled to its equivalent radius, while a small source
keeps only the base dilation. This guards the fix for 1/f over-subtraction
streaks through bright/extended sources (wings below the detection threshold
biasing the row/column medians).
"""

import numpy as np

from noobfriend.reduction.detector._masking import source_exclusion


def _frame_with_disk(radius: int, *, size: int = 256, seed: int = 0) -> np.ndarray:
    """Sky-noise frame with a bright disk of the given radius at the center."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(size, size))
    yy, xx = np.mgrid[:size, :size]
    disk = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 <= radius * radius
    data[disk] += 50.0
    return data


def test_big_segment_gets_scaled_wing_exclusion() -> None:
    radius = 14  # area ~615 px > big_px=300 -> wing radius ~2*14=28 beyond edge
    data = _frame_with_disk(radius)
    dq = np.zeros(data.shape, dtype=np.int32)

    excl = source_exclusion(data, dq)

    c = data.shape[0] // 2
    # well inside the wing zone (edge + ~half the wing radius): excluded
    assert excl[c, c + radius + 14]
    # well beyond the wing zone: not excluded
    assert not excl[c, c + radius + 40]


def test_small_segment_keeps_base_dilation_only() -> None:
    rng = np.random.default_rng(1)
    data = rng.normal(0.0, 1.0, size=(128, 128))
    data[60:64, 60:64] += 50.0  # 16 px, far below the big_px gate
    dq = np.zeros(data.shape, dtype=np.int32)

    excl = source_exclusion(data, dq)

    # base dilation (3) reaches a few px out; no wing exclusion beyond that
    assert excl[62, 65]
    assert not excl[62, 75]


def test_wing_cap_and_disable() -> None:
    data = _frame_with_disk(14)
    dq = np.zeros(data.shape, dtype=np.int32)
    c = data.shape[0] // 2

    capped = source_exclusion(data, dq, wing_max=5.0)
    assert not capped[c, c + 14 + 12]  # beyond cap + base dilation

    disabled = source_exclusion(data, dq, wing_scale=0.0)
    assert not disabled[c, c + 14 + 12]
    # detection + base dilation itself still works
    assert disabled[c, c + 14 + 1]


def test_caller_mask_skips_wing_tier() -> None:
    data = _frame_with_disk(14)
    dq = np.zeros(data.shape, dtype=np.int32)
    manual = np.zeros(data.shape, dtype=bool)
    manual[:10, :10] = True

    excl = source_exclusion(data, dq, mask=manual)

    c = data.shape[0] // 2
    assert excl[5, 5]
    assert not excl[c, c + 14 + 14]  # no auto detection, no wing zone
