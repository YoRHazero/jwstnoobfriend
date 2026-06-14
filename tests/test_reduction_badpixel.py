"""Tests for outlier-pixel flagging ``reduction.flag_outlier_pixels``.

Synthetic frames: sky noise plus compact sources, with injected hot/cold single
-pixel outliers in the background. The step must flag the outliers in ``dq`` and
leave the source pixels unflagged.
"""

import numpy as np

from noobfriend.reduction import flag_outlier_pixels

_SIZE = 128


def _frame() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(0)
    data = rng.normal(0.0, 1.0, size=(_SIZE, _SIZE)) + 10.0
    source = np.zeros((_SIZE, _SIZE), dtype=bool)
    source[60:68, 60:68] = True
    data[source] += 500.0
    hot = [(20, 30), (90, 100), (45, 15)]
    for y, x in hot:
        data[y, x] += 200.0  # bright single-pixel spikes (hot / CR)
    data[110, 40] -= 150.0  # a cold/dead pixel
    return data, source, hot + [(110, 40)], hot


def test_flags_outliers_and_protects_sources() -> None:
    data, source, outliers, _ = _frame()
    err = np.ones_like(data)
    dq = np.zeros(data.shape, dtype=np.int32)

    out, out_err, out_dq = flag_outlier_pixels(data, err, dq, nsigma=5.0)

    # Every injected outlier is flagged.
    for y, x in outliers:
        assert out_dq[y, x] & 1, f"outlier ({y},{x}) not flagged"
    # Source pixels are not flagged.
    assert not (out_dq[source] & 1).any()
    # data unchanged (set_nan=False), err passes through.
    assert out is data
    assert out_err is err


def test_set_nan_blanks_flagged_pixels() -> None:
    data, _, outliers, _ = _frame()
    err = np.ones_like(data)
    dq = np.zeros(data.shape, dtype=np.int32)

    out, _, out_dq = flag_outlier_pixels(data, err, dq, nsigma=5.0, set_nan=True)
    for y, x in outliers:
        assert np.isnan(out[y, x])
        assert out_dq[y, x] & 1


def test_rejects_non_2d() -> None:
    err = np.ones((2, 4, 4))
    dq = np.zeros((2, 4, 4), dtype=np.int32)
    try:
        flag_outlier_pixels(np.zeros((2, 4, 4)), err, dq)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("3-D data should raise ValueError")
