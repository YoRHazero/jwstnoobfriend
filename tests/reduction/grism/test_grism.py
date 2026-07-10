"""Tests for the grism reduction primitives.

All synthetic (no FITS): ``grism_trace_mask`` must cover injected horizontal
traces and bad pixels while leaving most background unmasked; the sky-template
helpers must downsample with per-block rejection, median away cross-frame
outliers, and recover a known per-frame scale.
"""

import numpy as np
import pytest

from noobfriend.reduction.grism import (
    combine_sky_template,
    fit_template_scalar,
    grism_trace_mask,
    sky_residual_grid,
    subtract_sky_template,
)

_SIZE = 256


def _grism_frame(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return ``(data, dq, trace_rows)``: sky gradient + noise + horizontal traces."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    data = 0.3 + 0.4 * (xx / _SIZE) + rng.normal(0, 0.01, (_SIZE, _SIZE))
    trace_rows = [60, 130, 200]
    for r in trace_rows:
        data[r, :] += 1.0  # bright dispersed trace, far above 4 sigma
    dq = np.zeros((_SIZE, _SIZE), dtype=np.int32)
    dq[10, 10] = 1  # a DO_NOT_USE pixel
    return data, dq, trace_rows


def _grismc_frame(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return ``(data, dq, trace_cols)``: sky gradient + noise + vertical traces."""
    rng = np.random.default_rng(seed)
    yy, _ = np.mgrid[0:_SIZE, 0:_SIZE]
    data = 0.3 + 0.4 * (yy / _SIZE) + rng.normal(0, 0.01, (_SIZE, _SIZE))
    trace_cols = [60, 130, 200]
    for c in trace_cols:
        data[:, c] += 1.0  # bright dispersed trace, far above 4 sigma
    dq = np.zeros((_SIZE, _SIZE), dtype=np.int32)
    dq[10, 10] = 1  # a DO_NOT_USE pixel
    return data, dq, trace_cols


def test_trace_mask_covers_traces_and_bad_but_not_background() -> None:
    """Traces and flagged pixels are masked; most background is left clean."""
    data, dq, trace_rows = _grism_frame()
    mask = grism_trace_mask(data, dq, nsigma=4.0)

    for r in trace_rows:
        assert mask[r].mean() > 0.95  # each trace row almost fully masked
    assert mask[10, 10]  # DO_NOT_USE pixel excluded
    assert mask.mean() < 0.3  # background not swallowed
    # a row between traces is overwhelmingly background
    assert mask[100].mean() < 0.1


def test_trace_mask_covers_vertical_grismc_traces() -> None:
    """``dispersion_axis='y'`` follows vertical GRISMC traces."""
    data, dq, trace_cols = _grismc_frame()
    mask = grism_trace_mask(data, dq, nsigma=4.0, dispersion_axis="y")

    for c in trace_cols:
        assert mask[:, c].mean() > 0.95  # each trace column almost fully masked
    assert mask[10, 10]  # DO_NOT_USE pixel excluded
    assert mask.mean() < 0.3  # background not swallowed
    # a column between traces is overwhelmingly background
    assert mask[:, 100].mean() < 0.1


def test_trace_mask_rejects_non_2d() -> None:
    """A non-2-D input raises ``ValueError``."""
    with pytest.raises(ValueError, match="2-D"):
        grism_trace_mask(np.zeros((4, 4, 4)), np.zeros((4, 4, 4), dtype=int))


def test_trace_mask_rejects_bad_dispersion_axis() -> None:
    """Only detector ``x`` or ``y`` can be used as the trace-growth axis."""
    with pytest.raises(ValueError, match="dispersion_axis"):
        grism_trace_mask(
            np.zeros((4, 4)),
            np.zeros((4, 4), dtype=int),
            dispersion_axis="z",  # type: ignore[arg-type]
        )


def test_sky_residual_grid_downsamples_and_excludes_mask() -> None:
    """Fully-masked blocks become NaN; clean blocks carry the block median."""
    data = np.full((8, 8), 2.0)
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:4, 0:4] = True  # mask one 4x4 block entirely

    grid = sky_residual_grid(data, mask, factor=4)

    assert grid.shape == (2, 2)
    assert np.isnan(grid[0, 0])  # fully masked -> NaN
    assert np.allclose(grid[~np.isnan(grid)], 2.0)


def test_combine_sky_template_medians_out_cross_frame_outliers() -> None:
    """A single outlier grid does not survive the cross-frame median."""
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    grids = [base, base, base, base + 100.0]  # one wild outlier frame

    template = combine_sky_template(grids, (8, 8), smooth_sigma=0.0)

    assert template.shape == (8, 8)
    assert template.max() < base.max() + 1.0  # outlier rejected by the median


def test_fit_and_subtract_template_recovers_scale() -> None:
    """The scalar fit recovers a known amplitude and the subtraction flattens it."""
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    template = np.exp(-(((xx - 128) ** 2 + (yy - 128) ** 2) / (2 * 50.0**2)))
    rng = np.random.default_rng(1)
    data = 0.5 * template + rng.normal(0, 1e-3, (_SIZE, _SIZE))

    s = fit_template_scalar(data, template)
    assert abs(s - 0.5) < 0.02

    out, err, dq = subtract_sky_template(
        data, np.zeros_like(data), np.zeros_like(data, dtype=int), template
    )
    assert np.std(out) < 0.1 * np.std(0.5 * template)  # template removed
    assert err is not None and dq is not None
