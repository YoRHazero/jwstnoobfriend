"""Tests for plot helpers: percentile limits, grid sizing, Mollweide projection."""

import numpy as np
import pytest

from noobfriend.core.display.plot._footprint import (
    _resolve_colors,
    orthographic,
    reference_center,
)
from noobfriend.core.display.plot._image import _grid_axis
from noobfriend.core.display.plot._norm import percentile_limits


class TestPercentileLimits:
    """Percentile-cut intensity limits, ignoring non-finite pixels."""

    def test_full_range(self) -> None:
        data = np.arange(101, dtype=float)
        assert percentile_limits(data, 0.0, 100.0) == (0.0, 100.0)

    def test_ignores_non_finite(self) -> None:
        data = np.array([np.nan, np.inf, -np.inf, 0.0, 100.0])
        # Limits are computed from the finite pair {0, 100} only.
        assert percentile_limits(data, 0.0, 100.0) == (0.0, 100.0)

    def test_flat_region_is_widened(self) -> None:
        low, high = percentile_limits(np.full(10, 5.0), 1.0, 99.0)
        assert low < high

    def test_all_non_finite_falls_back(self) -> None:
        assert percentile_limits(np.full(4, np.nan), 1.0, 99.0) == (0.0, 1.0)

    def test_invalid_percentiles_raise(self) -> None:
        with pytest.raises(ValueError, match="0 <= pmin < pmax <= 100"):
            percentile_limits(np.arange(10.0), 99.0, 1.0)


class TestGridAxis:
    """Adaptive RA/Dec hover grid sizing."""

    def test_scales_with_length(self) -> None:
        assert _grid_axis(2048) > _grid_axis(512)

    def test_clamped_to_minimum(self) -> None:
        assert _grid_axis(1) == 2

    def test_clamped_to_maximum(self) -> None:
        assert _grid_axis(10**6) == 129


class TestReferenceCenter:
    """Field-centre estimation across the RA = 0/360 seam."""

    def test_circular_mean_across_seam(self) -> None:
        # Vertices straddling the seam average to ~0 deg, not ~180.
        poly = np.array([[359.0, -1.0], [1.0, -1.0], [1.0, 1.0], [359.0, 1.0]])
        ra0, dec0 = reference_center([poly])
        assert min(ra0, 360.0 - ra0) == pytest.approx(0.0, abs=1e-6)
        assert dec0 == pytest.approx(0.0)


class TestResolveColors:
    """Colour-spec resolution (None / single / per-item)."""

    def test_none_uses_default(self) -> None:
        assert _resolve_colors(None, 2, ["a", "b"], "colors") == ["a", "b"]

    def test_single_string_broadcasts(self) -> None:
        assert _resolve_colors("red", 3, ["x"], "colors") == ["red", "red", "red"]

    def test_sequence_used_verbatim(self) -> None:
        assert _resolve_colors(["a", "b"], 2, ["x", "y"], "colors") == ["a", "b"]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="colors length 1 != 2"):
            _resolve_colors(["a"], 2, ["x", "y"], "colors")


class TestOrthographic:
    """Orthographic globe projection."""

    def test_center_maps_to_origin(self) -> None:
        x, y = orthographic(np.array([120.0]), np.array([30.0]), 120.0, 30.0)
        assert x.item() == pytest.approx(0.0, abs=1e-9)
        assert y.item() == pytest.approx(0.0, abs=1e-9)

    def test_pole_above_equator_centre(self) -> None:
        # North pole, viewed from a centre on the equator: top of the disk.
        x, y = orthographic(np.array([45.0]), np.array([90.0]), 0.0, 0.0)
        assert x.item() == pytest.approx(0.0, abs=1e-9)
        assert y.item() == pytest.approx(1.0, abs=1e-9)

    def test_far_hemisphere_is_nan(self) -> None:
        # A point 180 deg away is on the back of the globe -> hidden.
        x, y = orthographic(np.array([180.0]), np.array([0.0]), 0.0, 0.0)
        assert np.isnan(x).all() and np.isnan(y).all()

    def test_relative_position_and_unit_disk(self) -> None:
        # East of centre -> +x, west -> -x; visible points stay in the unit disk.
        x, y = orthographic(np.array([10.0, -10.0]), np.array([0.0, 0.0]), 0.0, 0.0)
        assert x[0] > 0 and x[1] < 0
        assert np.all(x**2 + y**2 <= 1.0 + 1e-9)
