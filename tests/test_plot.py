"""Tests for plot helpers: percentile limits, grid sizing, Mollweide projection."""

import numpy as np
import pytest

from noobfriend.core.display.plot._footprint import (
    _resolve_colors,
    orthographic,
    reference_center,
)
from noobfriend.core.display.plot._image import _grid_axis
from noobfriend.core.display.plot._norm import percentile_limits, resolve_limits
from noobfriend.core.display.plot._blink import _broadcast_limits, _union_bounds


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


class TestResolveLimits:
    """Explicit ``vmin``/``vmax`` taking precedence over percentile fallback."""

    def test_both_explicit_skip_percentiles(self) -> None:
        # Data is all-NaN, so any percentile use would fall back to (0, 1);
        # getting (-2, 5) proves the percentile path was not taken.
        data = np.full(4, np.nan)
        assert resolve_limits(data, -2.0, 5.0, 1.0, 99.0) == (-2.0, 5.0)

    def test_vmin_only_uses_percentile_for_upper(self) -> None:
        data = np.arange(101, dtype=float)
        assert resolve_limits(data, 10.0, None, 0.0, 100.0) == (10.0, 100.0)

    def test_vmax_only_uses_percentile_for_lower(self) -> None:
        data = np.arange(101, dtype=float)
        assert resolve_limits(data, None, 42.0, 0.0, 100.0) == (0.0, 42.0)

    def test_neither_is_pure_percentile(self) -> None:
        data = np.arange(101, dtype=float)
        assert resolve_limits(data, None, None, 0.0, 100.0) == (0.0, 100.0)

    def test_equal_explicit_limits_widened(self) -> None:
        low, high = resolve_limits(np.arange(10.0), 5.0, 5.0, 1.0, 99.0)
        assert low < high

    def test_inverted_explicit_limits_raise(self) -> None:
        with pytest.raises(ValueError, match="require vmin < vmax"):
            resolve_limits(np.arange(10.0), 9.0, 1.0, 1.0, 99.0)


class TestBroadcastLimits:
    """Broadcasting a scalar/sequence/None ``vmin``/``vmax`` to one per image."""

    def test_none_broadcasts_none(self) -> None:
        assert _broadcast_limits(None, 3, "vmin") == [None, None, None]

    def test_scalar_broadcasts(self) -> None:
        assert _broadcast_limits(2.0, 3, "vmin") == [2.0, 2.0, 2.0]

    def test_numpy_scalar_is_scalar(self) -> None:
        assert _broadcast_limits(np.float32(2.0), 2, "vmin") == [2.0, 2.0]

    def test_sequence_is_per_image(self) -> None:
        assert _broadcast_limits([1.0, 2.0, 3.0], 3, "vmax") == [1.0, 2.0, 3.0]

    def test_sequence_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="vmax has length 2, expected 3"):
            _broadcast_limits([1.0, 2.0], 3, "vmax")


class TestUnionBounds:
    """Axis-aligned bounds of offset frames, origin lower-left."""

    def test_single_frame_at_origin(self) -> None:
        img = np.zeros((4, 6))  # 4 rows, 6 cols
        assert _union_bounds([img], [(0.0, 0.0)]) == (0.0, 6.0, 0.0, 4.0)

    def test_offsets_extend_bounds_both_ways(self) -> None:
        a = np.zeros((4, 4))
        b = np.zeros((4, 4))
        # b shifted to (-2, 3): union spans x in [-2, 4], y in [0, 7].
        assert _union_bounds([a, b], [(0.0, 0.0), (-2.0, 3.0)]) == (-2.0, 4.0, 0.0, 7.0)

    def test_differing_shapes(self) -> None:
        a = np.zeros((2, 2))
        b = np.zeros((5, 3))  # 5 rows, 3 cols
        assert _union_bounds([a, b], [(0.0, 0.0), (1.0, 0.0)]) == (0.0, 4.0, 0.0, 5.0)


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
