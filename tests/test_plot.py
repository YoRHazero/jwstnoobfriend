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
from noobfriend.core.display.plot._spectrum import (
    _ModelCurve,
    _broadcast_colors,
    _resolve_errors,
    _resolve_wavelengths,
    _to_lines,
)
from noobfriend.core.display.plot._spectrum2d import (
    _centers_to_edges,
    _collapse_2d,
    _collapse_with_error,
    _parse_window,
    _window_row_mask,
)


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


class TestToLines:
    """Normalising a single spectrum vs several into a list of 1-D arrays."""

    def test_single_1d_array(self) -> None:
        lines = _to_lines(np.arange(5.0), "flux")
        assert len(lines) == 1 and lines[0].shape == (5,)

    def test_single_python_list(self) -> None:
        lines = _to_lines([1.0, 2.0, 3.0], "flux")
        assert len(lines) == 1 and lines[0].tolist() == [1.0, 2.0, 3.0]

    def test_sequence_of_arrays(self) -> None:
        lines = _to_lines([np.arange(3.0), np.arange(4.0)], "flux")
        assert [ln.shape for ln in lines] == [(3,), (4,)]

    def test_2d_array_is_rows(self) -> None:
        lines = _to_lines(np.zeros((3, 5)), "flux")
        assert len(lines) == 3 and all(ln.shape == (5,) for ln in lines)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="flux is empty"):
            _to_lines([], "flux")


class TestResolveWavelengths:
    """Sharing one wavelength grid or pairing one per line."""

    def test_shared_grid_broadcasts(self) -> None:
        flux = [np.arange(4.0), np.arange(4.0)]
        waves = _resolve_wavelengths(np.arange(4.0), flux)
        assert len(waves) == 2 and waves[0] is waves[1]

    def test_per_line_grids(self) -> None:
        flux = [np.arange(3.0), np.arange(3.0)]
        waves = _resolve_wavelengths([np.arange(3.0), np.arange(3.0)], flux)
        assert len(waves) == 2

    def test_count_mismatch_raises(self) -> None:
        flux = [np.arange(3.0), np.arange(3.0)]
        with pytest.raises(ValueError, match="expected 1 .shared. or 2"):
            _resolve_wavelengths([np.arange(3.0), np.arange(3.0), np.arange(3.0)], flux)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match=r"wavelength\[0\] length"):
            _resolve_wavelengths(np.arange(5.0), [np.arange(4.0)])


class TestResolveErrors:
    """Per-line optional error normalisation."""

    def test_none_is_all_none(self) -> None:
        assert _resolve_errors(None, [np.arange(3.0), np.arange(3.0)]) == [None, None]

    def test_single_sigma_for_single_line(self) -> None:
        errs = _resolve_errors(np.ones(3), [np.arange(3.0)])
        assert len(errs) == 1 and errs[0].tolist() == [1.0, 1.0, 1.0]

    def test_single_python_list_for_single_line(self) -> None:
        errs = _resolve_errors([0.1, 0.2, 0.3], [np.arange(3.0)])
        assert len(errs) == 1 and errs[0].shape == (3,)

    def test_per_line_with_a_hole(self) -> None:
        errs = _resolve_errors([np.ones(3), None], [np.arange(3.0), np.arange(3.0)])
        assert errs[0] is not None and errs[1] is None

    def test_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="expected 2"):
            _resolve_errors([np.ones(3)], [np.arange(3.0), np.arange(3.0)])

    def test_single_array_with_many_lines_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a list of 2"):
            _resolve_errors(np.ones(3), [np.arange(3.0), np.arange(3.0)])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="error length"):
            _resolve_errors(np.ones(4), [np.arange(3.0)])


class TestBroadcastColors:
    """Colour spec resolution (None lets matplotlib cycle)."""

    def test_none_is_all_none(self) -> None:
        assert _broadcast_colors(None, 2) == [None, None]

    def test_single_string_broadcasts(self) -> None:
        assert _broadcast_colors("C1", 3) == ["C1", "C1", "C1"]

    def test_sequence_used_verbatim(self) -> None:
        assert _broadcast_colors(["a", "b"], 2) == ["a", "b"]

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="colors length 1 != 2"):
            _broadcast_colors(["a"], 2)


class TestModelCurveFromInput:
    """Validation of the ModelSpec / callable / tuple model forms."""

    def test_callable_is_functional(self) -> None:
        mc = _ModelCurve.from_input(lambda w: w, 0)
        assert mc.flux_func is not None and mc.flux is None

    def test_wavelength_flux_tuple_is_sampled(self) -> None:
        mc = _ModelCurve.from_input((np.arange(3.0), np.ones(3)), 0)
        assert mc.flux_func is None and mc.wavelength is not None

    def test_dict_functional(self) -> None:
        mc = _ModelCurve.from_input({"flux_func": lambda w: w, "label": "fit"}, 0)
        assert mc.flux_func is not None and mc.label == "fit"

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown key"):
            _ModelCurve.from_input({"flux_func": lambda w: w, "colour": "r"}, 0)

    def test_both_forms_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            _ModelCurve.from_input(
                {
                    "flux_func": lambda w: w,
                    "wavelength": np.arange(3.0),
                    "flux": np.ones(3),
                },
                0,
            )

    def test_neither_form_raises(self) -> None:
        with pytest.raises(ValueError, match="provide a curve"):
            _ModelCurve.from_input({"label": "x"}, 0)

    def test_flux_without_wavelength_raises(self) -> None:
        with pytest.raises(ValueError, match="must be given together"):
            _ModelCurve.from_input({"flux": np.ones(3)}, 0)

    def test_error_with_func_raises(self) -> None:
        with pytest.raises(ValueError, match="'error' applies only"):
            _ModelCurve.from_input({"flux_func": lambda w: w, "error": np.ones(3)}, 0)

    def test_range_without_func_raises(self) -> None:
        with pytest.raises(ValueError, match="'wavelength_range' applies only"):
            _ModelCurve.from_input(
                {
                    "wavelength": np.arange(3.0),
                    "flux": np.ones(3),
                    "wavelength_range": (0.0, 1.0),
                },
                0,
            )

    def test_wavelength_flux_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="!= 'flux' length"):
            _ModelCurve.from_input((np.arange(4.0), np.ones(3)), 0)

    def test_two_tuple_of_non_arrays_raises(self) -> None:
        with pytest.raises(ValueError, match="pass a list for multiple models"):
            _ModelCurve.from_input(
                ({"flux_func": lambda w: w}, {"flux_func": lambda w: w}), 0
            )


class TestModelCurveSample:
    """Producing (x, mid, low, high) draw arrays."""

    def test_func_single_return_has_no_band(self) -> None:
        mc = _ModelCurve.from_input(lambda w: 2.0 * w, 0)
        x, mid, low, high = mc.sample((0.0, 1.0), "models[0]")
        assert low is None and high is None
        assert np.allclose(mid, 2.0 * x)

    def test_func_triple_return_is_low_mid_high(self) -> None:
        mc = _ModelCurve.from_input(lambda w: (w - 1.0, w, w + 1.0), 0)
        x, mid, low, high = mc.sample((0.0, 1.0), "models[0]")
        assert np.allclose(mid, x)
        assert np.allclose(high - low, 2.0)

    def test_func_double_return_raises(self) -> None:
        mc = _ModelCurve.from_input(lambda w: (w, w), 0)
        with pytest.raises(ValueError, match="must return 1 array .flux. or 3"):
            mc.sample((0.0, 1.0), "models[0]")

    def test_func_uses_explicit_range(self) -> None:
        mc = _ModelCurve.from_input(
            {"flux_func": lambda w: w, "wavelength_range": (5.0, 9.0)}, 0
        )
        x, _mid, _low, _high = mc.sample((0.0, 1.0), "models[0]")
        assert x[0] == pytest.approx(5.0) and x[-1] == pytest.approx(9.0)

    def test_arrays_sigma_makes_symmetric_band(self) -> None:
        mc = _ModelCurve.from_input(
            {
                "wavelength": np.arange(3.0),
                "flux": np.full(3, 10.0),
                "error": np.full(3, 2.0),
            },
            0,
        )
        _x, mid, low, high = mc.sample((0.0, 1.0), "models[0]")
        assert np.allclose(low, mid - 2.0) and np.allclose(high, mid + 2.0)

    def test_arrays_low_high_band(self) -> None:
        mc = _ModelCurve.from_input(
            {
                "wavelength": np.arange(3.0),
                "flux": np.full(3, 10.0),
                "error": (np.full(3, 8.0), np.full(3, 13.0)),
            },
            0,
        )
        _x, _mid, low, high = mc.sample((0.0, 1.0), "models[0]")
        assert np.allclose(low, 8.0) and np.allclose(high, 13.0)


class TestCentersToEdges:
    """Bin edges bracketing bin centers for the 2-D image panel."""

    def test_uniform_grid(self) -> None:
        edges = _centers_to_edges(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(edges, [0.5, 1.5, 2.5, 3.5])

    def test_single_center(self) -> None:
        assert np.allclose(_centers_to_edges(np.array([7.0])), [6.5, 7.5])

    def test_non_uniform_extrapolates_ends(self) -> None:
        edges = _centers_to_edges(np.array([0.0, 1.0, 3.0]))
        assert np.allclose(edges, [-0.5, 0.5, 2.0, 4.0])

    def test_length_is_n_plus_one(self) -> None:
        assert _centers_to_edges(np.arange(10.0)).size == 11


class TestCollapse2d:
    """NaN-aware spatial collapse fallback for the 2-D panel."""

    def test_sums_over_spatial_ignoring_nan(self) -> None:
        flux_2d = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan]])
        out = _collapse_2d(flux_2d)
        # col0: 1+3, col1: 2 (nan ignored), col2: all-nan -> nan
        assert out[0] == 4.0 and out[1] == 2.0 and np.isnan(out[2])

    def test_shape_is_wave_axis(self) -> None:
        assert _collapse_2d(np.ones((4, 7))).shape == (7,)


class TestCollapseWithError:
    """Joint flux (boxcar) + error (quadrature) collapse for the 2-D panel."""

    def test_quadrature_error(self) -> None:
        flux_1d, error_1d = _collapse_with_error(
            np.array([[3.0], [4.0]]), np.array([[3.0], [4.0]])
        )
        assert flux_1d[0] == 7.0
        assert error_1d[0] == pytest.approx(5.0)  # sqrt(9 + 16)

    def test_nan_cell_dropped_from_both(self) -> None:
        flux = np.array([[1.0, 2.0], [3.0, np.nan]])
        err = np.array([[1.0, 2.0], [1.0, np.nan]])
        flux_1d, error_1d = _collapse_with_error(flux, err)
        # col1 row1 is NaN -> only row0 contributes
        assert flux_1d[1] == 2.0 and error_1d[1] == pytest.approx(2.0)

    def test_nonpositive_error_dropped(self) -> None:
        flux_1d, error_1d = _collapse_with_error(
            np.array([[5.0], [5.0]]), np.array([[0.0], [2.0]])
        )
        # row0 has error 0 -> excluded
        assert flux_1d[0] == 5.0 and error_1d[0] == pytest.approx(2.0)

    def test_all_invalid_column_is_nan(self) -> None:
        flux_1d, error_1d = _collapse_with_error(
            np.full((2, 1), np.nan), np.full((2, 1), 1.0)
        )
        assert np.isnan(flux_1d[0]) and np.isnan(error_1d[0])


class TestSpatialWindow:
    """Parsing and row-masking of the 2-D panel collapse window."""

    def test_parse_returns_sorted_pair(self) -> None:
        assert _parse_window([3.0, 1.0]) == (1.0, 3.0)

    def test_parse_accepts_tuple_and_list(self) -> None:
        assert _parse_window((1.0, 2.0)) == _parse_window([1.0, 2.0])

    def test_parse_bad_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must be .low, high."):
            _parse_window([1.0, 2.0, 3.0])

    def test_row_mask_selects_inside(self) -> None:
        mask = _window_row_mask(np.arange(10.0), (2.0, 5.0))
        assert mask.tolist() == [False, False, True, True, True, True] + [False] * 4

    def test_row_mask_order_insensitive(self) -> None:
        a = _window_row_mask(np.arange(10.0), (5.0, 2.0))
        b = _window_row_mask(np.arange(10.0), (2.0, 5.0))
        assert np.array_equal(a, b)

    def test_row_mask_empty_selection_raises(self) -> None:
        with pytest.raises(ValueError, match="selects no rows"):
            _window_row_mask(np.arange(10.0), (100.0, 200.0))
