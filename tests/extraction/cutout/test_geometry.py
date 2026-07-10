"""Tests for pixel-space cutout geometry: offset resolution and bounds."""

import numpy as np
import pytest
from astropy import units as u

from noobfriend.extraction.cutout._geometry import CutoutGeometry, resolve_offsets


class TestResolveOffsets:
    """Size-specification resolution into (left, right, top, bottom)."""

    def test_half_is_square(self) -> None:
        assert resolve_offsets(half=5) == (5, 5, 5, 5)

    def test_half_width_height(self) -> None:
        assert resolve_offsets(half_width=3, half_height=7) == (3, 3, 7, 7)

    def test_explicit_offsets(self) -> None:
        assert resolve_offsets(left=1, right=2, top=3, bottom=4) == (1, 2, 3, 4)

    def test_half_world_converts_quantity_units(self) -> None:
        # 1.5 arcsec at 3600 px/deg -> 1.5 px -> ceil -> 2. Confirms the
        # Quantity is converted from arcsec to deg (1.5 deg would give 5400 px).
        offsets = resolve_offsets(half_world=1.5 * u.arcsec, scale=(3600.0, 3600.0))
        assert offsets == (2, 2, 2, 2)

    def test_half_world_bare_float_is_degrees(self) -> None:
        # 0.5 deg * 10 px/deg = 5 px in x; * 4 px/deg = 2 px in y.
        assert resolve_offsets(half_world=0.5, scale=(10.0, 4.0)) == (5, 5, 2, 2)

    def test_half_world_ceils_fractional_pixels(self) -> None:
        # 0.01 deg * 250 px/deg = 2.5 px -> ceil -> 3.
        assert resolve_offsets(half_world=0.01, scale=(250.0, 250.0)) == (3, 3, 3, 3)

    def test_no_size_raises(self) -> None:
        with pytest.raises(ValueError, match="No size specified"):
            resolve_offsets()

    def test_over_specified_raises(self) -> None:
        with pytest.raises(ValueError, match="over-specified"):
            resolve_offsets(half=5, half_width=3, half_height=3)

    def test_partial_offsets_raises(self) -> None:
        with pytest.raises(ValueError, match="all be provided together"):
            resolve_offsets(left=1, right=2, top=3)

    def test_partial_half_width_height_raises(self) -> None:
        with pytest.raises(ValueError, match="provided together"):
            resolve_offsets(half_width=3)

    def test_half_world_without_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a pixel scale"):
            resolve_offsets(half_world=1 * u.arcsec)


class TestCutoutGeometry:
    """Resolved bounds and their derived pixel-space products."""

    def test_from_center_square(self) -> None:
        geom = CutoutGeometry.from_center(10, 20, left=2, right=2, top=2, bottom=2)
        # 2*half + 1 = 5 pixels per side, centered on the rounded pixel.
        assert geom.x_bounds == (8, 13)
        assert geom.y_bounds == (18, 23)

    def test_from_center_rounds_fractional_center(self) -> None:
        geom = CutoutGeometry.from_center(10.6, 19.4, left=1, right=1, top=1, bottom=1)
        assert geom.x_bounds == (10, 13)  # round(10.6) = 11
        assert geom.y_bounds == (18, 21)  # round(19.4) = 19

    def test_from_center_asymmetric(self) -> None:
        geom = CutoutGeometry.from_center(0, 0, left=1, right=3, top=5, bottom=2)
        assert geom.x_bounds == (-1, 4)
        assert geom.y_bounds == (-2, 6)

    def test_shape_is_rows_cols(self) -> None:
        geom = CutoutGeometry(x_bounds=(0, 4), y_bounds=(0, 7))
        assert geom.shape == (7, 4)

    def test_meshgrid_covers_half_open_bounds(self) -> None:
        geom = CutoutGeometry(x_bounds=(2, 5), y_bounds=(10, 12))
        grid_x, grid_y = geom.meshgrid
        assert grid_x.shape == geom.shape == (2, 3)
        np.testing.assert_array_equal(grid_x[0], [2, 3, 4])
        np.testing.assert_array_equal(grid_y[:, 0], [10, 11])

    def test_corners_use_exclusive_upper_bounds(self) -> None:
        geom = CutoutGeometry(x_bounds=(0, 4), y_bounds=(0, 6))
        assert geom.corners == [(0, 0), (4, 0), (4, 6), (0, 6)]


class TestPaddedToMultiple:
    """Growing bounds so each axis divides the coarse reprojection step."""

    def test_exact_multiple_is_unchanged(self) -> None:
        geom = CutoutGeometry(x_bounds=(0, 32), y_bounds=(0, 16))
        assert geom.padded_to_multiple(16, 16) == geom

    def test_pads_to_next_multiple(self) -> None:
        # cols len 33 -> 48, rows len 17 -> 32 for step 16.
        geom = CutoutGeometry(x_bounds=(0, 33), y_bounds=(0, 17))
        padded = geom.padded_to_multiple(16, 16)
        assert padded.shape == (32, 48)
        assert padded.shape[0] % 16 == 0 and padded.shape[1] % 16 == 0

    def test_splits_padding_across_both_sides(self) -> None:
        # len 33, step 16 -> pad 15: 7 on the low side, 8 on the high side.
        geom = CutoutGeometry(x_bounds=(0, 33), y_bounds=(0, 33))
        padded = geom.padded_to_multiple(16, 16)
        assert padded.x_bounds == (-7, 41)
        assert padded.y_bounds == (-7, 41)

    def test_odd_axis_lengths_like_the_clear_cutout(self) -> None:
        # Regression for the real clear shape (2449, 3249), which no nice step
        # divides; step 16 pads each axis by 15.
        geom = CutoutGeometry(x_bounds=(0, 3249), y_bounds=(0, 2449))
        padded = geom.padded_to_multiple(16, 16)
        assert padded.shape == (2464, 3264)
        assert padded.shape[0] % 16 == 0 and padded.shape[1] % 16 == 0

    def test_step_of_one_leaves_axis_unchanged(self) -> None:
        geom = CutoutGeometry(x_bounds=(0, 33), y_bounds=(0, 17))
        assert geom.padded_to_multiple(1, 1) == geom

    def test_per_axis_steps_are_independent(self) -> None:
        geom = CutoutGeometry(x_bounds=(0, 10), y_bounds=(0, 10))
        padded = geom.padded_to_multiple(4, 8)  # rows%4==0, cols%8==0
        assert padded.shape[0] % 4 == 0
        assert padded.shape[1] % 8 == 0
