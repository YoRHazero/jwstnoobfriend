"""Tests for the morphology render geometry (WCS->affine, arcsec<->kpc)."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.extraction._wcs import tangent_plane_wcs
from noobfriend.inference.morphology._geometry import (
    build_geometry,
    center_to_subpixel,
    kpc_per_arcsec,
)

CENTER = (150.0, 2.0)


def _geom(shape: tuple[int, int] = (10, 10), oversample: int = 2):
    wcs = tangent_plane_wcs(*CENTER, 5.0, shape)  # 5 arcsec over the stamp -> 0.5"/pix
    return build_geometry(wcs, CENTER, shape, oversample)


class TestBuildGeometry:
    """Affine linearisation and the oversampled arcsec grids."""

    def test_subpixel_area_matches_pixel_scale(self) -> None:
        geom = _geom()
        # native pixel area = (0.5 arcsec)^2 = 0.25; subpixel = /oversample^2.
        assert geom.subpixel_area * geom.oversample**2 == pytest.approx(0.25, rel=1e-3)

    def test_grid_shape_is_oversampled(self) -> None:
        geom = _geom((10, 8), 3)
        assert geom.e_grid.shape == (30, 24)

    def test_centre_maps_near_zero_offset(self) -> None:
        geom = _geom()
        assert np.min(np.abs(geom.e_grid)) < 0.3
        assert np.min(np.abs(geom.n_grid)) < 0.3

    def test_east_increases_toward_lower_x(self) -> None:
        geom = _geom()
        # RA increases east; with a flipped x axis, east offset grows at small x.
        assert geom.e_grid[0, 0] > geom.e_grid[0, -1]

    def test_oversample_below_one_raises(self) -> None:
        wcs = tangent_plane_wcs(*CENTER, 5.0, (10, 10))
        with pytest.raises(ValueError, match="oversample"):
            build_geometry(wcs, CENTER, (10, 10), 0)


class TestCenterToSubpixel:
    """Mapping an arcsec offset to an oversampled index."""

    def test_zero_offset_is_grid_centre(self) -> None:
        geom = _geom()  # 10 px, oversample 2 -> 20 subpixels, centre at 9.5
        col, row = center_to_subpixel(geom, 0.0, 0.0)
        assert col == pytest.approx(9.5, abs=1e-6)
        assert row == pytest.approx(9.5, abs=1e-6)


class TestKpcPerArcsec:
    """Angular-to-physical conversion."""

    def test_highz_value_is_sane(self) -> None:
        # Proper kpc/arcsec near z~6 is ~5-6 for Planck18.
        assert 4.0 < kpc_per_arcsec(6.0) < 7.0

    def test_nonpositive_z_raises(self) -> None:
        with pytest.raises(ValueError, match="z > 0"):
            kpc_per_arcsec(0.0)
