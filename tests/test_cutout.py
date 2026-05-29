"""Tests for the slice primitive and the Cutout factory / extraction paths.

A small linear fake WCS stands in for a real GWCS so these tests stay free of
any FITS dependency while still exercising the world<->pixel plumbing.
"""

from typing import Any, Callable

import numpy as np
import pytest

from noobfriend.extraction.cutout import Cutout
from noobfriend.extraction.cutout._slice import slice_with_fill


class LinearWCS:
    """A minimal affine stand-in for a GWCS, with no grism frame.

    ``detector -> world``: ``ra = (x - x0) * scale + ra0`` (likewise for dec).
    The inverse maps world back to detector. Only the two frames the cutout
    code requests are supported.
    """

    available_frames = ["detector", "world"]

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        ra0: float = 0.0,
        dec0: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        self.x0 = x0
        self.y0 = y0
        self.ra0 = ra0
        self.dec0 = dec0
        self.scale = scale

    def get_transform(
        self, from_frame: str, to_frame: str
    ) -> Callable[[Any, Any], tuple[Any, Any]]:
        """Return the requested transform; only world<->detector is supported."""
        if (from_frame, to_frame) == ("world", "detector"):
            return self._world_to_detector
        if (from_frame, to_frame) == ("detector", "world"):
            return self._detector_to_world
        raise ValueError(f"Unsupported transform: {from_frame!r} -> {to_frame!r}.")

    def _world_to_detector(self, ra: Any, dec: Any) -> tuple[Any, Any]:
        return (
            (ra - self.ra0) / self.scale + self.x0,
            (dec - self.dec0) / self.scale + self.y0,
        )

    def _detector_to_world(self, x: Any, y: Any) -> tuple[Any, Any]:
        return (
            (x - self.x0) * self.scale + self.ra0,
            (y - self.y0) * self.scale + self.dec0,
        )


class TestSliceWithFill:
    """The out-of-bounds-aware slicing primitive."""

    def test_fully_in_bounds(self) -> None:
        data = np.arange(20).reshape(4, 5)  # rows=4, cols=5
        out = slice_with_fill(data, x_bounds=(1, 4), y_bounds=(1, 3))
        np.testing.assert_array_equal(out, data[1:3, 1:4])

    def test_partial_out_of_bounds_pads(self) -> None:
        data = np.ones((3, 3))
        out = slice_with_fill(data, x_bounds=(1, 4), y_bounds=(1, 4), fill_value=-1.0)
        assert out.shape == (3, 3)
        # Overlap is the bottom-right 2x2 of data; the rest is fill.
        np.testing.assert_array_equal(out[:2, :2], 1.0)
        assert out[2, 2] == -1.0
        assert out[0, 2] == -1.0

    def test_completely_out_of_bounds_is_all_fill(self) -> None:
        data = np.ones((3, 3))
        out = slice_with_fill(
            data, x_bounds=(10, 13), y_bounds=(10, 13), fill_value=7.0
        )
        assert out.shape == (3, 3)
        np.testing.assert_array_equal(out, 7.0)

    def test_negative_bounds_pad_low_edge(self) -> None:
        data = np.arange(9).reshape(3, 3)
        out = slice_with_fill(data, x_bounds=(-1, 2), y_bounds=(-1, 2), fill_value=0.0)
        assert out.shape == (3, 3)
        # data[0, 0] lands at out[1, 1]; first row/col are fill.
        assert out[1, 1] == data[0, 0]
        np.testing.assert_array_equal(out[0, :], 0.0)

    def test_nan_fill_on_int_data_promotes_to_float(self) -> None:
        data = np.arange(9, dtype=np.int32).reshape(3, 3)
        out = slice_with_fill(data, x_bounds=(0, 4), y_bounds=(0, 4))
        assert np.issubdtype(out.dtype, np.floating)
        assert np.isnan(out[3, 3])

    def test_int_fill_on_int_data_stays_int(self) -> None:
        data = np.arange(9, dtype=np.int32).reshape(3, 3)
        out = slice_with_fill(data, x_bounds=(0, 4), y_bounds=(0, 4), fill_value=0)
        assert np.issubdtype(out.dtype, np.integer)


class TestCutoutFactories:
    """The three explicit constructors."""

    def test_from_bounds(self) -> None:
        wcs = LinearWCS()
        cut = Cutout.from_bounds(wcs, x_bounds=(10, 20), y_bounds=(5, 15))
        assert cut.geometry.x_bounds == (10, 20)
        assert cut.geometry.y_bounds == (5, 15)

    def test_from_pixel_half(self) -> None:
        cut = Cutout.from_pixel(LinearWCS(), x=100, y=200, half=3)
        assert cut.geometry.x_bounds == (97, 104)
        assert cut.geometry.y_bounds == (197, 204)

    def test_from_world_maps_center_to_pixel(self) -> None:
        wcs = LinearWCS(x0=100, y0=200, ra0=10.0, dec0=20.0, scale=0.001)
        cut = Cutout.from_world(wcs, ra=10.0, dec=20.0, half=2)
        # ra0/dec0 map to (x0, y0) = (100, 200).
        assert cut.geometry.x_bounds == (98, 103)
        assert cut.geometry.y_bounds == (198, 203)

    def test_invalid_size_propagates(self) -> None:
        with pytest.raises(ValueError, match="No size specified"):
            Cutout.from_pixel(LinearWCS(), x=0, y=0)


class TestCutoutExtraction:
    """End-to-end array extraction, including cross-WCS alignment."""

    def test_array_same_wcs_slices(self) -> None:
        data = np.arange(100).reshape(10, 10)
        cut = Cutout.from_bounds(LinearWCS(), x_bounds=(2, 5), y_bounds=(3, 6))
        out = cut.array(data)
        np.testing.assert_array_equal(out, data[3:6, 2:5])

    def test_array_shifted_wcs_compensates_offset(self) -> None:
        # data_wcs is the same as ref_wcs but its detector origin is +5 in x and
        # +3 in y, so the slice window should shift by the same amount.
        ref_wcs = LinearWCS(x0=0, y0=0, ra0=0.0, dec0=0.0, scale=0.001)
        data_wcs = LinearWCS(x0=5, y0=3, ra0=0.0, dec0=0.0, scale=0.001)
        data = np.arange(400).reshape(20, 20)

        cut = Cutout.from_bounds(ref_wcs, x_bounds=(4, 7), y_bounds=(4, 7))
        out = cut.array(data, wcs=data_wcs)
        np.testing.assert_array_equal(out, data[7:10, 9:12])


class TestCutoutReproject:
    """Reprojection via noobase, focused on our corner/offset wiring."""

    def test_reproject_identity_recovers_subregion(self) -> None:
        # Source and target share one linear WCS, so the flux-conserving
        # reprojection of the cutout sub-window must reproduce the same pixels
        # a plain slice would — this validates the cutout-local -> full-frame
        # offset applied to target_pixel_to_world.
        wcs = LinearWCS(x0=0, y0=0, ra0=0.0, dec0=0.0, scale=0.001)
        data = np.arange(400, dtype=np.float64).reshape(20, 20)

        cut = Cutout.from_bounds(wcs, x_bounds=(2, 5), y_bounds=(3, 6))
        image, footprint, weight = cut.reproject(data, wcs)

        assert image.shape == footprint.shape == weight.shape == (3, 3)
        np.testing.assert_allclose(image, data[3:6, 2:5], atol=1e-9)
        np.testing.assert_allclose(footprint, 1.0, atol=1e-9)
        np.testing.assert_allclose(weight, 1.0, atol=1e-9)

    def test_reproject_accepts_big_endian_float(self) -> None:
        # JWST FITS science arrays are big-endian float32 (">f4"); noobase wants
        # native-endian float, so the reproject path must coerce. Regression for
        # "image_in must be ... float32 or float64".
        wcs = LinearWCS(x0=0, y0=0, ra0=0.0, dec0=0.0, scale=0.001)
        data = np.arange(400, dtype=">f4").reshape(20, 20)

        cut = Cutout.from_bounds(wcs, x_bounds=(2, 5), y_bounds=(3, 6))
        image, _, _ = cut.reproject(data, wcs)
        np.testing.assert_allclose(
            image, np.asarray(data, dtype=np.float64)[3:6, 2:5], atol=1e-9
        )

    def test_reproject_outside_source_has_zero_footprint(self) -> None:
        # A cutout window entirely off the source image yields no coverage.
        wcs = LinearWCS(x0=0, y0=0, ra0=0.0, dec0=0.0, scale=0.001)
        data = np.ones((10, 10), dtype=np.float64)

        cut = Cutout.from_bounds(wcs, x_bounds=(100, 103), y_bounds=(100, 103))
        image, footprint, weight = cut.reproject(data, wcs)

        assert image.shape == (3, 3)
        np.testing.assert_allclose(footprint, 0.0, atol=1e-9)
        np.testing.assert_allclose(weight, 0.0, atol=1e-9)
        assert np.isnan(image).all()
