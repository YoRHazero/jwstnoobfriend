"""Tests for the morphology data layer (``NoobImage`` + band normalisation).

Pure-structure: container assembly, validation, and the immutable select / merge /
with_masks operations. No real FITS -- WCS is either synthesised by the image layer
or built with ``tangent_plane_wcs``.
"""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.extraction._wcs import tangent_plane_wcs
from noobfriend.inference.morphology import NoobImage

CENTER = (150.0, 2.0)


def _spec(shape: tuple[int, int] = (5, 4), *, with_wcs: bool = True, **extra) -> dict:
    """Build a minimal band spec dict over ``shape``."""
    spec: dict = {"data": np.ones(shape), "error": np.full(shape, 0.1)}
    if with_wcs:
        spec["wcs"] = tangent_plane_wcs(*CENTER, 3.0, shape)
    spec.update(extra)
    return spec


def _image(**bands) -> NoobImage:
    """Build a NoobImage from named band specs at the shared centre."""
    return NoobImage.from_bands(bands or {"F150W": _spec()}, z=6.1, center=CENTER)


class TestFromBands:
    """Construction, validation, and WCS synthesis."""

    def test_builds_and_preserves_band_order(self) -> None:
        img = NoobImage.from_bands(
            {"F150W": _spec(), "F444W": _spec()}, z=6.1, center=CENTER
        )
        assert img.filters == ["F150W", "F444W"]
        assert len(img) == 2
        assert "F150W" in img and "F999W" not in img

    def test_empty_bands_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            NoobImage.from_bands({}, z=6.1, center=CENTER)

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown key"):
            NoobImage.from_bands(
                {"F150W": _spec(dq=np.ones((5, 4)))}, z=6.1, center=CENTER
            )

    def test_missing_data_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required key 'data'"):
            NoobImage.from_bands(
                {"F150W": {"error": np.ones((5, 4))}}, z=6.1, center=CENTER
            )

    def test_missing_error_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required key 'error'"):
            NoobImage.from_bands(
                {"F150W": {"data": np.ones((5, 4))}}, z=6.1, center=CENTER
            )

    def test_error_shape_mismatch_raises(self) -> None:
        spec = {"data": np.ones((5, 4)), "error": np.ones((5, 5))}
        with pytest.raises(ValueError, match="error shape"):
            NoobImage.from_bands({"F150W": spec}, z=6.1, center=CENTER)

    def test_z_at_or_below_minus_one_raises(self) -> None:
        with pytest.raises(ValueError, match="z must be > -1"):
            NoobImage.from_bands({"F150W": _spec()}, z=-1.0, center=CENTER)

    def test_dec_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="dec must be in"):
            NoobImage.from_bands({"F150W": _spec()}, z=6.1, center=(150.0, 95.0))

    def test_missing_wcs_without_size_raises(self) -> None:
        with pytest.raises(ValueError, match="no 'wcs'"):
            NoobImage.from_bands({"F150W": _spec(with_wcs=False)}, z=6.1, center=CENTER)

    def test_missing_wcs_is_synthesised_from_size(self) -> None:
        img = NoobImage.from_bands(
            {"F150W": _spec(with_wcs=False)},
            z=6.1,
            center=CENTER,
            cutout_size_arcsec=3.0,
        )
        assert img.bands["F150W"].wcs is not None

    def test_psf_must_be_2d(self) -> None:
        with pytest.raises(ValueError, match="psf must be 2-D"):
            NoobImage.from_bands({"F150W": _spec(psf=np.ones(7))}, z=6.1, center=CENTER)


class TestMask:
    """Fit-mask handling at construction and via with_masks."""

    def test_mask_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mask shape"):
            NoobImage.from_bands(
                {"F150W": _spec(mask=np.zeros((5, 5), dtype=bool))},
                z=6.1,
                center=CENTER,
            )

    def test_mask_coerced_to_bool(self) -> None:
        img = NoobImage.from_bands(
            {"F150W": _spec(mask=np.zeros((5, 4), dtype=int))}, z=6.1, center=CENTER
        )
        assert img.bands["F150W"].mask.dtype == bool

    def test_with_masks_overrides_one_band(self) -> None:
        img = NoobImage.from_bands(
            {"F150W": _spec(), "F444W": _spec()}, z=6.1, center=CENTER
        )
        new_mask = np.ones((5, 4), dtype=bool)
        out = img.with_masks({"F150W": new_mask})
        assert out.bands["F150W"].mask is not None and out.bands["F150W"].mask.all()
        assert out.bands["F444W"].mask is None  # untouched
        assert img.bands["F150W"].mask is None  # original unchanged (immutable)

    def test_with_masks_can_clear(self) -> None:
        img = NoobImage.from_bands(
            {"F150W": _spec(mask=np.ones((5, 4), dtype=bool))}, z=6.1, center=CENTER
        )
        out = img.with_masks({"F150W": None})
        assert out.bands["F150W"].mask is None

    def test_with_masks_unknown_band_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown band"):
            _image().with_masks({"F999W": np.ones((5, 4), dtype=bool)})

    def test_with_masks_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="mask shape"):
            _image().with_masks({"F150W": np.ones((9, 9), dtype=bool)})


class TestSelect:
    """Band-subset extraction (the manual include toggle)."""

    def test_select_subset_in_requested_order(self) -> None:
        img = NoobImage.from_bands(
            {"F090W": _spec(), "F150W": _spec(), "F444W": _spec()},
            z=6.1,
            center=CENTER,
        )
        out = img.select(["F444W", "F090W"])
        assert out.filters == ["F444W", "F090W"]
        assert out.z == img.z and out.center == img.center

    def test_select_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one band"):
            _image().select([])

    def test_select_duplicate_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            _image().select(["F150W", "F150W"])

    def test_select_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown band"):
            _image().select(["F999W"])


class TestMerge:
    """Same-target band union assembly."""

    def test_merge_unions_bands_in_order(self) -> None:
        a = NoobImage.from_bands({"F150W": _spec()}, z=6.1, center=CENTER)
        b = NoobImage.from_bands({"F444W": _spec()}, z=6.1, center=CENTER)
        out = a.merge(b)
        assert out.filters == ["F150W", "F444W"]

    def test_merge_duplicate_band_raises(self) -> None:
        a = NoobImage.from_bands({"F150W": _spec()}, z=6.1, center=CENTER)
        b = NoobImage.from_bands({"F150W": _spec()}, z=6.1, center=CENTER)
        with pytest.raises(ValueError, match="duplicate band"):
            a.merge(b)

    def test_merge_mismatched_z_raises(self) -> None:
        a = NoobImage.from_bands({"F150W": _spec()}, z=6.1, center=CENTER)
        b = NoobImage.from_bands({"F444W": _spec()}, z=5.0, center=CENTER)
        with pytest.raises(ValueError, match="z differs"):
            a.merge(b)

    def test_merge_mismatched_center_raises(self) -> None:
        a = NoobImage.from_bands({"F150W": _spec()}, z=6.1, center=CENTER)
        b = NoobImage.from_bands({"F444W": _spec()}, z=6.1, center=(150.01, 2.0))
        with pytest.raises(ValueError, match="arcsec apart"):
            a.merge(b)

    def test_merge_within_center_tolerance(self) -> None:
        a = NoobImage.from_bands({"F150W": _spec()}, z=6.1, center=CENTER)
        # ~0.36 arcsec east -> under the 0.5 arcsec default tolerance.
        b = NoobImage.from_bands({"F444W": _spec()}, z=6.1, center=(150.0001, 2.0))
        assert a.merge(b).filters == ["F150W", "F444W"]
