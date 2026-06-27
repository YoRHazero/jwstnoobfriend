"""Tests for the morphology auto-mask and PSF resolution."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.extraction._wcs import tangent_plane_wcs
from noobfriend.inference.morphology import NoobImage, Sersic
from noobfriend.inference.morphology._mask import auto_mask_band
from noobfriend.inference.morphology._psf import resolve_psf

CENTER = (150.0, 2.0)


def _two_blobs() -> np.ndarray:
    shape = (40, 40)
    ys, xs = np.indices(shape)
    target = 100.0 * np.exp(-((xs - 20) ** 2 + (ys - 20) ** 2) / (2 * 2.0**2))
    neighbour = 80.0 * np.exp(-((xs - 32) ** 2 + (ys - 32) ** 2) / (2 * 2.0**2))
    return target + neighbour


def _compact_blob(
    shape: tuple[int, int], x0: float, y0: float, *, amp: float = 100.0
) -> np.ndarray:
    """Return a compact positive source with zero-valued far wings."""
    ys, xs = np.indices(shape)
    blob = amp * np.exp(-((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * 1.4**2))
    blob[blob < 1e-2] = 0.0
    return blob


def _delta_psf() -> np.ndarray:
    """Small normalised delta PSF for model-build tests."""
    psf = np.zeros((5, 5), dtype=float)
    psf[2, 2] = 1.0
    return psf


class TestAutoMask:
    """Neighbour exclusion from per-band segmentation."""

    def test_masks_neighbour_keeps_target(self) -> None:
        mask = auto_mask_band(_two_blobs(), target_xy=(20.0, 20.0), dilate=1)
        assert mask[32, 32]  # neighbour excluded
        assert not mask[20, 20]  # target kept
        assert not mask[0, 0]  # sky kept

    def test_undetected_target_masks_nothing(self) -> None:
        # Target position on blank sky -> do not risk masking the (undetected) target.
        mask = auto_mask_band(_two_blobs(), target_xy=(5.0, 5.0))
        assert not mask.any()


class TestModelAutoMask:
    """Model-level auto-mask composition across bands."""

    def test_joint_mask_projects_deep_band_neighbour_to_shallow_band(self) -> None:
        shape = (61, 61)
        wcs = tangent_plane_wcs(*CENTER, 3.05, shape)  # 0.05 arcsec / pixel
        target = _compact_blob(shape, 30.0, 30.0)
        neighbour = _compact_blob(shape, 42.0, 30.0, amp=80.0)
        psf = _delta_psf()
        img = NoobImage.from_bands(
            {
                "F150W": {
                    "data": target + neighbour,
                    "error": np.ones(shape),
                    "wcs": wcs,
                    "psf": psf,
                },
                "F444W": {
                    "data": target,
                    "error": np.ones(shape),
                    "wcs": wcs,
                    "psf": psf,
                },
            },
            z=6.1,
            center=CENTER,
        )

        model = img.model(
            [Sersic("gal")],
            oversample=1,
            auto_mask=True,
            fit_radius_arcsec=1.0,
        )
        use = model.preview().use_mask["F444W"]

        assert bool(use[30, 30])  # protected target core
        assert not bool(use[30, 42])  # neighbour found in F150W and projected

    def test_auto_mask_default_fit_radius_excludes_far_sky(self) -> None:
        shape = (41, 41)
        wcs = tangent_plane_wcs(*CENTER, 4.1, shape)  # 0.1 arcsec / pixel
        target = _compact_blob(shape, 20.0, 20.0)
        img = NoobImage.from_bands(
            {
                "F150W": {
                    "data": target,
                    "error": np.ones(shape),
                    "wcs": wcs,
                    "psf": _delta_psf(),
                }
            },
            z=6.1,
            center=CENTER,
        )

        use = (
            img.model([Sersic("gal")], oversample=1, auto_mask=True).preview().use_mask
        )

        assert bool(use["F150W"][20, 20])
        assert not bool(use["F150W"][0, 0])


class TestResolvePsf:
    """PSF normalisation and the STPSF fallback guard."""

    def test_provided_psf_is_normalised(self) -> None:
        psf = resolve_psf(np.full((5, 5), 4.0), band="F150W", oversample=3)
        assert np.isclose(psf.sum(), 1.0)

    def test_nonpositive_psf_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive finite sum"):
            resolve_psf(np.zeros((5, 5)), band="F150W", oversample=3)

    def test_missing_psf_without_stpsf_raises(self) -> None:
        import importlib.util

        if importlib.util.find_spec("stpsf") is not None:
            pytest.skip("stpsf installed; the missing-PSF guard is not exercised")
        with pytest.raises(ValueError, match="STPSF"):
            resolve_psf(None, band="F150W", oversample=3)
