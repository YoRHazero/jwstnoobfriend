"""Tests for the morphology auto-mask and PSF resolution."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.inference.morphology._mask import auto_mask_band
from noobfriend.inference.morphology._psf import resolve_psf


def _two_blobs() -> np.ndarray:
    shape = (40, 40)
    ys, xs = np.indices(shape)
    target = 100.0 * np.exp(-((xs - 20) ** 2 + (ys - 20) ** 2) / (2 * 2.0**2))
    neighbour = 80.0 * np.exp(-((xs - 32) ** 2 + (ys - 32) ** 2) / (2 * 2.0**2))
    return target + neighbour


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
