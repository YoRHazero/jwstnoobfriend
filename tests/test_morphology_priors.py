"""Tests for the high-z default-prior policy and image seeding."""

from __future__ import annotations

import numpy as np
import pytest

from noobfriend.extraction._wcs import tangent_plane_wcs
from noobfriend.inference.morphology import NoobImage
from noobfriend.inference.morphology._geometry import build_geometry
from noobfriend.inference.morphology._param import (
    LogUniform,
    Normal,
    TruncNormal,
    Uniform,
)
from noobfriend.inference.morphology._priors import (
    HighzPriorPolicy,
    ImageSeeds,
    compute_seeds,
)

CENTER = (150.0, 2.0)


def _blob(shape: tuple[int, int] = (21, 21), amp: float = 100.0) -> np.ndarray:
    rows, cols = shape
    ys, xs = np.indices(shape)
    yc, xc = (rows - 1) / 2.0, (cols - 1) / 2.0
    return amp * np.exp(-((xs - xc) ** 2 + (ys - yc) ** 2) / (2 * 2.0**2))


def _seeds() -> ImageSeeds:
    shape = (21, 21)
    data = _blob(shape)
    img = NoobImage.from_bands(
        {
            "F150W": {
                "data": data,
                "error": np.full(shape, 1.0),
                "wcs": tangent_plane_wcs(*CENTER, 4.2, shape),
            }
        },
        z=6.1,
        center=CENTER,
    )
    geoms = {
        "F150W": build_geometry(img.bands["F150W"].wcs, CENTER, shape, 2),
    }
    return compute_seeds(img.bands, geoms)


class TestComputeSeeds:
    """Seeding centroid, size, and flux from the light."""

    def test_centred_blob_has_zero_offset(self) -> None:
        seeds = _seeds()
        assert abs(seeds.e0) < 0.1 and abs(seeds.n0) < 0.1

    def test_flux_and_size_positive(self) -> None:
        seeds = _seeds()
        assert seeds.flux["F150W"] > 0
        assert seeds.r_eff > seeds.pixscale * 0.5
        assert seeds.pixscale == pytest.approx(0.2, rel=1e-2)


class TestHighzPolicy:
    """The high-z default priors per axis."""

    def test_axes_map_to_expected_priors(self) -> None:
        seeds = _seeds()
        pol = HighzPriorPolicy()
        assert isinstance(pol.prior_for("x0", seeds), Normal)
        assert isinstance(pol.prior_for("r_eff", seeds), TruncNormal)
        assert pol.prior_for("n", seeds) == TruncNormal(1.5, 1.0, 0.3, 8.0)
        assert pol.prior_for("q", seeds) == Uniform(0.1, 1.0)
        assert pol.prior_for("theta", seeds) == Uniform(0.0, 180.0)

    def test_flux_needs_band(self) -> None:
        seeds = _seeds()
        with pytest.raises(ValueError, match="needs a band"):
            HighzPriorPolicy().prior_for("flux", seeds)

    def test_flux_is_loguniform_around_seed(self) -> None:
        seeds = _seeds()
        prior = HighzPriorPolicy().prior_for("flux", seeds, band="F150W")
        assert isinstance(prior, LogUniform)
        assert prior.low < seeds.flux["F150W"] < prior.high

    def test_unknown_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="no default prior"):
            HighzPriorPolicy().prior_for("bogus", _seeds())
