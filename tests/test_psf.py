"""Tests for the PSF source extractor: detection, sky-matching, and cutouts.

Synthetic frames + a linear fake WCS only -- no FITS / example_data.
"""

import collections
import re

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from noobfriend.extraction.psf import SourceExtractor


class _LinearWCS:
    """A minimal APE-14 WCS: pixel (x, y) -> sky via a linear scale."""

    def __init__(self, ra0: float = 53.0, dec0: float = -27.0, scale: float = 1e-4):
        self.ra0, self.dec0, self.scale = ra0, dec0, scale

    def pixel_to_world(self, x: np.ndarray, y: np.ndarray) -> SkyCoord:
        return SkyCoord(
            (self.ra0 + self.scale * np.asarray(x)) * u.deg,
            (self.dec0 + self.scale * np.asarray(y)) * u.deg,
        )


def _frame(
    positions: list[tuple],
    *,
    peak: float = 300.0,
    sigma: float = 2.0,
    shape: tuple[int, int] = (120, 120),
    noise: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic frame; each position is ``(y, x)`` or ``(y, x, sigma_x)``."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, noise, size=shape)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for pos in positions:
        y, x = pos[0], pos[1]
        sx = pos[2] if len(pos) > 2 else sigma
        img += peak * np.exp(-0.5 * (((yy - y) / sigma) ** 2 + ((xx - x) / sx) ** 2))
    return img


class TestSourceExtractor:
    """Accumulating matched detections and cutouts across frames."""

    def test_matches_same_source_across_frames(self) -> None:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=21, nsigma=5.0)
        ext.add_from_frame(_frame([(40, 40), (80, 80)], seed=1), wcs=wcs, filename="f1")
        assert ext.n_sources == 2
        assert len(ext) == 2

        ext.add_from_frame(_frame([(40, 40), (80, 80)], seed=2), wcs=wcs, filename="f2")

        assert ext.n_sources == 2  # same physical sources, not new ones
        assert len(ext) == 4  # two detections each
        counts = collections.Counter(ext.index.tolist())
        assert sorted(counts.values()) == [2, 2]

    def test_new_source_gets_new_index(self) -> None:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.add_from_frame(_frame([(40, 40)], seed=1), wcs=wcs)
        ext.add_from_frame(_frame([(40, 40), (80, 80)], seed=2), wcs=wcs)
        assert ext.n_sources == 2

    def test_drops_unmatched_high_ellipticity(self) -> None:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        # second source is elongated (sigma_x = 6) -> ellipticity > 0.1, no match
        ext.add_from_frame(
            _frame([(40, 40), (80, 80, 6.0)], seed=1), wcs=wcs, max_ellipticity=0.1
        )
        assert ext.n_sources == 1
        assert ext.catalog.x[0] == pytest.approx(40.0, abs=1.5)

    def test_keeps_matched_high_ellipticity_as_evidence(self) -> None:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.add_from_frame(_frame([(60, 60)], seed=1), wcs=wcs)  # round -> kept
        assert ext.n_sources == 1
        # same position, now elongated: matches, so kept despite high ellipticity
        ext.add_from_frame(
            _frame([(60, 60, 6.0)], seed=2), wcs=wcs, max_ellipticity=0.1
        )

        assert ext.n_sources == 1  # matched, not a new source
        assert len(ext) == 2  # both detections retained
        assert ext.catalog.ellipticity.max() > 0.3  # the elongated one kept as evidence

    def test_cutout_shape_and_dtype(self) -> None:
        wcs = _LinearWCS()
        frame = _frame([(60, 60)], seed=1)
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.add_from_frame(
            frame, np.ones_like(frame), np.zeros_like(frame, dtype=np.int32), wcs=wcs
        )
        cut = ext.cutouts[0]
        assert cut.data.shape == (21, 21)
        assert cut.data.dtype == np.float32
        assert cut.err.shape == (21, 21)
        assert cut.bad.shape == (21, 21) and cut.bad.dtype == bool

    def test_dq_marked_in_cutout(self) -> None:
        wcs = _LinearWCS()
        frame = _frame([(60, 60)], seed=1)
        dq = np.zeros_like(frame, dtype=np.int32)
        dq[55, 55] = 1  # off-centre bad pixel, inside the cutout
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.add_from_frame(frame, np.ones_like(frame), dq, wcs=wcs)
        assert ext.cutouts[0].bad.any()  # recorded, but the source was still detected
        assert len(ext) == 1

    def test_mask_excludes_from_detection(self) -> None:
        wcs = _LinearWCS()
        frame = _frame([(40, 40), (80, 80)], seed=1)
        mask = np.zeros_like(frame, dtype=bool)
        mask[25:56, 25:56] = True  # over the first source (unlike dq, this excludes it)
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.add_from_frame(frame, np.ones_like(frame), wcs=wcs, mask=mask)
        assert ext.n_sources == 1
        assert ext.catalog.x[0] == pytest.approx(80.0, abs=1.5)

    def test_edge_source_cutout_dropped(self) -> None:
        wcs = _LinearWCS()
        # source at y=9 is detected but its size-21 cutout (half 10) runs off the top
        frame = _frame([(60, 60), (9, 60)], seed=1)
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.add_from_frame(frame, np.ones_like(frame), wcs=wcs)
        assert len(ext) == 1
        assert ext.catalog.y[0] == pytest.approx(60.0, abs=1.5)

    def test_returns_kept_frame_catalogue(self) -> None:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        kept = ext.add_from_frame(_frame([(40, 40), (80, 80)], seed=1), wcs=wcs)
        assert len(kept) == 2

    def test_invalid_cutout_size_raises(self) -> None:
        with pytest.raises(ValueError, match="cutout_size must be positive"):
            SourceExtractor(fwhm=4.0, cutout_size=0)


class TestSelect:
    """Carving out one (filter, module) group of point-like sources."""

    def test_filter_restricts_to_one_band(self) -> None:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.add_from_frame(_frame([(60, 60)], seed=1), wcs=wcs, filter="F210M")
        ext.add_from_frame(_frame([(60, 60)], seed=2), wcs=wcs, filter="F182M")
        assert ext.n_sources == 1 and len(ext) == 2  # same source, two bands

        sub = ext.select(filter="F210M")
        assert len(sub) == 1 and sub.n_sources == 1

    def test_module_restricts(self) -> None:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.add_from_frame(_frame([(40, 40)], seed=1), wcs=wcs, detector="nrca1")
        ext.add_from_frame(_frame([(80, 80)], seed=2), wcs=wcs, detector="nrcb1")
        assert ext.n_sources == 2
        assert len(ext.select(module="a")) == 1
        assert len(ext.select(module="b")) == 1

    def test_label_matchers_accept_globs_and_regex(self) -> None:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.add_from_frame(
            _frame([(35, 35)], seed=1), wcs=wcs, filter="F210M", detector="nrca1"
        )
        ext.add_from_frame(
            _frame([(60, 60)], seed=2), wcs=wcs, filter="F182M", detector="nrca2"
        )
        ext.add_from_frame(
            _frame([(90, 90)], seed=3), wcs=wcs, filter="F210M", detector="nrcb1"
        )

        assert len(ext.select(filter="F2*")) == 2
        assert len(ext.select(detector="nrca*")) == 2
        assert len(ext.select(detector=re.compile(r"nrc[ab]1"))) == 2
        assert len(ext.select(module="[ab]")) == 3
        assert len(ext.select(module=re.compile(r"a|b"))) == 3

    def test_max_ellipticity_is_cross_band(self) -> None:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        # source A round in F182M but elongated in F210M; source B round in both
        ext.add_from_frame(
            _frame([(60, 60), (60, 100)], seed=1), wcs=wcs, filter="F182M"
        )
        ext.add_from_frame(
            _frame([(60, 60, 6.0), (60, 100)], seed=2), wcs=wcs, filter="F210M"
        )
        assert ext.n_sources == 2

        sub = ext.select(max_ellipticity=0.1)

        assert sub.n_sources == 1  # A dropped: its F210M ellipticity is high
        assert np.allclose(sub.catalog.x, 100.0, atol=2.0)  # only B survives

    def test_include_and_exclude_partition(self) -> None:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.add_from_frame(_frame([(40, 40), (80, 80)], seed=1), wcs=wcs)
        assert ext.n_sources == 2

        included = ext.select(include=[0])
        excluded = ext.select(exclude=[0])
        assert included.n_sources == 1 and excluded.n_sources == 1
        assert included.catalog.x[0] != excluded.catalog.x[0]


class TestBuild:
    """Driving build_epsf / build_extended_psf from cached cutouts."""

    def _filled(self, cutout_size: int) -> SourceExtractor:
        wcs = _LinearWCS()
        ext = SourceExtractor(fwhm=4.0, cutout_size=cutout_size, nsigma=5.0)
        positions = [(70, 70), (70, 150), (150, 70), (150, 150)]
        for k in range(6):
            oy = (0.37 * k) % 1 - 0.5
            ox = (0.61 * k) % 1 - 0.5
            frame = _frame(
                [(y + oy, x + ox) for y, x in positions], shape=(220, 220), seed=k
            )
            ext.add_from_frame(
                frame, np.ones_like(frame), wcs=wcs, filter="F210M", detector="nrca1"
            )
        return ext.select(filter="F210M", max_ellipticity=0.3)

    def test_core_shape_and_gauge(self) -> None:
        core = self._filled(29).build_psf_core(oversample=3, core_size=21)
        assert core.core.shape == (63, 63)
        assert core.core.sum() == pytest.approx(9.0, rel=1e-3)  # unit-volume gauge os^2
        assert core.n_stars > 0
        assert core.phase_grid.shape == (3, 3)

    def test_wings_stitch_onto_core(self) -> None:
        sub = self._filled(55)
        core = sub.build_psf_core(oversample=3, core_size=21)
        psf = sub.build_psf_wings(core, wing_size=51)
        assert psf.wing.shape == (51, 51)
        assert psf.core_plane.shape == (63, 63)
        assert psf.oversample == 3
        assert psf.n_wing_stars > 0

    def test_cutout_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="smaller than core_size"):
            self._filled(15).build_psf_core(core_size=21)

    def test_even_oversample_raises(self) -> None:
        with pytest.raises(ValueError, match="positive odd integer"):
            self._filled(29).build_psf_core(oversample=4)

    def test_spilled_extractor_still_selects_and_builds(self, tmp_path) -> None:
        wcs = _LinearWCS()
        positions = [(70, 70), (70, 150), (150, 70), (150, 150)]
        with SourceExtractor(
            fwhm=4.0, cutout_size=29, nsigma=5.0, max_in_memory=3, spill_dir=tmp_path
        ) as ext:
            for k in range(4):
                oy = (0.37 * k) % 1 - 0.5
                ox = (0.61 * k) % 1 - 0.5
                frame = _frame(
                    [(y + oy, x + ox) for y, x in positions], shape=(220, 220), seed=k
                )
                ext.add_from_frame(
                    frame,
                    np.ones_like(frame),
                    wcs=wcs,
                    filter="F210M",
                    detector="nrca1",
                )
            assert len(ext) > 3  # exceeded max_in_memory -> spilled to disk
            assert ext.cutouts[0].data.shape == (29, 29)  # readable back from disk

            sub = ext.select(filter="F210M", max_ellipticity=0.3)
            core = sub.build_psf_core(oversample=3, core_size=21)
            assert core.core.shape == (63, 63)
            sub.close()
