"""Tests for SourceExtractor.save / load round-trip.

Synthetic frames + a linear fake WCS only -- no FITS / example_data. Verifies the
detection table, per-detection labels (including ``None``), and cutouts round-trip
exactly, that pre-filtering with ``select`` shrinks what is saved, that a loaded
cache is read lazily and never deleted, and the edge cases (empty, overwrite,
missing manifest, spilled source).
"""

import numpy as np
import pytest

from noobfriend.extraction.psf import Cutout, SourceExtractor

from ._helpers import LinearWCS, make_frame


def _cutouts_equal(a: Cutout, b: Cutout) -> bool:
    """Whether two cutouts match in data, err, bad, and origin."""
    if a.origin != b.origin or not np.allclose(a.data, b.data):
        return False
    for left, right in ((a.err, b.err), (a.bad, b.bad)):
        if (left is None) != (right is None):
            return False
        if left is not None and not np.array_equal(left, right):
            return False
    return True


def _two_source_extractor(cutout_size: int = 21, **store_kwargs) -> SourceExtractor:
    """Two physical sources across two frames, with band / detector labels."""
    wcs = LinearWCS()
    ext = SourceExtractor(fwhm=4.0, cutout_size=cutout_size, nsigma=5.0, **store_kwargs)
    frame = make_frame([(40, 40), (80, 80)], seed=1)
    err = np.ones_like(frame)
    dq = np.zeros_like(frame, dtype=np.int32)
    ext.add_from_frame(frame, err, dq, wcs=wcs, filter="F210M", detector="nrca1")
    frame2 = make_frame([(40, 40), (80, 80)], seed=2)
    ext.add_from_frame(frame2, np.ones_like(frame2), wcs=wcs, filter="F182M")
    return ext


def _assert_round_trips(original: SourceExtractor, loaded: SourceExtractor) -> None:
    """Assert two extractors carry the same detections, labels, and cutouts."""
    assert len(loaded) == len(original)
    assert loaded.n_sources == original.n_sources
    assert np.array_equal(loaded.index, original.index)
    assert list(loaded._filter) == list(original._filter)
    assert list(loaded._detector) == list(original._detector)
    assert list(loaded.filename) == list(original.filename)
    for name in ("x", "y", "ra", "dec", "fwhm", "ellipticity", "flux", "snr"):
        assert np.allclose(
            getattr(loaded.catalog, name),
            getattr(original.catalog, name),
            equal_nan=True,
        )
    for i in range(len(original)):
        assert _cutouts_equal(loaded.cutouts[i], original.cutouts[i])


class TestRoundTrip:
    """The catalogue, labels, and cutouts survive a save / load."""

    def test_basic_round_trip(self, tmp_path) -> None:
        ext = _two_source_extractor()
        ext.save(tmp_path / "cache")
        loaded = SourceExtractor.load(tmp_path / "cache")
        _assert_round_trips(ext, loaded)

    def test_none_labels_round_trip(self, tmp_path) -> None:
        ext = _two_source_extractor()  # second frame has detector=None
        assert None in set(ext._detector)
        ext.save(tmp_path / "cache")
        loaded = SourceExtractor.load(tmp_path / "cache")
        assert list(loaded._detector) == list(ext._detector)  # None preserved

    def test_config_preserved(self, tmp_path) -> None:
        ext = SourceExtractor(fwhm=3.5, cutout_size=21, nsigma=4.0, match_radius=0.2)
        LinearWCS()
        ext.add_from_frame(make_frame([(60, 60)], seed=1), wcs=LinearWCS())
        ext.save(tmp_path / "cache")
        loaded = SourceExtractor.load(tmp_path / "cache")
        assert loaded.fwhm == 3.5
        assert loaded.nsigma == 4.0
        assert loaded.match_radius == 0.2

    def test_spilled_source_saves_all(self, tmp_path) -> None:
        # source store itself spills to disk; save must stream every chunk
        with _two_source_extractor(
            max_in_memory=1, spill_dir=tmp_path / "spill"
        ) as ext:
            n = len(ext)
            ext.save(tmp_path / "cache")
        loaded = SourceExtractor.load(tmp_path / "cache")
        assert len(loaded) == n


class TestSelectThenSave:
    """Pre-filtering with select carves down what is persisted."""

    def test_select_shrinks_saved_set(self, tmp_path) -> None:
        ext = _two_source_extractor()
        sub = ext.select(filter="F210M")
        assert len(sub) == 2 and len(sub) < len(ext)
        sub.save(tmp_path / "cache")
        loaded = SourceExtractor.load(tmp_path / "cache")
        _assert_round_trips(sub, loaded)
        assert set(loaded._filter) == {"F210M"}


class TestLoadedCacheIsSafe:
    """A loaded cache is read lazily and never deleted."""

    def test_close_does_not_delete_saved_dir(self, tmp_path) -> None:
        ext = _two_source_extractor()
        ext.save(tmp_path / "cache")
        loaded = SourceExtractor.load(tmp_path / "cache")
        loaded.close()  # must NOT remove the on-disk cache
        assert (tmp_path / "cache" / "meta.json").is_file()
        assert (tmp_path / "cache" / "catalog.fits").is_file()
        assert list((tmp_path / "cache" / "cutouts").glob("chunk_*.npz"))
        # the cache is still loadable after the first handle closed
        again = SourceExtractor.load(tmp_path / "cache")
        assert len(again) == len(ext)

    def test_select_after_load_streams_and_builds(self, tmp_path) -> None:
        wcs = LinearWCS()
        positions = [(70, 70), (70, 150), (150, 70), (150, 150)]
        src = SourceExtractor(fwhm=4.0, cutout_size=29, nsigma=5.0)
        for k in range(6):
            oy, ox = (0.37 * k) % 1 - 0.5, (0.61 * k) % 1 - 0.5
            frame = make_frame(
                [(y + oy, x + ox) for y, x in positions], shape=(220, 220), seed=k
            )
            src.add_from_frame(
                frame, np.ones_like(frame), wcs=wcs, filter="F210M", detector="nrca1"
            )
        src.save(tmp_path / "cache")

        loaded = SourceExtractor.load(tmp_path / "cache")
        sub = loaded.select(filter="F210M", max_ellipticity=0.3)
        core = sub.build_psf_core(oversample=3, core_size=21)
        assert core.core.shape == (63, 63)
        assert core.n_stars > 0


class TestRebuiltPositions:
    """Per-source sky positions are recomputed so matching still works."""

    def test_match_after_load(self, tmp_path) -> None:
        ext = _two_source_extractor()
        ext.save(tmp_path / "cache")
        loaded = SourceExtractor.load(tmp_path / "cache")
        before = loaded.n_sources
        # a new detection at an existing source's position must match, not add
        loaded.add_from_frame(make_frame([(40, 40)], seed=9), wcs=LinearWCS())
        assert loaded.n_sources == before


class TestEdgeCases:
    """Empty extractor, overwrite, and a missing manifest."""

    def test_empty_round_trip(self, tmp_path) -> None:
        ext = SourceExtractor(fwhm=4.0, cutout_size=21)
        ext.save(tmp_path / "cache")
        loaded = SourceExtractor.load(tmp_path / "cache")
        assert len(loaded) == 0 and loaded.n_sources == 0

    def test_overwrite_required(self, tmp_path) -> None:
        ext = _two_source_extractor()
        ext.save(tmp_path / "cache")
        with pytest.raises(FileExistsError):
            ext.save(tmp_path / "cache")
        ext.save(tmp_path / "cache", overwrite=True)  # succeeds
        assert len(SourceExtractor.load(tmp_path / "cache")) == len(ext)

    def test_missing_manifest_raises(self, tmp_path) -> None:
        (tmp_path / "cache").mkdir()
        with pytest.raises(FileNotFoundError, match="meta.json"):
            SourceExtractor.load(tmp_path / "cache")
