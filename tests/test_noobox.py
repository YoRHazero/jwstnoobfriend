"""Tests for NooBox discovery, subsetting, merge lineage and persistence.

No FITS file is ever opened: books are built thin (``NooBook.from_name``, which
only needs file names to exist for globbing) or constructed directly with the
identifier fields, so the logic under test never touches real data.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from noobfriend.navigation import Footprint, NooBook, NooBox


def _touch(directory: Path, name: str) -> None:
    (directory / name).write_text("")


def _exposure_book(
    stage: str,
    suffix: str,
    *,
    exposure: str = "00001",
    detector: str = "nrca1",
    program: str = "01345",
    parent_ids: tuple[str, ...] = (),
    filter: str | None = None,
    footprint: Footprint | None = None,
) -> NooBook:
    """Build a single-exposure NooBook with explicit identifier fields."""
    stem = f"jw{program}001001_02201_{exposure}_{detector}_{suffix}"
    return NooBook(
        id=f"{stem}@{stage}",
        location=f"/{stem}.fits",
        stage=stage,
        program_id=program,
        observation=("001",),
        visit=("001",),
        ggsaa=("02201",),
        exposure=(exposure,),
        detector=detector,
        parent_ids=parent_ids,
        filter=filter,
        footprint=footprint,
    )


def _stage3_book(
    parent_ids: tuple[str, ...],
    *,
    exposure: tuple[str, ...] = ("00001", "00002"),
    program: str = "01345",
) -> NooBook:
    """Build an aggregate stage-3 NooBook with explicit parent ids."""
    stem = f"jw{program}_tile_i2d"
    return NooBook(
        id=f"{stem}@3a",
        location=f"/{stem}.fits",
        stage="3a",
        program_id=program,
        observation=("001",),
        visit=("001",),
        ggsaa=("02201",),
        exposure=exposure,
        detector=None,
        parent_ids=parent_ids,
    )


def _footprint(offset: float = 0.0) -> Footprint:
    return Footprint.from_corners(
        [
            (offset, 0.0),
            (offset + 1.0, 0.0),
            (offset + 1.0, 1.0),
            (offset, 1.0),
        ]
    )


def _box(*books: NooBook) -> NooBox:
    box = NooBox()
    for book in books:
        box.add(book)
    return box


def _patch_book_pixels(monkeypatch, *, wcs_by_id: dict[str, object | None]) -> None:
    data_by_id = {
        book_id: np.full((5, 5), float(i + 1)) for i, book_id in enumerate(wcs_by_id)
    }
    err_by_id = {book_id: np.ones((5, 5)) for book_id in wcs_by_id}
    dq_by_id = {book_id: np.zeros((5, 5), dtype=np.int32) for book_id in wcs_by_id}

    monkeypatch.setattr(NooBook, "wcs", property(lambda self: wcs_by_id[self.id]))
    monkeypatch.setattr(NooBook, "data", property(lambda self: data_by_id[self.id]))
    monkeypatch.setattr(NooBook, "err", property(lambda self: err_by_id[self.id]))
    monkeypatch.setattr(NooBook, "dq", property(lambda self: dq_by_id[self.id]))


def _capture_source_extractor(monkeypatch):
    created = []

    class FakeSourceExtractor:
        """Capture SourceExtractor constructor and frame-ingest calls."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls = []
            created.append(self)

        def add_from_frame(
            self,
            data,
            err=None,
            dq=None,
            *,
            wcs,
            mask=None,
            filename=None,
            filter=None,
            detector=None,
            max_ellipticity=0.1,
        ):
            self.calls.append(
                {
                    "data": data,
                    "err": err,
                    "dq": dq,
                    "wcs": wcs,
                    "mask": mask,
                    "filename": filename,
                    "filter": filter,
                    "detector": detector,
                    "max_ellipticity": max_ellipticity,
                }
            )

    monkeypatch.setattr(
        "noobfriend.extraction.psf.SourceExtractor", FakeSourceExtractor
    )
    return created


def _patch_fake_cutout(monkeypatch):
    """Replace extraction.cutout.Cutout with a lightweight WCS-free stand-in."""

    class FakeCutout:
        def __init__(self, wcs):
            self.wcs = wcs

        @classmethod
        def from_world(cls, wcs, ra, dec, **size):
            return cls(wcs)

        def array(self, data):
            return np.asarray(data, dtype=float) * 10.0

        def reproject(self, data, wcs):
            image = np.asarray(data, dtype=float)
            ones = np.ones_like(image)
            return image, ones, ones

    monkeypatch.setattr("noobfriend.extraction.cutout.Cutout", FakeCutout)
    return FakeCutout


# -- discovery ----------------------------------------------------------------


def test_from_directory_thin_indexes_matching_files(tmp_path):
    _touch(tmp_path, "jw01345001001_02201_00001_nrca1_cal.fits")
    _touch(tmp_path, "jw01345001001_02201_00002_nrca1_cal.fits")
    _touch(tmp_path, "notes.txt")

    box = NooBox.from_directory("2a", root=str(tmp_path))

    assert len(box) == 2
    assert all(book.stage == "2a" for book in box)
    assert all(not book.is_probed for book in box)
    assert "jw01345001001_02201_00001_nrca1_cal@2a" in box


def test_from_directory_custom_wildcard_filters(tmp_path):
    _touch(tmp_path, "jw01345001001_02201_00001_nrca1_cal.fits")
    _touch(tmp_path, "jw01345001001_02201_00001_nrca1_rate.fits")

    box = NooBox.from_directory("2a", root=str(tmp_path), wildcard="*_cal.fits")

    assert [book.exposure for book in box] == [("00001",)]
    assert next(iter(box)).id.endswith("_cal@2a")


def test_from_directory_unresolved_stage_raises(monkeypatch):
    monkeypatch.delenv("STAGE_ZZ_PATH", raising=False)
    with pytest.raises(ValueError, match="no root for stage"):
        NooBox.from_directory("zz")


def test_from_directory_remote_root_thin(monkeypatch):
    def fake_list_remote_dir(spec):
        assert spec == "host:/data/stage2"
        return ["jw01345001001_02201_00001_nrca1_cal.fits", "notes.txt"]

    monkeypatch.setattr(
        "noobfriend.navigation.noobox._core.list_remote_dir", fake_list_remote_dir
    )

    box = NooBox.from_directory("2a", root="host:/data/stage2")

    assert len(box) == 1  # notes.txt excluded by the default *.fits wildcard
    book = next(iter(box))
    assert book.location == "host:/data/stage2/jw01345001001_02201_00001_nrca1_cal.fits"
    assert book.detector == "nrca1"
    assert not book.is_probed


def test_from_directory_composes_noob_server(monkeypatch):
    """With no root, a remote NOOB_SERVER prefixes STAGE_<STAGE>_PATH."""
    monkeypatch.setenv("STAGE_2A_PATH", "/disk/jwst/2a")
    monkeypatch.setattr(
        "noobfriend.navigation.noobox._core.get_settings",
        lambda: SimpleNamespace(server_host=lambda: "myhost"),
    )

    def fake_list_remote_dir(spec):
        assert spec == "myhost:/disk/jwst/2a"
        return ["jw01345001001_02201_00001_nrca1_cal.fits"]

    monkeypatch.setattr(
        "noobfriend.navigation.noobox._core.list_remote_dir", fake_list_remote_dir
    )

    box = NooBox.from_directory("2a")

    location = next(iter(box)).location
    assert location == "myhost:/disk/jwst/2a/jw01345001001_02201_00001_nrca1_cal.fits"


def test_from_directory_local_server_keeps_bare_path(tmp_path, monkeypatch):
    """A local NOOB_SERVER leaves STAGE_<STAGE>_PATH as a bare local path."""
    _touch(tmp_path, "jw01345001001_02201_00001_nrca1_cal.fits")
    monkeypatch.setenv("STAGE_2A_PATH", str(tmp_path))
    monkeypatch.setattr(
        "noobfriend.navigation.noobox._core.get_settings",
        lambda: SimpleNamespace(server_host=lambda: None),
    )

    box = NooBox.from_directory("2a")

    location = next(iter(box)).location
    assert location == str(tmp_path / "jw01345001001_02201_00001_nrca1_cal.fits")


# -- subsetting, grouping, selection ------------------------------------------


def test_filter_returns_sharing_subset(tmp_path):
    _touch(tmp_path, "jw01345001001_02201_00001_nrca1_cal.fits")
    _touch(tmp_path, "jw01345001001_02201_00001_nrcb1_cal.fits")
    box = NooBox.from_directory("2a", root=str(tmp_path))

    subset = box.filter(lambda book: book.detector == "nrca1")

    assert len(subset) == 1
    assert next(iter(subset)).detector == "nrca1"
    assert len(box) == 2  # original untouched
    assert subset._store is box._store  # shares the byte cache


def test_select_glob_membership_and_tuple_fields():
    box = _box(
        _exposure_book("2a", "cal", detector="nrca1"),
        _exposure_book("2a", "cal", detector="nrcb1"),
        _exposure_book("2a", "cal", detector="nrca3"),
    )

    assert {b.detector for b in box.select(detector="nrca*")} == {"nrca1", "nrca3"}
    assert {b.detector for b in box.select(detector=["nrcb1", "nrca3"])} == {
        "nrcb1",
        "nrca3",
    }
    assert len(box.select(exposure="00001")) == 3  # any element of the tuple field
    assert box.select(detector="nrca1")._store is box._store


def test_group_by_and_distinct():
    box = _box(
        _exposure_book("2a", "cal", detector="nrca1"),
        _exposure_book("2a", "cal", detector="nrcb1"),
    )

    groups = box.group_by("detector")
    assert set(groups) == {"nrca1", "nrcb1"}
    assert all(isinstance(g, NooBox) for g in groups.values())
    assert box.distinct("detector") == {"nrca1", "nrcb1"}
    assert box.distinct("stage") == {"2a"}


def test_copy_is_an_independent_sandbox():
    rate = _exposure_book("1b", "rate")
    cal = _exposure_book("2a", "cal")
    box = _box(cal)

    clone = box.copy()
    clone.merge(_box(rate), child="2a")  # prepend: stamps the clone's 2a parents

    assert clone[cal.id].parent_ids == (rate.id,)  # the edge lands on the copy
    assert box[cal.id].parent_ids == ()  # original book untouched
    assert clone._store is not box._store  # own byte cache
    assert len(clone) == 2


# -- collection extraction ------------------------------------------------------


def test_extract_sources_forwards_books_to_source_extractor(monkeypatch):
    book_a = _exposure_book(
        "2a", "cal", exposure="00001", detector="nrca1", filter="F210M"
    )
    book_b = _exposure_book(
        "2a", "cal", exposure="00002", detector="nrcb1", filter="F182M"
    )
    wcs_a, wcs_b = object(), object()
    _patch_book_pixels(monkeypatch, wcs_by_id={book_a.id: wcs_a, book_b.id: wcs_b})
    created = _capture_source_extractor(monkeypatch)

    extractor = _box(book_a, book_b).extract.sources(
        fwhm=4.0,
        cutout_size=55,
        nsigma=4.5,
        match_radius=0.2,
        max_in_memory=8,
        max_ellipticity=0.12,
        mask_fn=lambda _book, data, _err, _dq: data > 1.5,
        probe=False,
        progress=False,
    )

    assert extractor is created[0]
    assert extractor.kwargs == {
        "fwhm": 4.0,
        "cutout_size": 55,
        "nsigma": 4.5,
        "match_radius": 0.2,
        "aperture_radius": None,
        "min_distance": None,
        "dq_bad_bits": 1,
        "max_in_memory": 8,
        "spill_dir": None,
    }
    assert [call["filename"] for call in extractor.calls] == [book_a.id, book_b.id]
    assert [call["filter"] for call in extractor.calls] == ["F210M", "F182M"]
    assert [call["detector"] for call in extractor.calls] == ["nrca1", "nrcb1"]
    assert [call["wcs"] for call in extractor.calls] == [wcs_a, wcs_b]
    assert extractor.calls[0]["mask"].sum() == 0
    assert extractor.calls[1]["mask"].sum() == 25
    assert all(call["max_ellipticity"] == 0.12 for call in extractor.calls)


def test_extract_sources_probes_thin_books_for_filter_labels(monkeypatch):
    book = _exposure_book("2a", "cal", filter=None)
    _patch_book_pixels(monkeypatch, wcs_by_id={book.id: object()})
    created = _capture_source_extractor(monkeypatch)

    def fake_probe(self, *, store=None):
        return self.model_copy(update={"shape": (5, 5), "filter": "F210M"})

    monkeypatch.setattr(NooBook, "probe", fake_probe)

    extractor = _box(book).extract.sources(progress=False)

    assert extractor is created[0]
    assert extractor.kwargs["fwhm"] == 4.0
    assert extractor.kwargs["cutout_size"] == 55
    assert extractor.calls[0]["filter"] == "F210M"


def test_extract_sources_missing_wcs_raises_or_skips(monkeypatch):
    missing = _exposure_book("2a", "cal", exposure="00001")
    good = _exposure_book("2a", "cal", exposure="00002")
    _patch_book_pixels(monkeypatch, wcs_by_id={missing.id: None, good.id: object()})
    created = _capture_source_extractor(monkeypatch)

    with pytest.raises(ValueError, match="has no assigned WCS"):
        _box(missing).extract.sources(
            fwhm=4.0, cutout_size=21, probe=False, progress=False
        )

    extractor = _box(missing, good).extract.sources(
        fwhm=4.0,
        cutout_size=21,
        skip_missing_wcs=True,
        probe=False,
        progress=False,
    )

    assert extractor is created[-1]
    assert [call["filename"] for call in extractor.calls] == [good.id]


# -- footprint coverage -------------------------------------------------------


def test_footprint_contains_inside_and_outside():
    footprint = _footprint(0.0)  # the unit square [0, 1] x [0, 1]

    assert footprint.contains(0.5, 0.5)
    assert not footprint.contains(1.5, 0.5)  # east of the square
    assert not footprint.contains(0.5, 1.5)  # north of the square
    assert not footprint.contains(-0.1, 0.5)  # west of the square


# -- collection cutout --------------------------------------------------------


def test_extract_cutout_gates_by_footprint_native(monkeypatch):
    _patch_fake_cutout(monkeypatch)
    covering = _exposure_book("2a", "cal", exposure="00001", footprint=_footprint(0.0))
    far = _exposure_book("2a", "cal", exposure="00002", footprint=_footprint(5.0))
    _patch_book_pixels(monkeypatch, wcs_by_id={covering.id: object(), far.id: object()})

    result = _box(covering, far).extract.cutout(
        0.5, 0.5, half=2, reproject=False, probe=False, progress=False
    )

    assert result.books == [covering]  # the far footprint is excluded
    assert not result.reprojected
    assert result.grid is None
    np.testing.assert_array_equal(result[covering], np.full((5, 5), 10.0))


def test_extract_cutout_reproject_prefers_stage3_grid(monkeypatch):
    _patch_fake_cutout(monkeypatch)
    exposure = _exposure_book("2a", "cal", exposure="00001", footprint=_footprint(0.0))
    mosaic = _exposure_book("3a", "i2d", exposure="00001", footprint=_footprint(0.0))
    wcs_exposure, wcs_mosaic = object(), object()
    _patch_book_pixels(
        monkeypatch, wcs_by_id={exposure.id: wcs_exposure, mosaic.id: wcs_mosaic}
    )

    result = _box(exposure, mosaic).extract.cutout(
        0.5, 0.5, half=2, probe=False, progress=False
    )

    assert result.reprojected
    assert result.grid.wcs is wcs_mosaic  # grid anchored on the stage-3 mosaic
    assert set(result.books) == {exposure, mosaic}


def test_extract_cutout_no_coverage_raises(monkeypatch):
    _patch_fake_cutout(monkeypatch)
    far = _exposure_book("2a", "cal", footprint=_footprint(5.0))
    _patch_book_pixels(monkeypatch, wcs_by_id={far.id: object()})

    with pytest.raises(ValueError, match="no book in the box covers"):
        _box(far).extract.cutout(0.5, 0.5, half=2, probe=False, progress=False)


def test_extract_cutout_coadd_weighted_mean(monkeypatch):
    _patch_fake_cutout(monkeypatch)
    book_a = _exposure_book("2a", "cal", exposure="00001", footprint=_footprint(0.0))
    book_b = _exposure_book("2a", "cal", exposure="00002", footprint=_footprint(0.0))
    _patch_book_pixels(
        monkeypatch, wcs_by_id={book_a.id: object(), book_b.id: object()}
    )

    result = _box(book_a, book_b).extract.cutout(
        0.5, 0.5, half=2, probe=False, progress=False
    )

    # book_a stamps to 1.0, book_b to 2.0, equal weights -> coadd is 1.5.
    np.testing.assert_allclose(result.coadd(), np.full((5, 5), 1.5))


# -- merge: lineage resolution ------------------------------------------------


def test_merge_is_in_place_and_links_by_identity_key():
    rate = _exposure_book("1b", "rate")
    cal = _exposure_book("2a", "cal")
    box = _box(rate)

    result = box.merge(_box(cal))

    assert result is box  # in-place, returns self
    assert box[cal.id].parent_ids == (rate.id,)
    assert box[rate.id].parent_ids == ()  # genuine root stays empty


def test_merge_no_key_match_leaves_parents_empty():
    rate = _exposure_book("1b", "rate", exposure="00001")
    cal = _exposure_book("2a", "cal", exposure="00002")  # different exposure
    box = _box(rate)

    box.merge(_box(cal))

    assert box[cal.id].parent_ids == ()


def test_merge_never_links_same_stage():
    cal = _exposure_book("2a", "cal")
    crf = _exposure_book("2a", "crf")  # same identity key, same stage
    box = _box(cal)

    box.merge(_box(crf))

    assert box[crf.id].parent_ids == ()


def test_merge_keeps_already_stamped_parents():
    rate = _exposure_book("1b", "rate")
    cal = _exposure_book("2a", "cal", parent_ids=("stamped@99",))  # e.g. by reduction
    box = _box(rate)

    box.merge(_box(cal))

    assert box[cal.id].parent_ids == ("stamped@99",)  # not re-pointed


def test_merge_branch_with_explicit_parent():
    two_b = _exposure_book("2b", "cal")
    two_bi = _exposure_book("2bi", "cali")
    two_bii = _exposure_book("2bii", "calii")
    box = _box(two_b)

    box.merge(_box(two_bi))  # links to the leaf 2b
    box.merge(_box(two_bii), parent="2b")  # branch off 2b, not the current leaf

    assert box[two_bi.id].parent_ids == (two_b.id,)
    assert box[two_bii.id].parent_ids == (two_b.id,)


def test_merge_prepend_with_child():
    rate = _exposure_book("1b", "rate")
    cal = _exposure_book("2a", "cal")
    box = _box(cal)

    box.merge(_box(rate), child="2a")  # incoming 1b becomes parent of 2a

    assert box[cal.id].parent_ids == (rate.id,)
    assert box[rate.id].parent_ids == ()


def test_merge_parent_and_child_are_mutually_exclusive():
    with pytest.raises(ValueError, match="at most one"):
        NooBox().merge(NooBox(), parent="2a", child="1b")


def test_merge_overwrite_false_raises_on_duplicate():
    book = NooBook.from_name("/data/jw01345001001_02201_00001_nrca1_cal.fits", "2a")
    left, right = NooBox(), NooBox()
    left.add(book)
    right.add(book)

    with pytest.raises(ValueError, match="duplicate id"):
        left.merge(right)


def test_merge_overwrite_true_incoming_wins():
    left, right = NooBox(), NooBox()
    left.add(NooBook(id="x@2a", location="/old.fits", stage="2a", program_id="01345"))
    right.add(NooBook(id="x@2a", location="/new.fits", stage="2a", program_id="01345"))

    assert left.merge(right, overwrite=True) is left
    assert len(left) == 1
    assert left["x@2a"].location == "/new.fits"


# -- lineage boundary ---------------------------------------------------------


def test_leaves_and_roots():
    rate = _exposure_book("1b", "rate")
    cal = _exposure_book("2a", "cal")
    box = _box(rate).merge(_box(cal))

    assert {b.id for b in box.leaves()} == {cal.id}
    assert {b.id for b in box.roots()} == {rate.id}
    assert isinstance(box.leaves(), NooBox)
    assert box.leaves()._store is box._store


def test_family_returns_view_over_products_through_book():
    rate = _exposure_book("1b", "rate")
    cal = _exposure_book("2a", "cal")
    two_b = _exposure_book("2b", "cal")
    two_bi = _exposure_book("2bi", "cali")
    box = _box(rate).merge(_box(cal)).merge(_box(two_b)).merge(_box(two_bi))

    family = box.family(cal.id)

    assert {book.id for book in family} == {rate.id, cal.id, two_b.id, two_bi.id}
    assert family._store is box._store
    assert family[cal.id] is box[cal.id]


def test_family_keeps_sibling_branches_outside_seed_path():
    rate = _exposure_book("1b", "rate")
    cal = _exposure_book("2a", "cal")
    two_b = _exposure_book("2b", "cal")
    two_bi = _exposure_book("2bi", "cali")
    two_bii = _exposure_book("2bii", "calii")
    box = _box(rate).merge(_box(cal)).merge(_box(two_b))
    box.merge(_box(two_bi))
    box.merge(_box(two_bii), parent="2b")

    family = box.family(two_bi)

    assert {book.id for book in family} == {rate.id, cal.id, two_b.id, two_bi.id}
    assert two_bii.id not in family


def test_family_split_parents_returns_stage3_parent_branches():
    rate_a = _exposure_book("1b", "rate", exposure="00001")
    cal_a = _exposure_book("2a", "cal", exposure="00001", parent_ids=(rate_a.id,))
    two_bi_a = _exposure_book("2bi", "cali", exposure="00001", parent_ids=(cal_a.id,))
    rate_b = _exposure_book("1b", "rate", exposure="00002")
    cal_b = _exposure_book("2a", "cal", exposure="00002", parent_ids=(rate_b.id,))
    two_bi_b = _exposure_book("2bi", "cali", exposure="00002", parent_ids=(cal_b.id,))
    mosaic = _stage3_book((two_bi_a.id, two_bi_b.id))
    box = _box(rate_a, cal_a, two_bi_a, rate_b, cal_b, two_bi_b, mosaic)

    merged = box.family(mosaic)
    branches = box.family(mosaic, split_parents=True)

    assert {book.id for book in merged} == {
        rate_a.id,
        cal_a.id,
        two_bi_a.id,
        rate_b.id,
        cal_b.id,
        two_bi_b.id,
        mosaic.id,
    }
    assert len(branches) == 2
    assert [{book.id for book in branch} for branch in branches] == [
        {rate_a.id, cal_a.id, two_bi_a.id, mosaic.id},
        {rate_b.id, cal_b.id, two_bi_b.id, mosaic.id},
    ]
    assert all(branch._store is box._store for branch in branches)


def test_family_requires_parent_coverage():
    orphan = _exposure_book("2a", "cal", parent_ids=("missing@1b",))
    with pytest.raises(ValueError, match="parent_id .* is not present"):
        _box(orphan).family(orphan)

    present = _exposure_book("2bi", "cali")
    partial_mosaic = _stage3_book((present.id, "missing@2bi"))
    assert {
        book.id for book in _box(present, partial_mosaic).family(partial_mosaic)
    } == {
        present.id,
        partial_mosaic.id,
    }

    missing_mosaic = _stage3_book(("missing-a@2bi", "missing-b@2bi"))
    with pytest.raises(ValueError, match="none of its parent_ids"):
        _box(missing_mosaic).family(missing_mosaic)


def test_family_copy_returns_independent_box():
    rate = _exposure_book("1b", "rate")
    cal = _exposure_book("2a", "cal")
    box = _box(rate).merge(_box(cal))

    family = box.family(cal, copy=True)

    assert {book.id for book in family} == {rate.id, cal.id}
    assert family._store is not box._store
    assert family[cal.id] is not box[cal.id]


# -- introspection & persistence ----------------------------------------------


def test_to_table_summarises_members():
    box = _box(
        _exposure_book("1b", "rate"),
        _exposure_book("2a", "cal"),
    )

    table = box.to_table()

    assert len(table) == 2
    assert {"id", "stage", "detector", "exposure", "n_parents"} <= set(table.colnames)


def test_viz_footprints_defaults_to_one_colour(monkeypatch):
    captured = {}

    def fake_plot_footprints(polygons, **kwargs):
        captured["polygons"] = polygons
        captured.update(kwargs)
        return "plot"

    monkeypatch.setattr(
        "noobfriend.core.display.plot.plot_footprints", fake_plot_footprints
    )
    box = _box(
        _exposure_book("2a", "cal", exposure="00001", footprint=_footprint(0.0)),
        _exposure_book("2a", "cal", exposure="00002", footprint=_footprint(2.0)),
    )

    assert box.viz.footprints() == "plot"

    assert len(captured["polygons"]) == 2
    assert captured["colors"] == "#1f77b4"
    assert captured["labels"] == [book.id for book in box]


def test_viz_footprints_colours_by_group(monkeypatch):
    captured = {}

    def fake_plot_footprints(polygons, **kwargs):
        captured.update(kwargs)
        return "plot"

    monkeypatch.setattr(
        "noobfriend.core.display.plot.plot_footprints", fake_plot_footprints
    )
    box = _box(
        _exposure_book(
            "2a", "cal", exposure="00001", filter="F200W", footprint=_footprint(0.0)
        ),
        _exposure_book(
            "2a", "cal", exposure="00002", filter="F356W", footprint=_footprint(2.0)
        ),
        _exposure_book(
            "2a", "cal", exposure="00003", filter="F200W", footprint=_footprint(4.0)
        ),
    )

    assert (
        box.viz.footprints(
            group_by="filter",
            group_colors={"F200W": "red"},
            palette=["blue"],
        )
        == "plot"
    )

    assert captured["colors"] == ["red", "blue", "red"]


def test_save_load_roundtrip_preserves_lineage(tmp_path):
    rate = _exposure_book("1b", "rate")
    cal = _exposure_book("2a", "cal")
    box = _box(rate).merge(_box(cal))  # cal now carries parent_ids == (rate.id,)
    manifest = tmp_path / "noobox.json"

    box.save(manifest)
    assert isinstance(json.loads(manifest.read_text()), list)

    reloaded = NooBox.load(manifest)
    assert len(reloaded) == len(box)
    assert reloaded[cal.id] == box[cal.id]
    assert reloaded[cal.id].parent_ids == (rate.id,)
    assert reloaded[cal.id].exposure == cal.exposure
