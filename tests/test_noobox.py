"""Tests for NooBox container, discovery, subsetting, merge and persistence.

All cases stay in *thin* mode (``NooBook.from_name``), so no FITS file is ever
opened: discovery only needs file names to exist for globbing.
"""

import json
from pathlib import Path

import pytest

from noobfriend.navigation import NooBook, NooBox


def _touch(directory: Path, name: str) -> None:
    (directory / name).write_text("")


def test_from_directory_thin_indexes_matching_files(tmp_path):
    _touch(tmp_path, "jw01345001001_02201_00001_nrca1_cal.fits")
    _touch(tmp_path, "jw01345001001_02201_00002_nrca1_cal.fits")
    _touch(tmp_path, "notes.txt")

    box = NooBox.from_directory("2a", roots={"2a": str(tmp_path)})

    assert len(box) == 2
    assert all(book.stage == "2a" for book in box)
    assert all(not book.is_probed for book in box)
    assert "jw01345001001_02201_00001_nrca1_cal@2a" in box


def test_from_directory_custom_wildcard_filters(tmp_path):
    _touch(tmp_path, "jw01345001001_02201_00001_nrca1_cal.fits")
    _touch(tmp_path, "jw01345001001_02201_00001_nrca1_rate.fits")

    box = NooBox.from_directory(
        "2a", roots={"2a": str(tmp_path)}, wildcard="*_cal.fits"
    )

    assert [book.exposure for book in box] == [("00001",)]
    assert next(iter(box)).id.endswith("_cal@2a")


def test_from_directory_unresolved_stage_raises(tmp_path):
    with pytest.raises(ValueError, match="no root for stage"):
        NooBox.from_directory("zz", roots={"2a": str(tmp_path)})


def test_from_directory_remote_root_thin(monkeypatch):
    def fake_list_remote_dir(spec):
        assert spec == "host:/data/stage2"
        return ["jw01345001001_02201_00001_nrca1_cal.fits", "notes.txt"]

    monkeypatch.setattr(
        "noobfriend.navigation.noobox._core.list_remote_dir", fake_list_remote_dir
    )

    box = NooBox.from_directory("2a", roots={"2a": "host:/data/stage2"})

    assert len(box) == 1  # notes.txt excluded by the default *.fits wildcard
    book = next(iter(box))
    assert book.location == "host:/data/stage2/jw01345001001_02201_00001_nrca1_cal.fits"
    assert book.detector == "nrca1"
    assert not book.is_probed


def test_filter_returns_sharing_subset(tmp_path):
    _touch(tmp_path, "jw01345001001_02201_00001_nrca1_cal.fits")
    _touch(tmp_path, "jw01345001001_02201_00001_nrcb1_cal.fits")
    box = NooBox.from_directory("2a", roots={"2a": str(tmp_path)})

    subset = box.filter(lambda book: book.detector == "nrca1")

    assert len(subset) == 1
    assert next(iter(subset)).detector == "nrca1"
    assert len(box) == 2  # original untouched
    assert subset._store is box._store  # shares the byte cache


def test_merge_overwrite_false_raises_on_duplicate():
    book = NooBook.from_name("/data/jw01345001001_02201_00001_nrca1_cal.fits", "2a")
    left, right = NooBox(), NooBox()
    left.add(book)
    right.add(book)

    with pytest.raises(ValueError, match="duplicate id"):
        left.merge(right)


def test_merge_overwrite_true_right_wins():
    left, right = NooBox(), NooBox()
    left.add(NooBook(id="x@2a", location="/old.fits", stage="2a", program_id="01345"))
    right.add(NooBook(id="x@2a", location="/new.fits", stage="2a", program_id="01345"))

    merged = left.merge(right, overwrite=True)

    assert len(merged) == 1
    assert merged["x@2a"].location == "/new.fits"


def test_children_and_parents_use_lineage():
    box = NooBox()
    box.add(NooBook(id="parent@2a", location="/p.fits", stage="2a", program_id="01345"))
    box.add(
        NooBook(
            id="child@2b",
            location="/c.fits",
            stage="2b",
            program_id="01345",
            parent_ids=("parent@2a",),
        )
    )

    assert [b.id for b in box.children("parent@2a")] == ["child@2b"]
    assert [b.id for b in box.parents("child@2b")] == ["parent@2a"]


def test_save_load_roundtrip(tmp_path):
    _touch(tmp_path, "jw01345001001_02201_00001_nrca1_cal.fits")
    box = NooBox.from_directory("2a", roots={"2a": str(tmp_path)})
    manifest = tmp_path / "noobox.json"

    box.save(manifest)
    assert isinstance(json.loads(manifest.read_text()), list)

    reloaded = NooBox.load(manifest)
    assert len(reloaded) == len(box)
    original = next(iter(box))
    restored = reloaded[original.id]
    assert restored == original
    assert restored.detector == original.detector
    assert restored.exposure == original.exposure
