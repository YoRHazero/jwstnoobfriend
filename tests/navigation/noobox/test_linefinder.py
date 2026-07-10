"""Tests for the book/box blind grism line-finder views.

No FITS file is opened: GrismLineFinder is faked, so only the navigation
orchestration -- dispersion-from-pupil, grouping, the FrameLineFinder /
BoxLineFinder contracts -- is exercised.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from noobfriend.navigation import Footprint, NooBook
from noobfriend.navigation import _linefinder as shared
from noobfriend.navigation.noobox.extract import _linefinder as boxlf

from ._helpers import make_box, make_grism_book


def _square() -> Footprint:
    return Footprint.from_corners([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def _grism_book(
    *,
    ident: str = "00001",
    detector: str = "nrca1",
    pupil: str = "GRISMR",
    visit: str = "001",
    footprint: Footprint | None = None,
) -> NooBook:
    return make_grism_book(
        ident=ident,
        detector=detector,
        pupil=pupil,
        visit=visit,
        footprint=footprint or _square(),
        shape=(4, 4),
    )


def _patch_pixels(monkeypatch, *, err_none: bool = False) -> None:
    monkeypatch.setattr(NooBook, "data", property(lambda self: np.ones((4, 4))))
    monkeypatch.setattr(NooBook, "wcs", property(lambda self: object()))
    err = None if err_none else np.full((4, 4), 2.0)
    monkeypatch.setattr(NooBook, "err", property(lambda self: err))


def _patch_finder(monkeypatch):
    """Replace GrismLineFinder with a stand-in recording configure/combine/catalog."""

    class FakeFinder:
        def __init__(self, dispersion, config):
            self.dispersion = dispersion
            self.config = config

        @classmethod
        def configure(cls, *, dispersion, **config):
            return cls(dispersion, config)

        def exposure_heatmap(self, data, error):
            return np.asarray(data, dtype=float)  # marker = the (ones) data

        def combine(self, metas, load, *, reference_index=0):
            for meta in metas:
                load(meta.id)  # exercise BoxLineFinder._load_for
            return SimpleNamespace(heatmap=np.full((4, 4), float(len(list(metas)))))

        def catalog(self, heatmap):
            return [("peak", float(np.nanmax(heatmap)))]

    monkeypatch.setattr(
        "noobfriend.extraction.grism.linefind.GrismLineFinder", FakeFinder
    )
    return FakeFinder


# -- shared helpers -----------------------------------------------------------


def test_dispersion_of_and_default_group():
    assert shared._dispersion_of("GRISMR") == "row"
    assert shared._dispersion_of("GRISMC") == "column"
    assert shared._dispersion_of("CLEAR") is None
    assert shared._dispersion_of(None) is None

    book = _grism_book(detector="nrcb1", pupil="GRISMC", visit="004")
    assert boxlf._default_group(book) == (("001",), ("004",), "nrcb1", "GRISMC")


def test_configure_rejects_non_grism():
    with pytest.raises(ValueError, match="not a grism frame"):
        shared.configure(_grism_book(pupil="CLEAR"))


# -- FrameLineFinder (book) ---------------------------------------------------


def test_frame_linefinder_heatmap_catalog_and_cache(monkeypatch):
    _patch_finder(monkeypatch)
    _patch_pixels(monkeypatch)
    book = _grism_book()

    handle = book.extract.linefinder(threshold=5.0)
    heatmap = handle.heatmap

    np.testing.assert_array_equal(heatmap, np.ones((4, 4)))
    assert handle.heatmap is heatmap  # cached
    assert handle.catalog() == [("peak", 1.0)]


def test_frame_linefinder_requires_err(monkeypatch):
    _patch_finder(monkeypatch)
    _patch_pixels(monkeypatch, err_none=True)

    with pytest.raises(ValueError, match="no ERR"):
        _ = _grism_book().extract.linefinder().heatmap


# -- BoxLineFinder ------------------------------------------------------------


def test_box_linefinder_default_grouping_skips_non_grism(monkeypatch):
    _patch_finder(monkeypatch)
    a1 = _grism_book(ident="00001", visit="001", detector="nrca1")
    a2 = _grism_book(ident="00002", visit="001", detector="nrca1")  # same group as a1
    b = _grism_book(ident="00003", visit="002", detector="nrca1")  # different visit
    clear = _grism_book(ident="00004", pupil="CLEAR")  # not grism

    lf = make_box(a1, a2, b, clear).extract.linefinder(probe=False)

    groups = lf.groups
    assert len(groups) == 2  # (visit 001) and (visit 002); CLEAR excluded
    sizes = sorted(len(v) for v in groups.values())
    assert sizes == [1, 2]


def test_box_linefinder_heatmaps_catalog_exposure(monkeypatch):
    _patch_finder(monkeypatch)
    _patch_pixels(monkeypatch)
    a1 = _grism_book(ident="00001", visit="001")
    a2 = _grism_book(ident="00002", visit="001")
    lf = make_box(a1, a2).extract.linefinder(probe=False)

    heatmaps = lf.heatmaps
    assert len(heatmaps) == 1
    (combined,) = heatmaps.values()
    np.testing.assert_array_equal(combined.heatmap, np.full((4, 4), 2.0))  # 2 frames

    catalog = lf.catalog()
    assert len(catalog) == 1
    assert next(iter(catalog.values())) == [("peak", 2.0)]

    exposures = lf.exposure_heatmaps
    assert set(exposures) == {a1, a2}  # keyed by NooBook


def test_box_linefinder_group_by_override(monkeypatch):
    _patch_finder(monkeypatch)
    a = _grism_book(ident="00001", detector="nrca1")
    b = _grism_book(ident="00002", detector="nrcb1")
    lf = make_box(a, b).extract.linefinder(group_by="detector", probe=False)

    assert sorted(lf.groups) == ["nrca1", "nrcb1"]


def test_box_linefinder_no_grism_raises():
    with pytest.raises(ValueError, match="no grism"):
        make_box(_grism_book(pupil="CLEAR")).extract.linefinder(probe=False)
