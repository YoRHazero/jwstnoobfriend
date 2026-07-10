"""Tests for the BoxGrism navigation view over extraction.grism.

No FITS file is opened: the extraction layer (FrameMeta / find_coverage /
GrismExtractor) and the seed-trace WCS step are faked, so only the navigation
orchestration -- grism selection, the footprint coverage gate, grouping, and the
BoxGrism combine/spectra contract -- is exercised.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from noobfriend.navigation import Footprint, NooBook, NooBox
from noobfriend.navigation.noobox.extract import _grism

from ._helpers import make_grism_book


def _square(cx: float, cy: float, half: float = 0.01) -> Footprint:
    """Axis-aligned square footprint; corners in (0,0),(nx,0),(nx,ny),(0,ny) order."""
    return Footprint.from_corners(
        [
            (cx - half, cy - half),
            (cx + half, cy - half),
            (cx + half, cy + half),
            (cx - half, cy + half),
        ]
    )


# -- grism identity / grouping ------------------------------------------------


def test_grism_identity_and_default_group():
    a = make_grism_book(detector="nrca1", pupil="GRISMR")
    b = make_grism_book(detector="nrcb3", pupil="GRISMC")

    assert _grism._is_grism(a)
    assert not _grism._is_grism(make_grism_book(pupil="CLEAR"))
    assert _grism._default_group(a) == ("a", "row")
    assert _grism._default_group(b) == ("b", "column")


# -- gate geometry ------------------------------------------------------------


def test_segment_hits_polygon():
    unit_square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]

    assert _grism._segment_hits_polygon(
        (0.5, 0.5), (0.5, 2.0), unit_square
    )  # inside end
    assert _grism._segment_hits_polygon((-1.0, 0.5), (2.0, 0.5), unit_square)  # crosses
    assert not _grism._segment_hits_polygon((2.0, 2.0), (3.0, 3.0), unit_square)  # far


def test_dispersion_unit_row_and_column():
    corners = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]

    assert _grism._dispersion_unit(corners, "row") == (1.0, 0.0)
    assert _grism._dispersion_unit(corners, "column") == (0.0, 1.0)


# -- footprint pre-gate -------------------------------------------------------


def test_pre_gate_selects_by_trace_segment(monkeypatch):
    monkeypatch.setattr(_grism, "_seed_extent", lambda *a, **k: (0.0, 0.02))
    near = make_grism_book(ident="00001", footprint=_square(0.015, 0.0))
    far = make_grism_book(ident="00002", footprint=_square(0.5, 0.0))

    candidates = _grism._pre_gate([near, far], 0.0, 0.0)

    assert near in candidates
    assert far not in candidates


def test_pre_gate_per_frame_direction_handles_rotation(monkeypatch):
    """A frame whose dispersion axis is rotated still selects via its own corners."""
    monkeypatch.setattr(_grism, "_seed_extent", lambda *a, **k: (0.0, 0.02))
    axis_aligned = make_grism_book(ident="00001", footprint=_square(0.015, 0.0))
    # corners arranged so detector-x (corner1 - corner0) points +Dec, not +RA;
    # footprint sits north of the source, where the trace lands only if its
    # direction is taken from this frame's own corners.
    rotated = make_grism_book(
        ident="00002",
        footprint=Footprint.from_corners(
            [(0.01, 0.005), (0.01, 0.025), (-0.01, 0.025), (-0.01, 0.005)]
        ),
    )

    candidates = _grism._pre_gate([axis_aligned, rotated], 0.0, 0.0)

    assert (
        rotated in candidates
    )  # would be missed if the seed's +RA direction were reused


# -- BoxGrism combine / spectra contract --------------------------------------


@dataclass(frozen=True)
class _FakeSpec:
    group: object = None
    n: int = 1


class _FakeExtractor:
    """Minimal stand-in for GrismExtractor: rectify keys by id, combine by group."""

    def __init__(self, books: list[NooBook]) -> None:
        self._books = books
        self.wavelength = SimpleNamespace(values=np.array([1.0, 2.0, 3.0]))

    @property
    def coverage(self):
        return [SimpleNamespace(meta=SimpleNamespace(id=b.id)) for b in self._books]

    def rectify(self, load):
        out = {}
        for book in self._books:
            load(book.id)  # exercises BoxGrism._load (reads data/err)
            out[book.id] = _FakeSpec(group=None, n=1)
        return out

    def combine(self, spec_map):
        groups: dict[object, list[_FakeSpec]] = {}
        for spec in spec_map.values():
            groups.setdefault(spec.group, []).append(spec)
        return [_FakeSpec(group=key, n=len(v)) for key, v in groups.items()]


def _patch_pixels(monkeypatch, *, err_none: bool = False) -> None:
    monkeypatch.setattr(NooBook, "data", property(lambda self: np.ones((4, 4))))
    err = None if err_none else np.ones((4, 4))
    monkeypatch.setattr(NooBook, "err", property(lambda self: err))


def _boxgrism(books: list[NooBook]) -> _grism.BoxGrism:
    return _grism.BoxGrism(
        ra=0.0,
        dec=0.0,
        extractor=_FakeExtractor(books),
        coverage=[],
        candidates=books,
        books_by_id={b.id: b for b in books},
    )


def test_boxgrism_spectra_keyed_by_book(monkeypatch):
    _patch_pixels(monkeypatch)
    a = make_grism_book(ident="00001", detector="nrca1", pupil="GRISMR")
    b = make_grism_book(ident="00002", detector="nrcb1", pupil="GRISMC")
    g = _boxgrism([a, b])

    spectra = g.spectra

    assert set(spectra) == {a, b}
    assert g.spectra is spectra  # cached on second access


def test_boxgrism_combine_grouping(monkeypatch):
    _patch_pixels(monkeypatch)
    a = make_grism_book(ident="00001", detector="nrca1", pupil="GRISMR")
    b = make_grism_book(ident="00002", detector="nrcb1", pupil="GRISMC")
    g = _boxgrism([a, b])

    single = g.combine(group_by=None)
    assert not isinstance(single, list)
    assert single.n == 2  # both exposures co-added into one

    grouped = g.combine()  # default (module, dispersion) -> two distinct groups
    assert isinstance(grouped, list)
    assert len(grouped) == 2

    by_detector = g.combine(group_by="detector")
    assert len(by_detector) == 2


def test_boxgrism_spectra_requires_err(monkeypatch):
    _patch_pixels(monkeypatch, err_none=True)
    g = _boxgrism([make_grism_book(ident="00001")])

    with pytest.raises(ValueError, match="no ERR"):
        _ = g.spectra


# -- _build wiring ------------------------------------------------------------


def test_build_selects_grism_and_keeps_covered(monkeypatch):
    grism_covered = make_grism_book(
        ident="00001", pupil="GRISMR", footprint=_square(0.0, 0.0)
    )
    grism_uncovered = make_grism_book(
        ident="00002", pupil="GRISMR", footprint=_square(0.0, 0.0)
    )
    clear = make_grism_book(ident="00003", pupil="CLEAR")
    monkeypatch.setattr(NooBook, "wcs", property(lambda self: object()))
    monkeypatch.setattr(_grism, "_pre_gate", lambda books, ra, dec: list(books))

    @dataclass
    class _FakeMeta:
        id: str
        wcs: object
        shape: tuple

    def _fake_find_coverage(ra, dec, metas, *, min_fraction=0.0):
        return [
            SimpleNamespace(covered=(m.id == grism_covered.id), meta=m) for m in metas
        ]

    class _FakeGExtractor:
        @classmethod
        def from_world(cls, ra, dec, covered, *, spatial_half, oversample, wavelength):
            inst = cls()
            inst.coverage = [SimpleNamespace(meta=c.meta) for c in covered]
            inst.wavelength = SimpleNamespace(values=np.array([1.0, 2.0]))
            return inst

    monkeypatch.setattr("noobfriend.extraction.grism.FrameMeta", _FakeMeta)
    monkeypatch.setattr(
        "noobfriend.extraction.grism.find_coverage", _fake_find_coverage
    )
    monkeypatch.setattr("noobfriend.extraction.grism.GrismExtractor", _FakeGExtractor)

    box = NooBox()
    for book in (grism_covered, grism_uncovered, clear):
        box.add(book)

    result = box.extract.grism(0.0, 0.0, spatial_half=8, probe=False, progress=False)

    # CLEAR excluded (not grism); uncovered grism dropped by find_coverage.
    assert [b.id for b in result.books] == [grism_covered.id]


def test_build_raises_without_grism_frames():
    box = NooBox()
    box.add(make_grism_book(ident="00001", pupil="CLEAR"))

    with pytest.raises(ValueError, match="no grism"):
        box.extract.grism(0.0, 0.0, spatial_half=8, probe=False, progress=False)


def test_build_tolerates_uncoverable_frame(monkeypatch):
    """A frame whose dispersion model raises (source far off-array) is dropped."""
    covered = make_grism_book(
        ident="00001", pupil="GRISMR", footprint=_square(0.0, 0.0)
    )
    crashes = make_grism_book(
        ident="00002", pupil="GRISMR", footprint=_square(0.0, 0.0)
    )
    monkeypatch.setattr(NooBook, "wcs", property(lambda self: object()))
    monkeypatch.setattr(_grism, "_pre_gate", lambda books, ra, dec: list(books))
    monkeypatch.setattr(
        "noobfriend.extraction._wcs.world_detector_transforms",
        lambda wcs: ((lambda r, d: (0.0, 0.0)), (lambda x, y: (0.0, 0.0))),
    )

    @dataclass
    class _FakeMeta:
        id: str
        wcs: object
        shape: tuple

    def _fake_find_coverage(ra, dec, metas, *, min_fraction=0.0):
        if metas[0].id == crashes.id:
            raise ValueError("Wavelength should be greater than zero")
        return [SimpleNamespace(covered=True, meta=metas[0])]

    class _FakeGExtractor:
        @classmethod
        def from_world(cls, ra, dec, cov, *, spatial_half, oversample, wavelength):
            inst = cls()
            inst.coverage = [SimpleNamespace(meta=c.meta) for c in cov]
            inst.wavelength = SimpleNamespace(values=np.array([1.0, 2.0]))
            return inst

    monkeypatch.setattr("noobfriend.extraction.grism.FrameMeta", _FakeMeta)
    monkeypatch.setattr(
        "noobfriend.extraction.grism.find_coverage", _fake_find_coverage
    )
    monkeypatch.setattr("noobfriend.extraction.grism.GrismExtractor", _FakeGExtractor)

    box = NooBox()
    for book in (covered, crashes):
        box.add(book)

    result = box.extract.grism(0.0, 0.0, spatial_half=8, probe=False, progress=False)

    assert [b.id for b in result.books] == [covered.id]
    # both frames appear in the coverage table, the crasher marked not covered
    assert {c.meta.id: c.covered for c in result.coverage} == {
        covered.id: True,
        crashes.id: False,
    }
