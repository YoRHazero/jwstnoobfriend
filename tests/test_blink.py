"""Tests for the navigation frame adapter feeding the core blink comparator."""

import math

import numpy as np
import pytest

from noobfriend.navigation import NooBook, NooBox
from noobfriend.navigation._blink import (
    _resolve_frames,
    _translation_offset,
    blink_frames,
)


def _thin_book(book_id: str) -> NooBook:
    """Return a minimal, file-less book (its ``data`` is stubbed per test)."""
    return NooBook(id=book_id, location="/p.fits", stage="2a", program_id="01345")


class TestResolveFrames:
    """Reducing mixed NooBook/array frames to ``(images, labels)``."""

    def test_book_contributes_its_data_array(self, monkeypatch) -> None:
        arr = np.arange(6.0).reshape(2, 3)
        monkeypatch.setattr(NooBook, "data", property(lambda self: arr))
        raw = np.ones((2, 2))
        images, _ = _resolve_frames([_thin_book("x@2a"), raw], None)
        assert images[0] is arr  # book -> its SCI array
        assert images[1] is raw  # ndarray passed through, no copy

    def test_default_labels_are_id_or_index(self, monkeypatch) -> None:
        monkeypatch.setattr(NooBook, "data", property(lambda self: np.zeros((1, 1))))
        # Array sits at index 1, so it is labelled "1" (its position), not "0".
        _, labels = _resolve_frames([_thin_book("x@2a"), np.zeros((1, 1))], None)
        assert labels == ["x@2a", "1"]

    def test_explicit_labels_used_verbatim(self, monkeypatch) -> None:
        monkeypatch.setattr(NooBook, "data", property(lambda self: np.zeros((1, 1))))
        _, labels = _resolve_frames(
            [_thin_book("x@2a"), np.zeros((1, 1))], ["sci", "seg"]
        )
        assert labels == ["sci", "seg"]

    def test_labels_length_mismatch_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(NooBook, "data", property(lambda self: np.zeros((1, 1))))
        with pytest.raises(ValueError, match="labels has length 1, expected 2"):
            _resolve_frames([_thin_book("x@2a"), np.zeros((1, 1))], ["only-one"])


class TestBlinkFrames:
    """Explicit navigation blink options forwarded to the core plot."""

    def test_forwards_display_options(self, monkeypatch) -> None:
        arr = np.zeros((2, 2))
        monkeypatch.setattr(NooBook, "data", property(lambda self: arr))
        captured = {}

        def fake_imshow_blink(images, **kwargs):
            captured["images"] = images
            captured.update(kwargs)
            return "plot"

        monkeypatch.setattr(
            "noobfriend.core.display.plot.imshow_blink", fake_imshow_blink
        )

        result = blink_frames(
            [_thin_book("x@2a"), np.ones((3, 3))],
            labels=["science", "mask"],
            offsets=[(0.0, 0.0), (2.0, -1.0)],
            vmin=[0.0, 1.0],
            vmax=5.0,
            pmin=2.0,
            pmax=98.0,
            cmap="Viridis",
            stretch="log",
            size=512,
            title="Blink",
            blink=False,
        )

        assert result == "plot"
        assert captured["labels"] == ["science", "mask"]
        assert captured["offsets"] == [(0.0, 0.0), (2.0, -1.0)]
        assert captured["vmin"] == [0.0, 1.0]
        assert captured["vmax"] == 5.0
        assert captured["pmin"] == 2.0
        assert captured["pmax"] == 98.0
        assert captured["cmap"] == "Viridis"
        assert captured["stretch"] == "log"
        assert captured["size"] == 512
        assert captured["title"] == "Blink"
        assert captured["blink"] is False

    def test_align_wcs_conflicts_with_offsets(self, monkeypatch) -> None:
        monkeypatch.setattr(NooBook, "data", property(lambda self: np.zeros((1, 1))))

        with pytest.raises(ValueError, match="either align='wcs' or offsets"):
            blink_frames([_thin_book("x@2a")], align="wcs", offsets=[(0.0, 0.0)])


class TestVizBlinkAccessors:
    """NooBook/NooBox blink accessors expose the same explicit options."""

    def test_book_viz_forwards_imshow_options(self, monkeypatch) -> None:
        arr = np.zeros((2, 2))
        monkeypatch.setattr(NooBook, "data", property(lambda self: arr))
        captured = {}

        def fake_imshow(data, **kwargs):
            captured["data"] = data
            captured.update(kwargs)
            return "image"

        monkeypatch.setattr("noobfriend.core.display.plot.imshow", fake_imshow)
        book = _thin_book("x@2a")

        assert (
            book.viz.imshow(
                vmin=0.0,
                vmax=10.0,
                pmin=5.0,
                pmax=95.0,
                cmap="Viridis",
                stretch="eqhist",
                size=400,
                title="Image",
                coord_format="hms",
            )
            == "image"
        )

        assert captured["data"] is arr
        assert captured["vmin"] == 0.0
        assert captured["vmax"] == 10.0
        assert captured["pmin"] == 5.0
        assert captured["pmax"] == 95.0
        assert captured["cmap"] == "Viridis"
        assert captured["stretch"] == "eqhist"
        assert captured["size"] == 400
        assert captured["title"] == "Image"
        assert captured["coord_format"] == "hms"

    def test_book_viz_forwards_blink_options(self, monkeypatch) -> None:
        captured = {}

        def fake_blink_frames(frames, **kwargs):
            captured["frames"] = frames
            captured.update(kwargs)
            return "plot"

        monkeypatch.setattr(
            "noobfriend.navigation._blink.blink_frames", fake_blink_frames
        )
        book = _thin_book("x@2a")
        other = np.ones((2, 2))

        assert (
            book.viz.imshow_blink(
                other,
                labels=["book", "array"],
                offsets=[(0.0, 0.0), (1.0, 1.0)],
                blink=False,
            )
            == "plot"
        )

        assert captured["frames"] == [book, other]
        assert captured["labels"] == ["book", "array"]
        assert captured["offsets"] == [(0.0, 0.0), (1.0, 1.0)]
        assert captured["blink"] is False

    def test_box_viz_forwards_blink_options(self, monkeypatch) -> None:
        captured = {}

        def fake_blink_frames(frames, **kwargs):
            captured["frames"] = frames
            captured.update(kwargs)
            return "plot"

        monkeypatch.setattr(
            "noobfriend.navigation._blink.blink_frames", fake_blink_frames
        )
        box = NooBox()
        first = box.add(_thin_book("a@2a"))
        second = box.add(_thin_book("b@2a"))

        assert box.viz.imshow_blink(labels=["a", "b"], size=480, blink=False) == "plot"

        assert captured["frames"] == [first, second]
        assert captured["labels"] == ["a", "b"]
        assert captured["size"] == 480
        assert captured["blink"] is False


def _identity(a: float, b: float) -> tuple[float, float]:
    """Pass coordinates through, as a stand-in world<->detector leg."""
    return a, b


class TestTranslationOffset:
    """Recovering a pure-translation offset and rejecting non-translations.

    The transforms are plain callables (no GWCS): "world" coordinates are taken
    to equal the anchor's pixel coordinates, so a frame whose
    ``detector_to_world`` adds ``(dx, dy)`` sits ``(dx, dy)`` from the anchor.
    """

    def test_pure_translation_recovers_shift(self) -> None:
        def frame_d2w(x: float, y: float) -> tuple[float, float]:
            return x + 3.0, y - 5.0

        assert _translation_offset(_identity, frame_d2w, (50, 40), 1.0) == (3.0, -5.0)

    def test_identity_is_zero_offset(self) -> None:
        assert _translation_offset(_identity, _identity, (10, 10), 1.0) == (0.0, 0.0)

    def test_rotation_rejected(self) -> None:
        theta = 0.05  # radians; corner drift across 200 px >> atol
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        def frame_d2w(x: float, y: float) -> tuple[float, float]:
            return cos_t * x - sin_t * y, sin_t * x + cos_t * y

        with pytest.raises(ValueError, match="more than a translation"):
            _translation_offset(_identity, frame_d2w, (200, 200), 1.0)
