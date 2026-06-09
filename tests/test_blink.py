"""Tests for the navigation frame adapter feeding the core blink comparator."""

import math

import numpy as np
import pytest

from noobfriend.navigation import NooBook
from noobfriend.navigation._blink import _resolve_frames, _translation_offset


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
