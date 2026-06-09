"""Tests for the thin scikit-image segmentation wrapper in core."""

import numpy as np
import pytest

from noobfriend.core.imgutils import segment


def _blob(image: np.ndarray, top: int, left: int, size: int, value: float) -> None:
    """Set a square block of ``image`` to ``value`` in place."""
    image[top : top + size, left : left + size] = value


class TestSegment:
    """Behaviour of the segmentation wrapper on synthetic images."""

    def test_two_separated_blobs_get_distinct_labels(self) -> None:
        img = np.zeros((20, 20))
        _blob(img, 2, 2, 4, 10.0)
        _blob(img, 12, 12, 4, 10.0)

        labels = segment(img, threshold=1.0, deblend=False)

        ids = set(np.unique(labels)) - {0}
        assert len(ids) == 2
        assert labels[3, 3] != 0
        assert labels[13, 13] != 0
        assert labels[3, 3] != labels[13, 13]

    def test_threshold_excludes_faint_blob(self) -> None:
        img = np.zeros((20, 20))
        _blob(img, 2, 2, 4, 0.4)  # below threshold
        _blob(img, 12, 12, 4, 10.0)  # above threshold

        labels = segment(img, threshold=1.0, deblend=False)

        assert labels[3, 3] == 0
        assert labels[13, 13] != 0
        assert set(np.unique(labels)) - {0} == {labels[13, 13]}

    def test_min_size_removes_specks(self) -> None:
        img = np.zeros((10, 10))
        img[5, 5] = 10.0

        assert segment(img, threshold=1.0, min_size=5).max() == 0
        assert segment(img, threshold=1.0, min_size=1)[5, 5] != 0

    def test_deblend_splits_touching_blobs(self) -> None:
        yy, xx = np.mgrid[0:9, 0:15]
        img = 10.0 * np.exp(-(((xx - 4) ** 2 + (yy - 4) ** 2) / 4.0))
        img += 10.0 * np.exp(-(((xx - 10) ** 2 + (yy - 4) ** 2) / 4.0))

        merged = segment(img, threshold=1.0, deblend=False)
        split = segment(img, threshold=1.0, deblend=True)

        assert len(set(np.unique(merged)) - {0}) == 1
        assert len(set(np.unique(split)) - {0}) == 2

    def test_nan_is_background(self) -> None:
        img = np.full((10, 10), np.nan)
        _blob(img, 2, 2, 4, 10.0)

        labels = segment(img, threshold=1.0, deblend=False)

        assert labels[0, 0] == 0
        assert labels[3, 3] != 0
        assert segment(np.full((5, 5), np.nan)).max() == 0

    def test_auto_threshold_from_noise(self) -> None:
        # Deterministic background: median 0, MAD 1 -> sigma ~1.48,
        # so a 2-sigma threshold sits near 2.97; noise (max 2) stays background.
        img = np.tile(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), (20, 4))
        _blob(img, 8, 8, 4, 20.0)

        labels = segment(img, nsigma=2.0, deblend=False)

        assert labels[9, 9] != 0
        assert labels[0, 0] == 0

    def test_connectivity_distinguishes_diagonal_neighbours(self) -> None:
        img = np.zeros((6, 6))
        img[1, 1] = 10.0
        img[2, 2] = 10.0  # diagonally adjacent to (1, 1)

        eight = segment(img, threshold=1.0, min_size=1, connectivity=8, deblend=False)
        four = segment(img, threshold=1.0, min_size=1, connectivity=4, deblend=False)

        assert len(set(np.unique(eight)) - {0}) == 1
        assert eight[1, 1] == eight[2, 2]
        assert len(set(np.unique(four)) - {0}) == 2
        assert four[1, 1] != four[2, 2]

    def test_invalid_connectivity_raises(self) -> None:
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            segment(np.zeros((4, 4)), connectivity=6)  # type: ignore[arg-type]

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            segment(np.zeros((3, 3, 3)))
