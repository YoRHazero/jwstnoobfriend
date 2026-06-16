"""Tests for CutoutStore: in-memory, disk spill, round-trip, subset, cleanup."""

import numpy as np
import pytest

from noobfriend.extraction.psf import Cutout, CutoutStore


def _cutout(
    s: int = 5,
    val: float = 1.0,
    *,
    with_err: bool = True,
    with_bad: bool = True,
    origin: tuple[int, int] = (0, 0),
) -> Cutout:
    """Return a small Cutout with distinctive data / err / bad for round-trip."""
    err = np.full((s, s), 2.0, dtype=np.float32) if with_err else None
    bad = None
    if with_bad:
        bad = np.zeros((s, s), dtype=bool)
        bad[0, 0] = True
    return Cutout(
        data=np.full((s, s), val, dtype=np.float32), err=err, bad=bad, origin=origin
    )


class TestInMemory:
    """Default (no spill) behaviour."""

    def test_append_len_getitem_iter(self) -> None:
        store = CutoutStore(5)
        for i in range(3):
            store.append(_cutout(val=float(i), origin=(i, i)))
        assert len(store) == 3
        assert store[1].data[0, 0] == 1.0
        assert store[-1].origin == (2, 2)
        assert [c.data[0, 0] for c in store] == [0.0, 1.0, 2.0]

    def test_no_directory_created(self) -> None:
        store = CutoutStore(5)
        store.append(_cutout())
        assert store._dir is None  # pure memory, nothing on disk

    def test_out_of_range_raises(self) -> None:
        store = CutoutStore(5)
        store.append(_cutout())
        with pytest.raises(IndexError):
            _ = store[5]


class TestSpill:
    """Spilling to disk and reading back."""

    def test_spills_and_round_trips_in_order(self, tmp_path) -> None:
        store = CutoutStore(5, max_in_memory=2, spill_dir=tmp_path)
        for i in range(5):
            store.append(_cutout(val=float(i), origin=(i, 0)))
        assert len(store) == 5
        assert len(store._chunks) == 2  # two sealed chunks of 2, one still buffered

        assert [c.data[0, 0] for c in store] == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert [store[i].origin for i in range(5)] == [(i, 0) for i in range(5)]
        first = store[0]  # from a sealed chunk
        assert first.err is not None and first.err[0, 0] == 2.0
        assert first.bad is not None and bool(first.bad[0, 0]) is True
        store.close()

    def test_mixed_err_bad_round_trip(self, tmp_path) -> None:
        store = CutoutStore(5, max_in_memory=2, spill_dir=tmp_path)
        store.append(_cutout(with_err=True, with_bad=False))
        store.append(_cutout(with_err=False, with_bad=True))
        store.append(_cutout())  # third append seals the first two to disk
        assert store[0].err is not None and store[0].bad is None
        assert store[1].err is None and store[1].bad is not None
        store.close()

    def test_private_subdir_and_no_temp_leftovers(self, tmp_path) -> None:
        store = CutoutStore(5, max_in_memory=1, spill_dir=tmp_path)
        store.append(_cutout())
        store.append(_cutout())
        subdirs = list(tmp_path.iterdir())
        assert len(subdirs) == 1 and subdirs[0].name.startswith("noobpsf-")
        assert not list(tmp_path.glob("**/*.tmp"))  # atomic write left no temp file

        store.close()
        assert list(tmp_path.iterdir()) == []  # our subdir gone, spill_dir kept

    def test_context_manager_cleans_up(self, tmp_path) -> None:
        with CutoutStore(5, max_in_memory=1, spill_dir=tmp_path) as store:
            store.append(_cutout())
            store.append(_cutout())
            spill_dir = store._dir
            assert spill_dir.exists()
        assert not spill_dir.exists()
        store.close()  # idempotent

    def test_append_after_close_raises(self) -> None:
        store = CutoutStore(5)
        store.close()
        with pytest.raises(ValueError, match="closed"):
            store.append(_cutout())


class TestSubset:
    """subset() is order-preserving and independent of its parent."""

    def test_subset_in_order(self, tmp_path) -> None:
        store = CutoutStore(5, max_in_memory=2, spill_dir=tmp_path)
        for i in range(6):
            store.append(_cutout(val=float(i)))
        sub = store.subset([0, 2, 4])
        assert len(sub) == 3
        assert [c.data[0, 0] for c in sub] == [0.0, 2.0, 4.0]
        store.close()
        sub.close()

    def test_subset_survives_parent_close(self, tmp_path) -> None:
        store = CutoutStore(5, max_in_memory=2, spill_dir=tmp_path)
        for i in range(6):
            store.append(_cutout(val=float(i)))
        sub = store.subset([1, 3, 5])
        store.close()  # parent's files removed
        assert [c.data[0, 0] for c in sub] == [1.0, 3.0, 5.0]  # sub unaffected
        sub.close()
