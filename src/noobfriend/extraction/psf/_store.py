"""Bounded-memory storage for PSF cutouts, spilling to disk past a threshold.

Holding every cutout of a whole field in memory can exhaust RAM, so a
:class:`CutoutStore` keeps at most ``max_in_memory`` cutouts resident and seals
the rest into append-only ``.npz`` chunks on disk. Safety is the priority:

- **Independent ownership.** Each store gets its own private ``mkdtemp``
  directory, so two stores (e.g. an extractor and a sub-extractor from
  :meth:`subset`) never share files; cleaning one can never corrupt another.
- **Atomic, append-only chunks.** A chunk is written to a temporary file,
  fsynced, then ``os.replace``-d into place; a crash mid-write cannot leave a
  half-written chunk, and a sealed chunk is never rewritten.
- **Mixed inputs are safe.** ``err`` / ``bad`` may be absent per cutout; chunks
  carry presence flags so a missing ``err`` round-trips back to ``None``.
- **Deterministic cleanup with a backstop.** :meth:`close` (and the context
  manager) removes the temp directory; a :func:`weakref.finalize` backstop runs
  it if the caller forgets. Only the directory the store created is removed --
  an explicit ``spill_dir`` is never deleted, only the private sub-directory
  inside it.

Pure in-memory behaviour (no ``max_in_memory`` and no ``spill_dir``) creates no
directory and is the default.
"""

import os
import tempfile
import weakref
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Cutouts buffered in memory before a chunk is sealed when only ``spill_dir``
#: (not ``max_in_memory``) is given.
_DEFAULT_CHUNK = 4096


@dataclass(frozen=True)
class Cutout:
    """A cached fixed-size cutout around one detection (build-time stamp source).

    Attributes
    ----------
    data : numpy.ndarray
        ``(cutout_size, cutout_size)`` float32 data window.
    err : numpy.ndarray or None
        Matching error window, or ``None`` when no error map was given.
    bad : numpy.ndarray or None
        Boolean window where ``True`` marks DQ-flagged pixels, or ``None``.
    origin : tuple of int
        ``(row, col)`` of the window's top-left corner in the source frame.
    """

    data: np.ndarray
    err: np.ndarray | None
    bad: np.ndarray | None
    origin: tuple[int, int]


def _cleanup_dir(path: str) -> None:
    """Remove a spill directory and its chunks (idempotent, never raises)."""
    import shutil

    shutil.rmtree(path, ignore_errors=True)


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory so a rename into it is durable."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_chunk(path: Path, cutouts: list[Cutout], cutout_size: int) -> None:
    """Atomically write ``cutouts`` to ``path`` as a single ``.npz`` chunk."""
    n = len(cutouts)
    s = cutout_size
    data = np.empty((n, s, s), dtype=np.float32)
    err = np.full((n, s, s), np.nan, dtype=np.float32)
    bad = np.zeros((n, s, s), dtype=bool)
    has_err = np.zeros(n, dtype=bool)
    has_bad = np.zeros(n, dtype=bool)
    origin = np.empty((n, 2), dtype=np.int64)
    for j, cut in enumerate(cutouts):
        data[j] = cut.data
        if cut.err is not None:
            err[j] = cut.err
            has_err[j] = True
        if cut.bad is not None:
            bad[j] = cut.bad
            has_bad[j] = True
        origin[j] = cut.origin

    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(
            handle,
            data=data,
            err=err,
            bad=bad,
            has_err=has_err,
            has_bad=has_bad,
            origin=origin,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _read_chunk(path: Path) -> list[Cutout]:
    """Load a ``.npz`` chunk back into independent :class:`Cutout` objects."""
    with np.load(path) as npz:
        data = npz["data"]
        err = npz["err"]
        bad = npz["bad"]
        has_err = npz["has_err"]
        has_bad = npz["has_bad"]
        origin = npz["origin"]
        return [
            Cutout(
                data=data[j].copy(),
                err=err[j].copy() if has_err[j] else None,
                bad=bad[j].copy() if has_bad[j] else None,
                origin=(int(origin[j, 0]), int(origin[j, 1])),
            )
            for j in range(data.shape[0])
        ]


class CutoutStore:
    """An append-only store of :class:`Cutout`, bounded in memory, spilling to disk.

    Parameters
    ----------
    cutout_size : int
        Edge length of every cutout (chunks stack uniformly-shaped windows).
    max_in_memory : int, optional
        Cutouts kept resident before a chunk is sealed to disk. ``None`` (with no
        ``spill_dir``) keeps everything in memory.
    spill_dir : str or Path, optional
        Base directory for spilled chunks; a private sub-directory is created
        inside it. When given, spilling is enabled even if ``max_in_memory`` is
        ``None`` (a default chunk size is used). ``None`` uses the system temp
        area when spilling.
    """

    def __init__(
        self,
        cutout_size: int,
        *,
        max_in_memory: int | None = None,
        spill_dir: str | Path | None = None,
    ) -> None:
        """See the class docstring for parameters."""
        self.cutout_size = int(cutout_size)
        self._max_in_memory = max_in_memory
        self._spill_dir_arg = spill_dir
        self._spilling = max_in_memory is not None or spill_dir is not None
        self._chunk_size = (
            max_in_memory if max_in_memory is not None else _DEFAULT_CHUNK
        )

        self._buffer: list[Cutout] = []
        self._chunks: list[Path] = []
        self._chunk_lens: list[int] = []
        self._n = 0

        self._dir: Path | None = None
        self._owns_dir: bool = True
        self._finalizer: weakref.finalize | None = None
        self._loaded_index = -1
        self._loaded: list[Cutout] | None = None
        self._closed = False

    # -- mutation -------------------------------------------------------------

    def append(self, cutout: Cutout) -> None:
        """Append a cutout, sealing a chunk to disk if the buffer is full."""
        if self._closed:
            raise ValueError("cannot append to a closed CutoutStore.")
        self._buffer.append(cutout)
        self._n += 1
        if self._spilling and len(self._buffer) >= self._chunk_size:
            self._seal()

    def _seal(self) -> None:
        """Write the in-memory buffer to a new chunk and clear it."""
        if not self._buffer:
            return
        if self._dir is None:
            self._open_dir()
        assert self._dir is not None
        path = self._dir / f"chunk_{len(self._chunks):06d}.npz"
        _write_chunk(path, self._buffer, self.cutout_size)
        self._chunks.append(path)
        self._chunk_lens.append(len(self._buffer))
        self._buffer = []

    def _open_dir(self) -> None:
        """Create this store's private spill directory and register cleanup."""
        base = self._spill_dir_arg
        if base is not None:
            Path(base).mkdir(parents=True, exist_ok=True)
        self._dir = Path(tempfile.mkdtemp(prefix="noobpsf-", dir=base))
        self._finalizer = weakref.finalize(self, _cleanup_dir, str(self._dir))

    # -- access ---------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of cutouts stored."""
        return self._n

    def __getitem__(self, key: int | slice) -> "Cutout | list[Cutout]":
        """Return the cutout at an index (or a list for a slice)."""
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(self._n))]
        index = key + self._n if key < 0 else key
        if not 0 <= index < self._n:
            raise IndexError(f"cutout index {key} out of range (len {self._n}).")
        offset = index
        for chunk_index, length in enumerate(self._chunk_lens):
            if offset < length:
                return self._load(chunk_index)[offset]
            offset -= length
        return self._buffer[offset]

    def __iter__(self) -> Iterator[Cutout]:
        """Iterate cutouts in append order, loading at most one chunk at a time."""
        for chunk_index in range(len(self._chunks)):
            yield from self._load(chunk_index)
        yield from self._buffer

    def _load(self, chunk_index: int) -> list[Cutout]:
        """Return chunk ``chunk_index``, caching the most recently loaded one."""
        if self._loaded_index != chunk_index:
            self._loaded = _read_chunk(self._chunks[chunk_index])
            self._loaded_index = chunk_index
        assert self._loaded is not None
        return self._loaded

    def subset(self, indices: Sequence[int]) -> "CutoutStore":
        """Return a new independent store with the cutouts at ``indices``.

        The new store inherits this store's spill configuration (so a large
        subset spills too) but owns its own directory and files -- closing or
        finalising either store never affects the other. Ascending ``indices``
        (as produced by a boolean selection) touch each chunk once.
        """
        sub = CutoutStore(
            self.cutout_size,
            max_in_memory=self._max_in_memory,
            spill_dir=self._spill_dir_arg,
        )
        for index in indices:
            sub.append(self[index])  # type: ignore[arg-type]
        return sub

    # -- persistence ----------------------------------------------------------

    def save(
        self, directory: str | Path, *, chunk_size: int | None = None
    ) -> list[int]:
        """Write every cutout to ``directory`` as append-only ``.npz`` chunks.

        Streams the cutouts (at most one source chunk resident) into fixed-size
        ``chunk_000000.npz`` ... files, independent of this store's own in-memory
        or spill state. The files use the same atomic format as the live spill
        chunks and are **not** owned by any store -- they persist past
        :meth:`close` and are reopened by :meth:`load`.

        Parameters
        ----------
        directory : str or Path
            Destination directory (created if missing).
        chunk_size : int, optional
            Cutouts per chunk file; defaults to this store's chunk size.

        Returns
        -------
        list of int
            The number of cutouts in each written chunk, in order.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        size = max(1, int(chunk_size if chunk_size is not None else self._chunk_size))
        lens: list[int] = []
        buffer: list[Cutout] = []
        for cut in self:
            buffer.append(cut)
            if len(buffer) >= size:
                _write_chunk(
                    directory / f"chunk_{len(lens):06d}.npz", buffer, self.cutout_size
                )
                lens.append(len(buffer))
                buffer = []
        if buffer:
            _write_chunk(
                directory / f"chunk_{len(lens):06d}.npz", buffer, self.cutout_size
            )
            lens.append(len(buffer))
        return lens

    @classmethod
    def load(
        cls,
        directory: str | Path,
        cutout_size: int,
        *,
        max_in_memory: int | None = None,
        spill_dir: str | Path | None = None,
    ) -> "CutoutStore":
        """Open a store backed by persisted chunks written by :meth:`save`.

        The chunks stay on disk and are read lazily (one chunk resident at a
        time), so a large cache is never fully materialised. The returned store
        does **not** own ``directory``: :meth:`close` and the finalizer backstop
        never delete it. ``max_in_memory`` / ``spill_dir`` configure only the
        *owned* stores that :meth:`subset` derives from it.

        Parameters
        ----------
        directory : str or Path
            Directory of ``chunk_*.npz`` files written by :meth:`save`.
        cutout_size : int
            Edge length of the stored cutouts.
        max_in_memory, spill_dir
            Spill configuration inherited by stores derived via :meth:`subset`.

        Returns
        -------
        CutoutStore
        """
        directory = Path(directory)
        store = cls(cutout_size, max_in_memory=max_in_memory, spill_dir=spill_dir)
        store._owns_dir = False
        for path in sorted(directory.glob("chunk_*.npz")):
            with np.load(path) as npz:
                length = int(npz["origin"].shape[0])  # reads only the tiny origin array
            store._chunks.append(path)
            store._chunk_lens.append(length)
            store._n += length
        if store._chunks:
            store._dir = directory
        return store

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Remove the spill directory (idempotent); the store is unusable after."""
        self._closed = True
        self._buffer = []
        self._loaded = None
        self._loaded_index = -1
        if self._owns_dir and self._finalizer is not None:
            self._finalizer()  # runs _cleanup_dir once; subsequent calls are no-ops
        self._dir = None
        self._chunks = []
        self._chunk_lens = []

    def __enter__(self) -> "CutoutStore":
        """Enter a context that :meth:`close`-s the store on exit."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the store, removing any spilled chunks."""
        self.close()
