"""The accumulated detection table behind :class:`SourceExtractor`.

:class:`SourceExtractor` used to carry eight hand-aligned parallel lists -- a
list of per-frame catalogues, the physical-source ``index``, three provenance
label lists (``filename`` / ``filter`` / ``detector``), and a per-source
sky-position accumulator (``sum_ra`` / ``sum_dec`` / ``count``) -- and every
method that added, subset, saved, or loaded had to keep all eight in lockstep by
hand. :class:`_Detections` owns them as one structure so that alignment
invariant lives in a single place: the extractor keeps just a ``_Detections`` and
a cutout store.

The detection rows are held columnar (the existing :class:`SourceCatalog`), with
the ``index`` and the three per-frame labels alongside. The sky-position
accumulator is a denormalised running mean kept incrementally (``O(1)`` per
detection) purely to make cross-frame sky matching cheap during accumulation; it
is rebuilt from the catalogue on :meth:`from_table` and :meth:`subset`, never
persisted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from noobfriend.core.imgutils import SourceCatalog

if TYPE_CHECKING:
    from astropy.table import Table


def _labels_to_column(labels: list[object]) -> object:
    """Encode object labels as a masked string column (``None`` -> masked).

    Labels are stored as strings; ``None`` becomes a masked entry, restored by
    :func:`_labels_from_column`. A minimum width of one keeps a column that is
    entirely ``None`` writable as a FITS string.
    """
    from astropy.table import MaskedColumn

    mask = [v is None for v in labels]
    data = np.array(["" if v is None else str(v) for v in labels], dtype=str)
    if data.dtype.itemsize == 0:
        data = data.astype("U1")
    return MaskedColumn(data=data, mask=mask)


def _labels_from_column(column: object) -> list[object]:
    """Decode a masked string column back to labels (masked -> ``None``).

    FITS character columns round-trip as byte strings, so unmasked values are
    decoded to ``str``.
    """
    values = np.asarray(column)
    mask = np.asarray(getattr(column, "mask", np.zeros(values.shape, dtype=bool)))
    return [
        None
        if is_masked
        else (v.decode() if isinstance(v, bytes | np.bytes_) else str(v))
        for v, is_masked in zip(values, mask)
    ]


class _Detections:
    """One aligned detection table: catalogue rows + index + labels (+ position cache).

    Row ``j`` of :attr:`catalog` corresponds to ``index[j]`` (its physical-source
    id) and to the ``j``-th entry of :attr:`filenames` / :attr:`filters` /
    :attr:`detectors`. The cutout store held separately by the extractor is
    aligned the same way.
    """

    __slots__ = (
        "_parts",
        "_index",
        "_filename",
        "_filter",
        "_detector",
        "_sum_ra",
        "_sum_dec",
        "_count",
    )

    def __init__(self) -> None:
        """Create an empty detection table."""
        self._parts: list[SourceCatalog] = []
        self._index: list[int] = []
        self._filename: list[object] = []
        self._filter: list[object] = []
        self._detector: list[object] = []
        # running per-physical-source sky position (sum / count by index)
        self._sum_ra: list[float] = []
        self._sum_dec: list[float] = []
        self._count: list[int] = []

    # -- accumulation ---------------------------------------------------------

    def new_index(self) -> int:
        """Reserve and return a fresh physical-source index."""
        index = len(self._count)
        self._sum_ra.append(0.0)
        self._sum_dec.append(0.0)
        self._count.append(0)
        return index

    def append_frame(
        self,
        rows: SourceCatalog,
        index: list[int],
        *,
        filename: object,
        filter: object,  # noqa: A002 -- the JWST band label, an astro convention
        detector: object,
    ) -> None:
        """Append one frame's kept detections (their ``index`` already assigned).

        ``filename`` / ``filter`` / ``detector`` are per-frame and stored once per
        row. The sky-position accumulator is updated from ``rows.ra`` / ``rows.dec``
        for each detection's (possibly pre-existing) ``index``.
        """
        n = len(index)
        if n == 0:
            return
        self._parts.append(rows)
        self._index.extend(index)
        self._filename.extend([filename] * n)
        self._filter.extend([filter] * n)
        self._detector.extend([detector] * n)
        for k, ra, dec in zip(index, rows.ra, rows.dec):
            self._sum_ra[k] += float(ra)
            self._sum_dec[k] += float(dec)
            self._count[k] += 1

    def source_means(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the per-source mean ``(ra, dec)`` arrays, aligned to source index."""
        count = np.asarray(self._count, dtype=float)
        return np.asarray(self._sum_ra) / count, np.asarray(self._sum_dec) / count

    # -- selection ------------------------------------------------------------

    def subset(self, rows: np.ndarray) -> "_Detections":
        """Return a new table over detection rows ``rows`` (source ids re-compacted)."""
        sub = _Detections()
        if rows.size == 0:
            return sub
        _, compact = np.unique(self.index[rows], return_inverse=True)
        catalogue = self.catalog[rows]
        sub._parts = [catalogue]
        sub._index = compact.tolist()
        sub._filename = [self._filename[r] for r in rows]
        sub._filter = [self._filter[r] for r in rows]
        sub._detector = [self._detector[r] for r in rows]
        sub._rebuild_positions(compact, catalogue)
        return sub

    def _rebuild_positions(self, index: np.ndarray, catalogue: SourceCatalog) -> None:
        """Recompute the running sky-position accumulator from a catalogue + index."""
        n_sources = int(index.max()) + 1 if index.size else 0
        self._sum_ra = [0.0] * n_sources
        self._sum_dec = [0.0] * n_sources
        self._count = [0] * n_sources
        for j, k in enumerate(index.tolist()):
            self._sum_ra[k] += float(catalogue.ra[j])
            self._sum_dec[k] += float(catalogue.dec[j])
            self._count[k] += 1

    # -- persistence ----------------------------------------------------------

    def to_table(self) -> "Table":
        """Return the catalogue + ``index`` + label columns as one astropy table."""
        table = self.catalog.to_table()
        table["index"] = self.index
        table["filename"] = _labels_to_column(self._filename)
        table["filter"] = _labels_to_column(self._filter)
        table["detector"] = _labels_to_column(self._detector)
        return table

    @classmethod
    def from_table(cls, table: "Table") -> "_Detections":
        """Rebuild a detection table from one written by :meth:`to_table`."""
        det = cls()
        catalogue = SourceCatalog.from_table(table)
        index = np.asarray(table["index"], dtype=int)
        det._parts = [catalogue] if len(catalogue) else []
        det._index = index.tolist()
        det._filename = _labels_from_column(table["filename"])
        det._filter = _labels_from_column(table["filter"])
        det._detector = _labels_from_column(table["detector"])
        det._rebuild_positions(index, catalogue)
        return det

    # -- accessors ------------------------------------------------------------

    @property
    def catalog(self) -> SourceCatalog:
        """The accumulated detection catalogue (one row per kept detection)."""
        return SourceCatalog.concat(self._parts)

    @property
    def index(self) -> np.ndarray:
        """Physical-source index of each detection (groups rows across frames)."""
        return np.asarray(self._index, dtype=int)

    @property
    def filenames(self) -> list[object]:
        """Per-detection provenance ``filename`` labels."""
        return self._filename

    @property
    def filters(self) -> list[object]:
        """Per-detection ``filter`` (band) labels."""
        return self._filter

    @property
    def detectors(self) -> list[object]:
        """Per-detection ``detector`` labels."""
        return self._detector

    @property
    def n_sources(self) -> int:
        """Number of distinct physical sources collected so far."""
        return len(self._count)

    def __len__(self) -> int:
        """Return the number of detections collected so far."""
        return len(self._index)
