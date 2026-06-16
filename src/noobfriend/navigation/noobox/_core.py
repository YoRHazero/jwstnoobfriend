"""The :class:`NooBox`: a managed collection of NooBooks over a shared cache.

The collection container, shared byte cache, single-stage directory discovery
(local or remote over SSH), lineage-resolving assembly (:meth:`NooBox.merge`),
subsetting / grouping / selection and JSON save / load are all implemented
here.
"""

import fnmatch
import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from noobfriend.core.display import track
from noobfriend.core.env import get_settings, stage_path_var
from noobfriend.core.io import list_remote_dir
from noobfriend.navigation._store import ByteStore, LruByteStore
from noobfriend.navigation.noobook import NooBook
from noobfriend.navigation.noobox._extract import BoxExtract
from noobfriend.navigation.noobox._viz import BoxViz

if TYPE_CHECKING:
    from astropy.table import Table

#: The cross-stage identity key a product shares with its same-exposure kin.
_MatchKey = tuple[
    str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], str | None
]


def _is_remote_root(root: str) -> bool:
    """Return whether ``root`` is a remote ``[user@]host:path`` spec.

    Uses the ``scp`` rule: remote only when a ``:`` appears before any ``/``.
    """
    colon = root.find(":")
    slash = root.find("/")
    return colon != -1 and (slash == -1 or colon < slash)


def _discover_locations(root: str, wildcard: str) -> list[str]:
    """Return the ``wildcard``-matching file locations directly under ``root``.

    Works for a local directory or a remote ``host:path`` directory. Remote
    listing is done over SSH (:func:`noobfriend.core.io.list_remote_dir`) and
    matched against ``wildcard`` with the same :mod:`fnmatch` semantics that
    :meth:`pathlib.Path.glob` uses locally, so both paths behave alike.

    Parameters
    ----------
    root : str
        A local directory path or a remote ``host:path`` directory spec.
    wildcard : str
        Glob pattern selecting file names in ``root`` (non-recursive).

    Returns
    -------
    list of str
        Full locations (local paths, or ``host:path`` specs), sorted.

    Raises
    ------
    ValueError
        ``root`` is local but is not an existing directory.
    RemoteReadError
        ``root`` is remote and the SSH listing failed.
    """
    if _is_remote_root(root):
        names = list_remote_dir(root)
        base = root.rstrip("/")
        return [f"{base}/{name}" for name in sorted(fnmatch.filter(names, wildcard))]

    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"root is not a directory: {root}")
    return [str(path) for path in sorted(directory.glob(wildcard))]


def _resolve_stage_root(stage: str, root: str | None) -> str:
    """Resolve a stage's directory root from ``root`` then the environment.

    With an explicit ``root`` that value is used verbatim. Otherwise the stage's
    ``STAGE_<STAGE>_PATH`` is read from the environment and, following the same
    storage model as ``fetch`` and ``env check``, composed with ``NOOB_SERVER``:
    a remote ``NOOB_SERVER`` makes it a ``host:path`` spec read over SSH, while a
    local ``NOOB_SERVER`` leaves the bare path.

    Parameters
    ----------
    stage : str
        Stage label whose directory to resolve.
    root : str or None
        Explicit override; used as-is when given (local path or ``host:path``).

    Returns
    -------
    str
        The directory root for ``stage`` -- a local path or a ``host:path`` spec.

    Raises
    ------
    ValueError
        If ``root`` is ``None`` and the ``STAGE_<STAGE>_PATH`` environment
        variable is unset.
    """
    if root is not None:
        return root
    stage_path = os.getenv(stage_path_var(stage))
    if stage_path is None:
        raise ValueError(
            f"no root for stage {stage!r}: `root` is None and "
            f"{stage_path_var(stage)} is unset."
        )
    host = get_settings().server_host()
    return stage_path if host is None else f"{host}:{stage_path}"


def _resolve_noobox_path(path: Path | str | None) -> Path:
    """Resolve a save / load path, falling back to ``NOOBOX_PATH``.

    Raises
    ------
    ValueError
        If ``path`` is ``None`` and ``noobox_path`` is unset in the environment.
    """
    if path is not None:
        return Path(path)
    configured = get_settings().noobox_path
    if configured is None:
        raise ValueError("no path given and NOOBOX_PATH is not set in the environment.")
    return configured


def _match_key(book: NooBook) -> _MatchKey:
    """Return the cross-stage identity key of ``book``.

    Products of the same physical exposure(s) share this key across pipeline
    stages, so it is what lineage resolution matches on. The product suffix and
    stage are deliberately excluded; the detector is included, so per-detector
    products stay distinct.
    """
    return (
        book.program_id,
        book.observation,
        book.visit,
        book.ggsaa,
        book.exposure,
        book.detector,
    )


def _matches(book: NooBook, criteria: dict[str, object]) -> bool:
    """Return whether ``book`` satisfies every ``field=value`` in ``criteria``.

    A string value is an :mod:`fnmatch` glob, a list / tuple / set value matches
    by membership, and anything else by equality. Tuple-valued attributes match
    when any element matches. See :meth:`NooBox.select`.
    """
    for field, value in criteria.items():
        actual = getattr(book, field)
        candidates = actual if isinstance(actual, tuple) else (actual,)
        if isinstance(value, str):
            ok = any(
                c is not None and fnmatch.fnmatch(str(c), value) for c in candidates
            )
        elif isinstance(value, (list, tuple, set, frozenset)):
            ok = any(c in value for c in candidates)
        else:
            ok = any(c == value for c in candidates)
        if not ok:
            return False
    return True


class NooBox:
    """A collection of :class:`NooBook`, sharing one bounded byte cache.

    NooBox is the runtime owner of the shared :class:`ByteStore`: every NooBook
    added to it reads its heavy data through the same LRU cache, so the total
    resident pixel bytes stay bounded across the whole collection.

    A box is assembled from single-stage boxes (:meth:`from_directory`) stitched
    together with :meth:`merge`, which resolves the cross-stage lineage as it
    goes. Subsetting (:meth:`filter`, :meth:`select`), grouping
    (:meth:`group_by`, :meth:`distinct`) and the lineage boundary
    (:meth:`leaves`, :meth:`roots`) all return lightweight views sharing this
    box's cache.

    Parameters
    ----------
    max_cache_bytes : int or None, optional
        Ceiling on the shared byte cache; forwarded to :class:`LruByteStore`
        when given, otherwise the store's own default is used.
    """

    def __init__(self, *, max_cache_bytes: int | None = None) -> None:
        """See the class docstring for parameters."""
        self._books: dict[str, NooBook] = {}
        self._store: ByteStore = (
            LruByteStore(max_cache_bytes)
            if max_cache_bytes is not None
            else LruByteStore()
        )

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        stage: str,
        root: str | None = None,
        *,
        wildcard: str = "*.fits",
        probe: bool = False,
        max_cache_bytes: int | None = None,
    ) -> "NooBox":
        """Discover one stage's products under a single directory.

        Every ``wildcard``-matching file directly under the directory becomes a
        book of ``stage``; the result is therefore a *single-stage* box, the unit
        you assemble a pipeline ladder from with :meth:`merge`. The directory is
        ``root`` when given, else ``STAGE_<STAGE>_PATH`` composed with
        ``NOOB_SERVER`` (a remote ``NOOB_SERVER`` makes it a ``host:path`` read
        over SSH, the same place ``fetch`` downloads to). By default the books
        are *thin* (:meth:`NooBook.from_name`, no file opened); pass
        ``probe=True`` to read each header now, or call :meth:`probe` later.

        Parameters
        ----------
        stage : str
            Stage label assigned to every discovered file.
        root : str, optional
            Directory to scan -- a local path or a ``host:path`` spec. Defaults
            to ``NOOB_SERVER`` + ``STAGE_<STAGE>_PATH``.
        wildcard : str, optional
            Glob pattern selecting product files, by default ``"*.fits"``.
        probe : bool, optional
            When ``True``, populate header-derived fields by reading each file
            now (slower, shown with a progress bar); when ``False`` (default),
            build thin books.
        max_cache_bytes : int or None, optional
            Forwarded to the new box's :class:`LruByteStore`.

        Returns
        -------
        NooBox

        Raises
        ------
        ValueError
            If the root cannot be resolved, or a local root is not a directory.
        RemoteReadError
            If a remote ``host:path`` root cannot be listed over SSH.
        """
        get_settings()  # load layered .env so STAGE_<STAGE>_PATH is visible
        resolved = _resolve_stage_root(stage, root)

        box = cls(max_cache_bytes=max_cache_bytes)
        locations = _discover_locations(resolved, wildcard)
        tracked = track(locations, f"Probing {stage} headers") if probe else locations
        for location in tracked:
            book = (
                NooBook.from_file(location, stage, store=box._store)
                if probe
                else NooBook.from_name(location, stage)
            )
            box.add(book)
        return box

    @classmethod
    def load(
        cls, path: Path | str | None = None, *, max_cache_bytes: int | None = None
    ) -> "NooBox":
        """Rebuild a box from a JSON manifest written by :meth:`save`.

        The manifest round-trips each book's ``parent_ids``, so lineage stamped
        at production time (e.g. on stage-3 aggregates) comes back intact -- this
        is the path by which reduction-produced products re-enter a box with
        their lineage, as opposed to a bare :meth:`from_directory` scan.

        Parameters
        ----------
        path : Path or str, optional
            Manifest file to read; defaults to ``NOOBOX_PATH``.
        max_cache_bytes : int or None, optional
            Forwarded to the new box's :class:`LruByteStore`.

        Returns
        -------
        NooBox
        """
        source = _resolve_noobox_path(path)
        records = json.loads(source.read_text())
        box = cls(max_cache_bytes=max_cache_bytes)
        for record in records:
            box.add(NooBook.from_record(record))
        return box

    # -- mutation -------------------------------------------------------------

    def add(self, book: NooBook) -> NooBook:
        """Add ``book`` to the collection, binding it to the shared cache.

        Parameters
        ----------
        book : NooBook
            The book to add. Re-adding an existing id replaces it.

        Returns
        -------
        NooBook
            The added book (now bound to this box's byte store).
        """
        book._bind_store(self._store)
        self._books[book.id] = book
        return book

    def probe(self) -> "NooBox":
        """Populate every thin member by reading its file; returns ``self``.

        Each unprobed book is replaced in place by its
        :meth:`NooBook.probe` result (read through the shared cache), so
        afterwards every member carries ``pupil`` / ``filter`` / ``shape`` /
        ``footprint``. Already-probed books are left untouched.

        Returns
        -------
        NooBox
            This box, for chaining.
        """
        for book in list(self._books.values()):
            if not book.is_probed:
                self._books[book.id] = book.probe(store=self._store)
        return self

    # -- views, grouping and lineage boundary ---------------------------------

    def _view(self, books: Iterable[NooBook]) -> "NooBox":
        """Return a sub-box over ``books``, sharing this box's byte cache."""
        sub = NooBox()
        sub._store = self._store
        sub._books = {book.id: book for book in books}
        return sub

    def filter(self, predicate: Callable[[NooBook], bool]) -> "NooBox":
        """Return a sub-box of the members satisfying ``predicate``.

        The result is a lightweight *view*: it shares this box's byte cache and
        the same :class:`NooBook` instances, so cached bytes are reused and no
        files are re-read.

        Parameters
        ----------
        predicate : callable
            ``predicate(book) -> bool``; books for which it returns ``True`` are
            kept.

        Returns
        -------
        NooBox
            A new box over the matching books, sharing this box's cache.
        """
        return self._view(book for book in self._books.values() if predicate(book))

    def select(self, **criteria: object) -> "NooBox":
        """Return the sub-box of books matching every ``field=value`` criterion.

        For each criterion the book's attribute is compared to ``value``:

        - a string ``value`` is an :mod:`fnmatch` glob (so ``filter="F2*"``
          works, plain strings match exactly);
        - a list / tuple / set ``value`` matches by membership;
        - any other ``value`` matches by equality.

        Tuple-valued attributes (``observation``, ``visit``, ``ggsaa``,
        ``exposure``) match when *any* element matches. The result is a view
        sharing this box's cache; for arbitrary logic use :meth:`filter`.

        Parameters
        ----------
        **criteria
            ``field=value`` pairs; a book is kept only if it matches all of them.

        Returns
        -------
        NooBox
        """
        return self._view(
            book for book in self._books.values() if _matches(book, criteria)
        )

    def group_by(
        self, key: str | Callable[[NooBook], object]
    ) -> dict[object, "NooBox"]:
        """Partition the collection into sub-boxes keyed by ``key``.

        Parameters
        ----------
        key : str or callable
            A book attribute name (e.g. ``"stage"``, ``"filter"``) or a function
            ``key(book) -> value``. Books sharing a key value land in the same
            group.

        Returns
        -------
        dict
            ``{key_value: NooBox}``; each sub-box is a view sharing this box's
            cache.
        """
        keyfn = key if callable(key) else (lambda book: getattr(book, key))
        buckets: dict[object, list[NooBook]] = {}
        for book in self._books.values():
            buckets.setdefault(keyfn(book), []).append(book)
        return {value: self._view(books) for value, books in buckets.items()}

    def distinct(self, field: str) -> set[Any]:
        """Return the set of distinct values of ``field`` across the members.

        Parameters
        ----------
        field : str
            Name of a (hashable) :class:`NooBook` attribute, e.g. ``"filter"``,
            ``"detector"`` or ``"stage"``.

        Returns
        -------
        set
        """
        return {getattr(book, field) for book in self._books.values()}

    def leaves(self) -> "NooBox":
        """Return the sub-box of books that nothing in the collection consumes.

        A book is a *leaf* when no member lists its id among its
        :attr:`~NooBook.parent_ids`; these are the current terminal products, and
        the default candidate-parent pool for :meth:`merge`. The result is a view
        sharing this box's cache.

        Returns
        -------
        NooBox
        """
        referenced = {pid for book in self._books.values() for pid in book.parent_ids}
        return self._view(
            book for book in self._books.values() if book.id not in referenced
        )

    def roots(self) -> "NooBox":
        """Return the sub-box of books with no parent present in the collection.

        A book is a *root* here when none of its :attr:`~NooBook.parent_ids`
        resolve to a member -- whether it is a genuine pipeline entry point or its
        upstream simply is not in this box. The result is a view sharing this
        box's cache.

        Returns
        -------
        NooBox
        """
        return self._view(
            book
            for book in self._books.values()
            if not any(pid in self._books for pid in book.parent_ids)
        )

    def copy(self) -> "NooBox":
        """Return an independent deep copy: fresh books over a fresh byte cache.

        Unlike the views from :meth:`filter` / :meth:`select` (which share this
        box's :class:`NooBook` instances and cache), the copy owns
        :meth:`~NooBook.model_copy` books bound to its own :class:`LruByteStore`,
        so mutating it -- assembling new stages, resolving lineage with
        :meth:`merge` -- never touches this box. Pair it with :meth:`select` to
        carve out a standalone sandbox for a reduction experiment.

        Returns
        -------
        NooBox
            A standalone box holding copies of this box's books.
        """
        clone = NooBox()
        for book in self._books.values():
            clone.add(book.model_copy())
        return clone

    # -- assembly -------------------------------------------------------------

    def merge(
        self,
        other: "NooBox",
        *,
        parent: str | None = None,
        child: str | None = None,
        overwrite: bool = False,
    ) -> "NooBox":
        """Absorb ``other`` into this box, resolving the seam lineage in place.

        ``other``'s books are added to this collection (sharing this box's byte
        cache) and the new seam edges are filled by matching identity keys
        (:func:`_match_key`): a book whose ``parent_ids`` is still empty is linked
        to the same-key books in a candidate pool. Books that already carry
        ``parent_ids`` (stamped at production, or resolved by an earlier merge)
        are never re-pointed, so a merge only ever *adds* lineage edges -- it
        cannot re-route an existing one. That is why a stage can be appended or
        branched but never spliced into the interior of an existing chain.

        The candidate pool, hence the direction of the join, is chosen by:

        - default (neither ``parent`` nor ``child``): forward extension -- the
          incoming books' parents are sought among this box's current
          :meth:`leaves`;
        - ``parent="<stage>"``: forward branch -- sought among this box's books
          at ``<stage>`` (use when the attach point is an interior node, e.g. a
          second branch off a node that already has a child);
        - ``child="<stage>"``: backward extension (prepend) -- the incoming books
          are the candidate parents of this box's books at ``<stage>``.

        Matching is exact on the identity key and never links two books at the
        same stage, so steps that do not preserve the key (aggregations,
        recombinations) must instead be stamped at production via
        :meth:`NooBook.from_file`.

        Parameters
        ----------
        other : NooBox
            The box to absorb. Its books are rebound to this box's cache, so
            treat ``other`` as consumed afterwards.
        parent : str, optional
            Restrict the candidate-parent pool to this box's books at the given
            stage (forward branch). Mutually exclusive with ``child``.
        child : str, optional
            Resolve this box's books at the given stage against the incoming
            books (backward extension / prepend). Mutually exclusive with
            ``parent``.
        overwrite : bool, optional
            On an id already present: when ``True``, the incoming book wins; when
            ``False`` (default), a :class:`ValueError` is raised.

        Returns
        -------
        NooBox
            This box, for chaining.

        Raises
        ------
        ValueError
            If both ``parent`` and ``child`` are given, or an id collides while
            ``overwrite`` is ``False``.
        """
        if parent is not None and child is not None:
            raise ValueError("pass at most one of `parent` / `child`.")

        if child is not None:
            children = [book for book in self._books.values() if book.stage == child]
            pool = list(other)
        else:
            pool = (
                [book for book in self._books.values() if book.stage == parent]
                if parent is not None
                else list(self.leaves())
            )
            children = list(other)

        self._absorb(other, overwrite=overwrite)
        self._resolve(children, pool)
        return self

    def _absorb(self, other: "NooBox", *, overwrite: bool) -> None:
        """Add every book of ``other`` to this collection (see :meth:`merge`)."""
        for book in other:
            if not overwrite and book.id in self._books:
                raise ValueError(
                    f"duplicate id {book.id!r} while merging; "
                    "pass overwrite=True to allow."
                )
            self.add(book)

    def _resolve(self, children: list[NooBook], pool: list[NooBook]) -> None:
        """Fill the empty ``parent_ids`` of ``children`` from ``pool`` by key.

        A child keeps any ``parent_ids`` it already has; otherwise it is linked
        to every pool book that shares its identity key and sits at a different
        stage. See :meth:`merge`.
        """
        index: dict[_MatchKey, list[NooBook]] = {}
        for candidate in pool:
            index.setdefault(_match_key(candidate), []).append(candidate)
        for book in children:
            if book.parent_ids:
                continue
            matches = [
                candidate
                for candidate in index.get(_match_key(book), ())
                if candidate.stage != book.stage and candidate.id != book.id
            ]
            if matches:
                book._stamp_parents(
                    tuple(sorted(candidate.id for candidate in matches))
                )

    # -- persistence ----------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        """Write the collection's records to a JSON manifest.

        Parameters
        ----------
        path : Path or str, optional
            Destination file; defaults to ``NOOBOX_PATH``. Parent directories
            are created as needed.

        Returns
        -------
        Path
            The path written to.
        """
        target = _resolve_noobox_path(path)
        records = [book.to_record() for book in self._books.values()]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(records, indent=2))
        return target

    # -- accessors ------------------------------------------------------------

    @property
    def viz(self) -> BoxViz:
        """Plotting sugar bound to this box, e.g. ``box.viz.footprints()``."""
        return BoxViz(self)

    @property
    def extract(self) -> BoxExtract:
        """Extraction sugar bound to this box, e.g. ``box.extract.psf(...)``."""
        return BoxExtract(self)

    def get(self, book_id: str) -> NooBook | None:
        """Return the book with ``book_id``, or ``None`` if absent."""
        return self._books.get(book_id)

    def to_table(self) -> "Table":
        """Return an :class:`astropy.table.Table` summarising the members.

        One row per book, with the identity and resident-metadata fields most
        useful for an overview (tuple fields are joined with ``"+"``; missing
        values render as empty strings). Intended for interactive display, not
        as a serialization format -- use :meth:`save` for that.

        Returns
        -------
        astropy.table.Table
        """
        from astropy.table import Table

        rows = [
            {
                "id": book.id,
                "stage": book.stage,
                "program_id": book.program_id,
                "detector": book.detector or "",
                "pupil": book.pupil or "",
                "filter": book.filter or "",
                "exposure": "+".join(book.exposure),
                "shape": "" if book.shape is None else "×".join(map(str, book.shape)),
                "n_parents": len(book.parent_ids),
            }
            for book in self._books.values()
        ]
        return Table(rows=rows) if rows else Table()

    def __repr__(self) -> str:
        """Return a one-line summary: member count and per-stage breakdown."""
        if not self._books:
            return "NooBox(empty)"
        counts = Counter(book.stage for book in self._books.values())
        breakdown = ", ".join(f"{stage}×{n}" for stage, n in sorted(counts.items()))
        return f"NooBox({len(self)} books; {breakdown})"

    @overload
    def __getitem__(self, key: str | int) -> NooBook: ...
    @overload
    def __getitem__(self, key: slice) -> "NooBox": ...
    def __getitem__(self, key: str | int | slice) -> "NooBook | NooBox":
        """Return a member by id, by position, or a sub-box by slice.

        ``box["<id>@<stage>"]`` looks a book up by its id (``KeyError`` if
        absent); ``box[0]`` / ``box[-1]`` index into the insertion order; and
        ``box[:5]`` slices a positional view (a sub-box sharing this box's
        cache). Ids always contain ``@`` and never look like integers, so the
        key type disambiguates the three forms cleanly.
        """
        if isinstance(key, slice):
            return self._view(list(self._books.values())[key])
        if isinstance(key, int):
            return list(self._books.values())[key]
        return self._books[key]

    def __contains__(self, book_id: object) -> bool:
        """Whether a book with ``book_id`` is in the collection."""
        return book_id in self._books

    def __iter__(self) -> Iterator[NooBook]:
        """Iterate over the collection's books."""
        return iter(self._books.values())

    def __len__(self) -> int:
        """Return the number of books in the collection."""
        return len(self._books)
