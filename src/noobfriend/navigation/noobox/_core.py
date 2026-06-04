"""The :class:`NooBox`: a managed collection of NooBooks over a shared cache.

The collection container, lineage reverse-index, shared byte cache, directory
discovery (local or remote over SSH), subsetting, merging and JSON save / load
are all implemented here.
"""

import fnmatch
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path

from noobfriend.core.env import get_settings, stage_path_var
from noobfriend.core.io import list_remote_dir
from noobfriend.navigation._store import ByteStore, LruByteStore
from noobfriend.navigation.noobook import NooBook
from noobfriend.navigation.noobox._viz import BoxViz


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


def _resolve_stage_root(stage: str, roots: dict[str, str] | None) -> str:
    """Resolve a stage's directory root from ``roots`` then the environment.

    Parameters
    ----------
    stage : str
        Stage label whose directory to resolve.
    roots : dict or None
        Explicit ``{stage: root}`` overrides; consulted first, per stage.

    Returns
    -------
    str
        The directory root for ``stage`` (possibly a ``host:path`` spec).

    Raises
    ------
    ValueError
        If ``stage`` is in neither ``roots`` nor the ``STAGE_<STAGE>_PATH``
        environment variable.
    """
    if roots is not None and stage in roots:
        return roots[stage]
    env_value = os.getenv(stage_path_var(stage))
    if env_value is None:
        raise ValueError(
            f"no root for stage {stage!r}: not in `roots` and "
            f"{stage_path_var(stage)} is unset."
        )
    return env_value


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


class NooBox:
    """A collection of :class:`NooBook`, sharing one bounded byte cache.

    NooBox is the runtime owner of the shared :class:`ByteStore`: every NooBook
    added to it reads its heavy data through the same LRU cache, so the total
    resident pixel bytes stay bounded across the whole collection. It also
    inverts each book's ``parent_ids`` into a children lookup, since a
    product's downstream consumers are a property of the collection, not of the
    file.

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
        stages: str | list[str],
        *,
        roots: dict[str, str] | None = None,
        wildcard: str = "*.fits",
        probe: bool = False,
        max_cache_bytes: int | None = None,
    ) -> "NooBox":
        """Discover products under one or more stage directories.

        For each stage, the directory root is taken from ``roots`` if present,
        otherwise from the ``STAGE_<STAGE>_PATH`` environment variable; files in
        it matching ``wildcard`` become books. By default the books are *thin*
        (built with :meth:`NooBook.from_name`, no file opened); pass
        ``probe=True`` to read each file's header immediately, or call
        :meth:`probe` later.

        Parameters
        ----------
        stages : str or list of str
            Stage label(s) to discover.
        roots : dict, optional
            Explicit ``{stage: directory}`` overrides; a directory may be a
            local path or a ``host:path`` spec. Per stage, this wins over the
            environment.
        wildcard : str, optional
            Glob pattern selecting product files in each root, by default
            ``"*.fits"``.
        probe : bool, optional
            When ``True``, populate header-derived fields by reading each file
            now (slower); when ``False`` (default), build thin books.
        max_cache_bytes : int or None, optional
            Forwarded to the new box's :class:`LruByteStore`.

        Returns
        -------
        NooBox

        Raises
        ------
        ValueError
            If a stage's root cannot be resolved, or a local root is not a
            directory.
        RemoteReadError
            If a remote ``host:path`` root cannot be listed over SSH.
        """
        stage_list = [stages] if isinstance(stages, str) else list(stages)
        get_settings()  # load layered .env so STAGE_<STAGE>_PATH is visible

        box = cls(max_cache_bytes=max_cache_bytes)
        for stage in stage_list:
            root = _resolve_stage_root(stage, roots)
            for location in _discover_locations(root, wildcard):
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

    # -- views and combination ------------------------------------------------

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
        subset = NooBox()
        subset._store = self._store
        subset._books = {
            book_id: book for book_id, book in self._books.items() if predicate(book)
        }
        return subset

    def merge(self, other: "NooBox", *, overwrite: bool = False) -> "NooBox":
        """Combine this box and ``other`` into a new, independent box.

        The result has its own fresh byte cache and holds copies of both
        inputs' books (the inputs are left untouched).

        Parameters
        ----------
        other : NooBox
            The box to merge in.
        overwrite : bool, optional
            On an id present in both boxes: when ``True``, ``other`` wins; when
            ``False`` (default), a :class:`ValueError` is raised.

        Returns
        -------
        NooBox
            A new box containing the union of both collections.

        Raises
        ------
        ValueError
            If an id occurs in both boxes and ``overwrite`` is ``False``.
        """
        merged = NooBox()
        for source in (self, other):
            for book in source:
                if not overwrite and book.id in merged:
                    raise ValueError(
                        f"duplicate id {book.id!r} while merging; "
                        "pass overwrite=True to allow."
                    )
                merged.add(book.model_copy())
        return merged

    # -- lineage --------------------------------------------------------------

    def get(self, book_id: str) -> NooBook | None:
        """Return the book with ``book_id``, or ``None`` if absent."""
        return self._books.get(book_id)

    def children(self, book_id: str) -> list[NooBook]:
        """Return the books that list ``book_id`` among their parents.

        The downstream (child) links are derived on demand by inverting every
        member's :attr:`~NooBook.parent_ids`, so they always reflect the
        current collection.

        Parameters
        ----------
        book_id : str
            Id of the (upstream) product whose children to find.

        Returns
        -------
        list of NooBook
            Members whose ``parent_ids`` include ``book_id``.
        """
        return [book for book in self._books.values() if book_id in book.parent_ids]

    def parents(self, book_id: str) -> list[NooBook]:
        """Return the in-collection parent books of ``book_id``."""
        book = self._books.get(book_id)
        if book is None:
            return []
        return [self._books[pid] for pid in book.parent_ids if pid in self._books]

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

    def __getitem__(self, book_id: str) -> NooBook:
        """Return the book with ``book_id`` (raising ``KeyError`` if absent)."""
        return self._books[book_id]

    def __contains__(self, book_id: object) -> bool:
        """Whether a book with ``book_id`` is in the collection."""
        return book_id in self._books

    def __iter__(self) -> Iterator[NooBook]:
        """Iterate over the collection's books."""
        return iter(self._books.values())

    def __len__(self) -> int:
        """Return the number of books in the collection."""
        return len(self._books)
