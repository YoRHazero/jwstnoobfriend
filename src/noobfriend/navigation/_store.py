"""Byte sources that back a NooBook's lazy data access.

A NooBook never holds its file's bytes itself; it reads through a
:class:`ByteStore` each time. The store exposes whole-file, ranged and trailing
reads, so a book can pull just one extension (or just the metadata tail) instead
of the whole file. A standalone book reads directly (uncached) via a plain
:class:`~noobfriend.core.io.ByteAccessor`; a book owned by a
:class:`~noobfriend.navigation.NooBox` reads through the box's shared
:class:`LruByteStore`, which caches results under a bounded byte budget and -- for
a remote host that turns out not to support ranged reads -- transparently falls
back to fetching the whole file and slicing it.
"""

from collections import OrderedDict
from io import BytesIO
from typing import BinaryIO, Protocol

from noobfriend.core.io import (
    RemoteReadError,
    fetch_bytes,
    open_accessor,
)
from noobfriend.core.io.remote import _parse_spec, range_capable, set_range_capable

#: Default ceiling for :class:`LruByteStore`, in bytes (2 GiB).
_DEFAULT_MAX_BYTES: int = 2 * 1024**3


class ByteStore(Protocol):
    """A source of raw file bytes, keyed by a NooBook id.

    Implementations decide whether and how to cache; a NooBook calls these
    whenever it needs to (re-)read part or all of its file.
    """

    def fetch(self, book_id: str, location: str) -> bytes:
        """Return the full bytes of the file at ``location`` for ``book_id``."""
        ...

    def fetch_range(
        self, book_id: str, location: str, offset: int, length: int
    ) -> bytes:
        """Return ``length`` bytes at ``offset`` of ``book_id``'s file."""
        ...

    def fetch_tail(self, book_id: str, location: str, length: int) -> bytes:
        """Return the last ``length`` bytes of ``book_id``'s file."""
        ...


class StoreAccessor:
    """Adapt a :class:`ByteStore` + one book to the ``ByteAccessor`` interface.

    Lets the stateless FITS readers drive a book's cached, shared store through
    the same ``open`` / ``read_range`` / ``read_tail`` verbs they use for any
    other accessor.

    Parameters
    ----------
    store : ByteStore
        The backing store (typically a box's :class:`LruByteStore`).
    book_id : str
        The requesting book's id, used as the store's cache key.
    location : str
        The book's ``[user@]host:path`` or local-path spec.
    """

    def __init__(self, store: ByteStore, book_id: str, location: str) -> None:
        """See the class docstring for parameters."""
        self._store = store
        self._book_id = book_id
        self._location = location

    def open(self) -> BinaryIO:
        """Return the whole file as an in-memory stream (cached by the store)."""
        return BytesIO(self._store.fetch(self._book_id, self._location))

    def read_range(self, offset: int, length: int) -> bytes:
        """Return ``length`` bytes at ``offset``, through the store's cache."""
        return self._store.fetch_range(self._book_id, self._location, offset, length)

    def read_tail(self, length: int) -> bytes:
        """Return the file's last ``length`` bytes, through the store's cache."""
        return self._store.fetch_tail(self._book_id, self._location, length)


class LruByteStore:
    """An in-memory LRU cache of file byte blobs, bounded by total size.

    Suitable as the shared byte cache owned by a
    :class:`~noobfriend.navigation.NooBox`: every member book reads through one
    instance, so total resident bytes stay capped at ``max_bytes``. Whole-file,
    ranged and trailing reads are cached under distinct keys sharing one budget;
    the least-recently-used entries are evicted once a new entry would exceed the
    ceiling. Not thread-safe.

    Parameters
    ----------
    max_bytes : int, optional
        Soft ceiling on the total cached bytes. A single blob larger than
        ``max_bytes`` is still cached (then immediately evictable).
    """

    def __init__(self, max_bytes: int = _DEFAULT_MAX_BYTES) -> None:
        """See the class docstring for parameters."""
        self._max_bytes: int = max_bytes
        self._entries: OrderedDict[str, bytes] = OrderedDict()
        self._total_bytes: int = 0

    def fetch(self, book_id: str, location: str) -> bytes:
        """Return cached whole-file bytes for ``book_id``, reading on a miss."""
        return self._cached(book_id, lambda: fetch_bytes(location))

    def fetch_range(
        self, book_id: str, location: str, offset: int, length: int
    ) -> bytes:
        """Return cached ``[offset, offset+length)`` bytes, reading on a miss."""
        key = f"{book_id}#r:{offset}:{length}"
        return self._cached(
            key, lambda: self._read_range(book_id, location, offset, length)
        )

    def fetch_tail(self, book_id: str, location: str, length: int) -> bytes:
        """Return cached trailing ``length`` bytes, reading on a miss."""
        key = f"{book_id}#t:{length}"
        return self._cached(key, lambda: self._read_tail(book_id, location, length))

    def _read_range(
        self, book_id: str, location: str, offset: int, length: int
    ) -> bytes:
        """Read a byte range, falling back to whole-file-and-slice if unsupported."""
        host, _ = _parse_spec(location)
        if host is not None and not range_capable(host):
            return self.fetch(book_id, location)[offset : offset + length]
        try:
            return open_accessor(location).read_range(offset, length)
        except RemoteReadError:
            whole = self.fetch(book_id, location)
            if host is not None:
                set_range_capable(host, False)
            return whole[offset : offset + length]

    def _read_tail(self, book_id: str, location: str, length: int) -> bytes:
        """Read the trailing bytes, falling back to whole-file-and-slice."""
        host, _ = _parse_spec(location)
        if host is not None and not range_capable(host):
            whole = self.fetch(book_id, location)
            return whole[max(0, len(whole) - length) :]
        try:
            return open_accessor(location).read_tail(length)
        except RemoteReadError:
            whole = self.fetch(book_id, location)
            if host is not None:
                set_range_capable(host, False)
            return whole[max(0, len(whole) - length) :]

    def _cached(self, key: str, read) -> bytes:
        """Return the cached blob for ``key``, calling ``read`` on a miss."""
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached
        raw = read()
        self._insert(key, raw)
        return raw

    def _insert(self, key: str, raw: bytes) -> None:
        """Insert ``raw`` under ``key`` and evict until within budget."""
        self._entries[key] = raw
        self._total_bytes += len(raw)
        while self._total_bytes > self._max_bytes and len(self._entries) > 1:
            _, evicted = self._entries.popitem(last=False)
            self._total_bytes -= len(evicted)
