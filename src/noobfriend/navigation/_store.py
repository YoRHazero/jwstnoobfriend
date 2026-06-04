"""Byte sources that back a NooBook's lazy data access.

A NooBook never holds its file's bytes itself; it asks a :class:`ByteStore`
each time it needs to read. A standalone book reads directly (uncached), while
a book owned by a :class:`~noobfriend.navigation.NooBox` reads through the
box's shared :class:`LruByteStore`, so the resident pixel bytes across a whole
collection stay bounded no matter how many files are touched.
"""

from collections import OrderedDict
from typing import Protocol

from noobfriend.core.io import fetch_bytes

#: Default ceiling for :class:`LruByteStore`, in bytes (2 GiB).
_DEFAULT_MAX_BYTES: int = 2 * 1024**3


class ByteStore(Protocol):
    """A source of raw file bytes, keyed by a NooBook id.

    Implementations decide whether and how to cache; a NooBook calls
    :meth:`fetch` whenever it needs to (re-)read its file.
    """

    def fetch(self, book_id: str, location: str) -> bytes:
        """Return the full bytes of the file at ``location`` for ``book_id``.

        Parameters
        ----------
        book_id : str
            Identity of the requesting NooBook, usable as a cache key.
        location : str
            The file's ``[user@]host:path`` or local-path spec, used to read
            the bytes on a cache miss.

        Returns
        -------
        bytes
            The complete file content.
        """
        ...


class LruByteStore:
    """An in-memory LRU cache of raw file bytes, bounded by total size.

    Suitable as the shared byte cache owned by a
    :class:`~noobfriend.navigation.NooBox`: every member book reads through one
    instance, so total resident bytes stay capped at ``max_bytes``. The
    least-recently-used entries are evicted once a new entry would exceed the
    ceiling. Not thread-safe.

    Parameters
    ----------
    max_bytes : int, optional
        Soft ceiling on the total cached bytes. A single file larger than
        ``max_bytes`` is still cached (then immediately evictable).
    """

    def __init__(self, max_bytes: int = _DEFAULT_MAX_BYTES) -> None:
        """See the class docstring for parameters."""
        self._max_bytes: int = max_bytes
        self._entries: OrderedDict[str, bytes] = OrderedDict()
        self._total_bytes: int = 0

    def fetch(self, book_id: str, location: str) -> bytes:
        """Return cached bytes for ``book_id``, reading ``location`` on a miss."""
        cached = self._entries.get(book_id)
        if cached is not None:
            self._entries.move_to_end(book_id)
            return cached
        raw = fetch_bytes(location)
        self._insert(book_id, raw)
        return raw

    def _insert(self, book_id: str, raw: bytes) -> None:
        """Insert ``raw`` under ``book_id`` and evict until within budget."""
        self._entries[book_id] = raw
        self._total_bytes += len(raw)
        while self._total_bytes > self._max_bytes and len(self._entries) > 1:
            _, evicted = self._entries.popitem(last=False)
            self._total_bytes -= len(evicted)
