"""A uniform, location-transparent handle to one file's bytes.

A :class:`ByteAccessor` hides whether a file is local or remote behind one small
set of verbs -- :meth:`~ByteAccessor.open` (a fresh seekable stream over the whole
file), :meth:`~ByteAccessor.read_range` (an arbitrary byte slice) and
:meth:`~ByteAccessor.read_tail` (the trailing bytes). Callers that know a file's
structure (which bytes hold which FITS extension) drive these verbs to move only
the bytes they need; they never branch on local-versus-remote themselves.

The single local-versus-remote decision lives in :func:`open_accessor`, which
inspects the spec and returns a :class:`LocalAccessor` or :class:`RemoteAccessor`.
Both delegate the partial reads to the location-transparent transport in
:mod:`noobfriend.core.io.remote`; they differ only in :meth:`~ByteAccessor.open`,
where a local file yields a real (lazily readable) file handle and a remote file
yields the whole content wrapped in :class:`io.BytesIO`.

This module is format-agnostic: it knows nothing about FITS. Computing *which*
byte ranges to ask for is the job of the FITS layer
(:mod:`noobfriend.core.io.fits`).
"""

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Protocol, runtime_checkable

from noobfriend.core.io.remote import (
    _parse_spec,
    fetch_bytes,
    fetch_range,
    fetch_tail,
)


@runtime_checkable
class ByteAccessor(Protocol):
    """A location-transparent source of one file's bytes.

    Implementations provide whole-file streaming and two partial reads, so a
    caller can move only the bytes it needs without knowing where the file lives.
    """

    def open(self) -> BinaryIO:
        """Return a fresh seekable binary stream over the whole file.

        Returns
        -------
        BinaryIO
            A stream positioned at the start of the file. The caller owns it and
            should close it (e.g. with a ``with`` block).
        """
        ...

    def read_range(self, offset: int, length: int) -> bytes:
        """Return ``length`` bytes starting at ``offset`` (a short read at EOF)."""
        ...

    def read_tail(self, length: int) -> bytes:
        """Return the file's last ``length`` bytes (the whole file if shorter)."""
        ...


class LocalAccessor:
    """A :class:`ByteAccessor` backed by a path on the local filesystem.

    Parameters
    ----------
    path : str or Path
        The local file path.
    """

    def __init__(self, path: str | Path) -> None:
        """See the class docstring for parameters."""
        self._path: Path = Path(path)

    def open(self) -> BinaryIO:
        """Open the file for reading, so a reader can lazily seek and read it."""
        return self._path.open("rb")

    def read_range(self, offset: int, length: int) -> bytes:
        """Return ``length`` bytes from ``offset`` via ``seek`` + ``read``."""
        return fetch_range(self._path, offset, length)

    def read_tail(self, length: int) -> bytes:
        """Return the file's last ``length`` bytes, seeking from end-of-file."""
        return fetch_tail(self._path, length)


class RemoteAccessor:
    """A :class:`ByteAccessor` backed by an ``[user@]host:path`` spec over SSH.

    Parameters
    ----------
    spec : str
        A remote ``[user@]host:path`` location.
    """

    def __init__(self, spec: str) -> None:
        """See the class docstring for parameters."""
        self._spec: str = spec

    def open(self) -> BinaryIO:
        """Return the whole remote file as an in-memory seekable stream.

        Unlike the partial reads, this pulls the entire file over SSH -- it backs
        the whole-file consumers (e.g. opening a product as a JWST datamodel).
        """
        return BytesIO(fetch_bytes(self._spec))

    def read_range(self, offset: int, length: int) -> bytes:
        """Return ``length`` bytes from ``offset`` via a remote ranged read."""
        return fetch_range(self._spec, offset, length)

    def read_tail(self, length: int) -> bytes:
        """Return the file's last ``length`` bytes via a remote trailing read."""
        return fetch_tail(self._spec, length)


class BytesAccessor:
    """A :class:`ByteAccessor` over bytes already held in memory.

    Lets a caller that just produced (and still holds) a file's bytes -- e.g. a
    reduction step that wrote a product -- feed them to the FITS readers through
    the same accessor interface, without any disk or network read.

    Parameters
    ----------
    data : bytes
        The whole file content.
    """

    def __init__(self, data: bytes) -> None:
        """See the class docstring for parameters."""
        self._data: bytes = data

    def open(self) -> BinaryIO:
        """Return a seekable stream over the in-memory bytes."""
        return BytesIO(self._data)

    def read_range(self, offset: int, length: int) -> bytes:
        """Return ``length`` bytes from ``offset`` (a short slice at EOF)."""
        return self._data[offset : offset + length]

    def read_tail(self, length: int) -> bytes:
        """Return the last ``length`` bytes (the whole content if shorter)."""
        return self._data[max(0, len(self._data) - length) :]


def open_accessor(spec: str | Path) -> ByteAccessor:
    """Return the :class:`ByteAccessor` for ``spec`` -- the one local/remote branch.

    Parameters
    ----------
    spec : str or Path
        A local path, or an ``[user@]host:path`` string (see
        :func:`noobfriend.core.io.remote._parse_spec` for the rule).

    Returns
    -------
    ByteAccessor
        A :class:`LocalAccessor` for a local spec, a :class:`RemoteAccessor` for a
        remote one.

    Raises
    ------
    ValueError
        ``spec`` is malformed (see :func:`_parse_spec`).
    """
    host, path = _parse_spec(spec)
    if host is None:
        return LocalAccessor(path)
    return RemoteAccessor(spec if isinstance(spec, str) else str(spec))
