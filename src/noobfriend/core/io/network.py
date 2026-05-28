"""Async HTTP session management and high-level fetch / download helpers.

Built on :mod:`aiohttp` (transport) and :mod:`aiofiles` (streaming writes).
The single public class :class:`HTTPSession` owns a refcounted
:class:`aiohttp.ClientSession`: nested or concurrent acquisitions share one
underlying session, which is created on first use and closed when the last
acquisition releases it.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import aiofiles
from aiohttp import ClientSession, ClientTimeout, TCPConnector

HTTPMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]


class HTTPSession:
    """Refcounted manager of an :class:`aiohttp.ClientSession`.

    Multiple concurrent users acquire the same underlying ``ClientSession``
    via :meth:`acquire`; the session is created on first acquisition and
    closed when the reference count returns to zero.

    The high-level ``fetch_*`` / ``download_*`` helpers wrap :meth:`acquire`
    automatically, so a casual caller never needs to handle the session
    object directly. Calling them in a row reuses the same underlying
    session if wrapped in an outer :meth:`acquire` block.

    Parameters
    ----------
    timeout : float or None, default ``None``
        Total per-request timeout in seconds. ``None`` means no timeout.
    max_connections : int, default ``100``
        Maximum number of simultaneous TCP connections (``TCPConnector.limit``).
    keepalive_timeout : float, default ``600.0``
        How long idle connections are kept open, in seconds.

    Examples
    --------
    Casual one-shot use::

        http = HTTPSession()
        data = await http.fetch_json("https://example.org/info")

    Batch use sharing one session::

        async with http.acquire():
            for url in urls:
                await http.download_to_file(url, output_dir / url.rsplit("/", 1)[-1])
    """

    def __init__(
        self,
        *,
        timeout: float | None = None,
        max_connections: int = 100,
        keepalive_timeout: float = 600.0,
    ) -> None:
        """Store configuration; the underlying session is created lazily on first acquire."""
        self._timeout = ClientTimeout(total=timeout)
        self._max_connections = max_connections
        self._keepalive_timeout = keepalive_timeout
        self._session: ClientSession | None = None
        self._refcount = 0
        self._lock = asyncio.Lock()

    @property
    def ref_count(self) -> int:
        """Number of currently outstanding :meth:`acquire` calls."""
        return self._refcount

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[ClientSession, None]:
        """Acquire the shared :class:`aiohttp.ClientSession` for the duration of the ``async with`` block.

        Nested or concurrent calls increment a reference count and reuse the
        same underlying session. The session is closed only when the count
        returns to zero.

        Yields
        ------
        aiohttp.ClientSession
            The shared session instance.
        """
        async with self._lock:
            if self._session is None or self._session.closed:
                connector = TCPConnector(
                    limit=self._max_connections,
                    keepalive_timeout=self._keepalive_timeout,
                )
                self._session = ClientSession(
                    timeout=self._timeout, connector=connector
                )
            self._refcount += 1

        try:
            yield self._session
        finally:
            async with self._lock:
                self._refcount -= 1
                if self._refcount == 0:
                    session, self._session = self._session, None
                    if session is not None and not session.closed:
                        await session.close()

    async def fetch_json(
        self,
        url: str,
        *,
        method: HTTPMethod = "GET",
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        """Fetch ``url`` and return the parsed JSON body.

        Parameters
        ----------
        url : str
            Target URL.
        method : {"GET", "POST", "PUT", "DELETE", "PATCH"}, default ``"GET"``
            HTTP method.
        params : mapping or None
            Query-string parameters.
        json_body : mapping or None
            JSON body to send.
        headers : mapping or None
            Extra request headers.

        Returns
        -------
        Any
            Parsed JSON content (usually ``dict`` or ``list``).

        Raises
        ------
        aiohttp.ClientResponseError
            Response status is 4xx or 5xx.
        aiohttp.ClientError
            Any other client-side networking error.
        """
        async with self.acquire() as session:
            async with session.request(
                method, url, params=params, json=json_body, headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def fetch_content(
        self,
        url: str,
        *,
        method: HTTPMethod = "GET",
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        """Fetch ``url`` and return the raw response body.

        Parameters are identical to :meth:`fetch_json`.

        Returns
        -------
        bytes
            Raw response body.

        Raises
        ------
        aiohttp.ClientResponseError
            Response status is 4xx or 5xx.
        aiohttp.ClientError
            Any other client-side networking error.
        """
        async with self.acquire() as session:
            async with session.request(
                method, url, params=params, json=json_body, headers=headers
            ) as response:
                response.raise_for_status()
                return await response.read()

    async def download_to_file(
        self,
        url: str,
        output_path: Path,
        *,
        method: HTTPMethod = "GET",
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        chunk_size: int = 512 * 1024,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Stream ``url`` to ``output_path`` on disk, optionally reporting progress.

        Parameters
        ----------
        url : str
            Target URL.
        output_path : Path
            File path to write to. The parent directory must already exist.
        method : {"GET", "POST", "PUT", "DELETE", "PATCH"}, default ``"GET"``
            HTTP method.
        params, json_body, headers
            See :meth:`fetch_json`.
        chunk_size : int, default ``524288``
            Chunk size in bytes for the streaming read (512 KiB).
        progress_callback : callable or None
            Invoked as ``progress_callback(downloaded, total)`` after each
            chunk is written, where ``total`` comes from the
            ``Content-Length`` header (``0`` if the server omits it).
            Suitable as a bridge to a :class:`rich.progress.Progress` task::

                async with progress:
                    task = progress.add_task("download", total=None)
                    def cb(d: int, t: int) -> None:
                        progress.update(task, completed=d, total=t or None)
                    await http.download_to_file(url, path, progress_callback=cb)

        Raises
        ------
        aiohttp.ClientResponseError
            Response status is 4xx or 5xx.
        aiohttp.ClientError
            Any other client-side networking error.
        """
        async with self.acquire() as session:
            async with session.request(
                method, url, params=params, json=json_body, headers=headers
            ) as response:
                response.raise_for_status()
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                async with aiofiles.open(output_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback is not None:
                            progress_callback(downloaded, total)
