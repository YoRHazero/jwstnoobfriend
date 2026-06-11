"""Async HTTP session management and high-level fetch / download helpers.

Built on :mod:`aiohttp` (transport) and :mod:`aiofiles` (streaming writes).
The single public class :class:`HTTPSession` owns a refcounted
:class:`aiohttp.ClientSession`: nested or concurrent acquisitions share one
underlying session, which is created on first use and closed when the last
acquisition releases it.
"""

import asyncio
import random
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import aiofiles
from aiohttp import (
    ClientConnectionError,
    ClientError,
    ClientResponse,
    ClientResponseError,
    ClientSession,
    ClientTimeout,
    TCPConnector,
)

HTTPMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]

#: HTTP status codes worth retrying: request timeout, rate limit, and 5xx.
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


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
        retries: int = 0,
        backoff_factor: float = 0.5,
        backoff_max: float = 30.0,
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
        retries : int, default ``0``
            Maximum number of *additional* attempts after the first on a
            transient failure. ``0`` (the default) disables retrying and keeps
            the original single-attempt behaviour. See :meth:`_request_with_retry`.
        backoff_factor : float, default ``0.5``
            Base seconds for the exponential backoff between retries.
        backoff_max : float, default ``30.0``
            Upper bound, in seconds, on any single backoff wait.

        Returns
        -------
        Any
            Parsed JSON content (usually ``dict`` or ``list``).

        Raises
        ------
        aiohttp.ClientResponseError
            Response status is 4xx or 5xx (after retries are exhausted).
        aiohttp.ClientError
            Any other client-side networking error.
        """
        return await self._request_with_retry(
            method,
            url,
            params=params,
            json_body=json_body,
            headers=headers,
            retries=retries,
            backoff_factor=backoff_factor,
            backoff_max=backoff_max,
            consume=lambda response: response.json(),
        )

    async def fetch_content(
        self,
        url: str,
        *,
        method: HTTPMethod = "GET",
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        retries: int = 0,
        backoff_factor: float = 0.5,
        backoff_max: float = 30.0,
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
            Response status is 4xx or 5xx (after retries are exhausted).
        aiohttp.ClientError
            Any other client-side networking error.
        """
        return await self._request_with_retry(
            method,
            url,
            params=params,
            json_body=json_body,
            headers=headers,
            retries=retries,
            backoff_factor=backoff_factor,
            backoff_max=backoff_max,
            consume=lambda response: response.read(),
        )

    async def _request_with_retry(
        self,
        method: HTTPMethod,
        url: str,
        *,
        params: Mapping[str, Any] | None,
        json_body: Mapping[str, Any] | None,
        headers: Mapping[str, str] | None,
        retries: int,
        backoff_factor: float,
        backoff_max: float,
        consume: Callable[[ClientResponse], Awaitable[Any]],
    ) -> Any:
        """Send one request, retrying transient failures with exponential backoff.

        An attempt is retried only when the error is transient — a connection
        error, a timeout, or a ``408`` / ``429`` / ``5xx`` response — and at most
        ``retries`` times. Each retry waits an exponentially growing, jittered
        delay (never immediate); a parseable ``Retry-After`` header raises the
        wait to honour the server's request. ``consume`` reads the body while the
        response context is still open (e.g. ``response.read`` or ``response.json``).
        """
        attempt = 0
        while True:
            try:
                async with self.acquire() as session:
                    async with session.request(
                        method, url, params=params, json=json_body, headers=headers
                    ) as response:
                        response.raise_for_status()
                        return await consume(response)
            except (ClientError, asyncio.TimeoutError) as exc:
                if attempt >= retries or not _is_retryable(exc):
                    raise
                await asyncio.sleep(
                    _retry_delay(attempt, exc, backoff_factor, backoff_max)
                )
                attempt += 1

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


def _is_retryable(exc: BaseException) -> bool:
    """Return whether ``exc`` is a transient error worth retrying.

    Retryable: a connection error, a timeout, or a response whose status is in
    :data:`_RETRYABLE_STATUS`. A ``ClientResponseError`` with any other status
    (e.g. ``404``) is permanent and is not retried.
    """
    if isinstance(exc, ClientResponseError):
        return exc.status in _RETRYABLE_STATUS
    return isinstance(exc, (ClientConnectionError, asyncio.TimeoutError))


def _retry_delay(
    attempt: int, exc: BaseException, backoff_factor: float, backoff_max: float
) -> float:
    """Return the jittered exponential backoff delay before the next attempt.

    ``attempt`` is 0-based for the first retry, so the base delay grows as
    ``backoff_factor * 2 ** attempt`` (capped at ``backoff_max``). Equal jitter
    keeps the wait within ``[base / 2, base]`` — always strictly positive, so a
    retry never fires immediately. A parseable ``Retry-After`` header raises the
    delay to honour the server's request, still capped at ``backoff_max``.
    """
    base = min(backoff_max, backoff_factor * 2**attempt)
    delay = base / 2 + random.random() * (base / 2)
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(backoff_max, max(delay, retry_after))
    return delay


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Return the ``Retry-After`` delay in seconds, if the server sent one.

    Only the integer/float *delta-seconds* form is understood; the HTTP-date
    form returns ``None`` so the caller falls back to computed backoff.
    """
    if not isinstance(exc, ClientResponseError) or exc.headers is None:
        return None
    value = exc.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
