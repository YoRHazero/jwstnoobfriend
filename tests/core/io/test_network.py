"""Tests for the transient-failure retry logic in :class:`HTTPSession`."""

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import ClientConnectionError, ClientResponseError

from noobfriend.core.io.network import (
    HTTPSession,
    _is_retryable,
    _retry_after_seconds,
    _retry_delay,
)


def _response_error(
    status: int, headers: dict[str, str] | None = None
) -> ClientResponseError:
    """Build a :class:`ClientResponseError` with a given status and headers."""
    return ClientResponseError(None, (), status=status, headers=headers)


class _FakeResponse:
    """Minimal stand-in for an :class:`aiohttp.ClientResponse` context manager."""

    def __init__(
        self, *, status: int = 200, body: bytes = b"ok", headers: dict | None = None
    ) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {}
        self.request_info = None
        self.history: tuple = ()

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise ClientResponseError(
                self.request_info,
                self.history,
                status=self.status,
                headers=self.headers,
            )

    async def read(self) -> bytes:
        return self._body

    async def json(self):
        return json.loads(self._body)


class _FakeSession:
    """Returns a scripted sequence of responses, clamping at the last one."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.request_count = 0

    def request(self, *args: object, **kwargs: object) -> _FakeResponse:
        index = min(self.request_count, len(self._responses) - 1)
        self.request_count += 1
        return self._responses[index]


def _wire_session(
    monkeypatch: pytest.MonkeyPatch, http: HTTPSession, session: _FakeSession
) -> list[float]:
    """Patch ``http.acquire`` to yield ``session`` and capture backoff sleeps."""
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    @asynccontextmanager
    async def fake_acquire():
        yield session

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(http, "acquire", fake_acquire)
    return sleeps


class TestIsRetryable:
    """Which errors are treated as transient."""

    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
    def test_retryable_statuses(self, status: int) -> None:
        assert _is_retryable(_response_error(status)) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_permanent_statuses_are_not_retried(self, status: int) -> None:
        assert _is_retryable(_response_error(status)) is False

    def test_connection_and_timeout_errors_retry(self) -> None:
        assert _is_retryable(ClientConnectionError("reset")) is True
        assert _is_retryable(TimeoutError()) is True

    def test_unrelated_error_is_not_retried(self) -> None:
        assert _is_retryable(ValueError("bad input")) is False


class TestRetryAfterSeconds:
    """Parsing of the ``Retry-After`` header."""

    def test_integer_seconds(self) -> None:
        exc = _response_error(503, {"Retry-After": "2"})
        assert _retry_after_seconds(exc) == pytest.approx(2.0)

    def test_http_date_form_is_ignored(self) -> None:
        exc = _response_error(503, {"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"})
        assert _retry_after_seconds(exc) is None

    def test_missing_header_and_non_response_errors(self) -> None:
        assert _retry_after_seconds(_response_error(503, {})) is None
        assert _retry_after_seconds(_response_error(503, None)) is None
        assert _retry_after_seconds(ClientConnectionError("x")) is None


class TestRetryDelay:
    """Exponential backoff timing."""

    def test_first_retry_is_positive_and_within_equal_jitter_band(self) -> None:
        # base = 0.5 * 2**0 = 0.5 -> delay in [0.25, 0.5], never immediate.
        for _ in range(50):
            delay = _retry_delay(0, ClientConnectionError("x"), 0.5, 30.0)
            assert 0.25 <= delay <= 0.5

    def test_delay_grows_with_attempt(self) -> None:
        # base = min(30, 0.5 * 2**3) = 4 -> delay in [2, 4].
        for _ in range(50):
            delay = _retry_delay(3, ClientConnectionError("x"), 0.5, 30.0)
            assert 2.0 <= delay <= 4.0

    def test_retry_after_raises_the_delay(self) -> None:
        exc = _response_error(503, {"Retry-After": "5"})
        assert _retry_delay(0, exc, 0.5, 30.0) == pytest.approx(5.0)

    def test_retry_after_is_capped_at_backoff_max(self) -> None:
        exc = _response_error(503, {"Retry-After": "1000"})
        assert _retry_delay(0, exc, 0.5, 30.0) == pytest.approx(30.0)


class TestRequestRetryLoop:
    """End-to-end retry behaviour through the public ``fetch_content``."""

    def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        http = HTTPSession()
        session = _FakeSession(
            [
                _FakeResponse(status=503),
                _FakeResponse(status=503),
                _FakeResponse(status=200, body=b"done"),
            ]
        )
        sleeps = _wire_session(monkeypatch, http, session)

        result = asyncio.run(
            http.fetch_content("http://x", retries=3, backoff_factor=0.01)
        )

        assert result == b"done"
        assert session.request_count == 3
        assert len(sleeps) == 2  # two backoff waits before the success

    def test_raises_after_exhausting_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = HTTPSession()
        session = _FakeSession([_FakeResponse(status=503)])
        sleeps = _wire_session(monkeypatch, http, session)

        with pytest.raises(ClientResponseError):
            asyncio.run(http.fetch_content("http://x", retries=2, backoff_factor=0.01))

        assert session.request_count == 3  # initial attempt + 2 retries
        assert len(sleeps) == 2

    def test_permanent_error_is_not_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        http = HTTPSession()
        session = _FakeSession([_FakeResponse(status=404)])
        sleeps = _wire_session(monkeypatch, http, session)

        with pytest.raises(ClientResponseError):
            asyncio.run(http.fetch_content("http://x", retries=5, backoff_factor=0.01))

        assert session.request_count == 1  # failed once, no retry
        assert sleeps == []
