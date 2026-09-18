"""Which price-fetch failures are worth another attempt, and which are not.

A nightly job that dies on one dropped connection is fragile; a job that retries
a rejected credential is worse, because the failure stops being legible and
starts looking like a day with no prices.
"""

import httpx
import pytest

from arbiter.ingestion.market import _is_retryable


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://data.alpaca.markets/v2/stocks/bars")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError(f"{code}", request=request, response=response)


@pytest.mark.parametrize(
    ("failure", "why"),
    [
        (
            httpx.RemoteProtocolError("server disconnected"),
            "the connection dropped mid-request",
        ),
        (httpx.ConnectError("refused"), "the host was briefly unreachable"),
        (httpx.ReadTimeout("slow"), "the response did not arrive in time"),
    ],
)
def test_transport_failures_are_retried(failure: Exception, why: str):
    assert _is_retryable(failure), why


def test_rate_limiting_is_retried():
    """The service is asking for patience, not refusing the request."""
    assert _is_retryable(_status_error(429))


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_server_errors_are_retried(code: int):
    assert _is_retryable(_status_error(code))


@pytest.mark.parametrize(
    ("code", "why"),
    [
        (400, "a symbol or window the service rejects will be rejected again"),
        (401, "a bad credential retried into silence looks like a quiet day"),
        (403, "an unentitled feed is a configuration error, not a blip"),
        (404, "the endpoint is wrong"),
    ],
)
def test_client_errors_are_not_retried(code: int, why: str):
    assert not _is_retryable(_status_error(code)), why


def test_unrelated_exceptions_are_not_retried():
    """A defect in our own code must surface immediately, not four times."""
    assert not _is_retryable(ValueError("a bug"))
    assert not _is_retryable(KeyError("missing"))
