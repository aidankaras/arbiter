"""Batched price requests must not let one bad symbol cost a whole day.

Symbols reach the price service from two places with different guarantees: a
Form 4 reports one directly, while an 8-K names only a CIK whose ticker is
looked up. The second path produced `OAK-PA`, a preferred issue the service
refuses, and because a day's symbols travel in a single request to stay inside
the rate limit, that one symbol cost every label for the day.

Two defences are tested here. The first keeps unusable symbols out of the
request. The second accepts that a well-formed symbol may still be uncovered,
and confines the damage to that symbol.
"""

from datetime import date

import httpx
import pytest

from arbiter.ingestion import market
from arbiter.ingestion.market import (
    SystemicRejectionError,
    daily_bars_tolerating_gaps,
    is_tradeable_symbol,
)

START = date(2026, 8, 4)
END = date(2026, 9, 1)
TODAY = date(2026, 9, 17)


@pytest.mark.parametrize("symbol", ["MO", "AGNC", "CAVA", "BRK.B", "agnc", " MO "])
def test_a_plain_equity_symbol_is_accepted(symbol: str):
    assert is_tradeable_symbol(symbol)


@pytest.mark.parametrize(
    "symbol",
    [
        "OAK-PA",  # the preferred issue that caused a live request to be refused
        "BRK.PRA",
        "ABCDEF",
        "",
        "   ",
        "N/A",
        "NONE",
        "TBD",
        "123",
        None,
    ],
)
def test_anything_that_is_not_a_plain_equity_symbol_is_refused(symbol: object):
    assert not is_tradeable_symbol(symbol)


def _responding(refused: set[str]):
    """Return a `daily_bars` stand-in that refuses a request containing `refused`.

    This is how the service behaves: the rejection names the request, not the
    offending symbol, so the caller can only learn which symbol it was by
    splitting the batch.
    """

    def fake_daily_bars(symbols, start, end, today):
        if set(symbols) & refused:
            request = httpx.Request("GET", market.BARS_ENDPOINT)
            response = httpx.Response(400, request=request)
            raise httpx.HTTPStatusError("invalid symbol", request=request, response=response)
        return {symbol: [f"bar-for-{symbol}"] for symbol in symbols}

    return fake_daily_bars


def test_one_refused_symbol_costs_only_itself(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(market, "daily_bars", _responding({"OAK-PA"}))

    bars = daily_bars_tolerating_gaps(["MO", "OAK-PA", "AGNC", "CAVA"], START, END, TODAY)

    assert bars["OAK-PA"] == []
    assert bars["MO"] == ["bar-for-MO"]
    assert set(bars) == {"MO", "OAK-PA", "AGNC", "CAVA"}


def test_an_unrefused_batch_is_fetched_in_one_request(monkeypatch: pytest.MonkeyPatch):
    """Splitting costs requests, and the rate limit is the reason for batching."""
    calls: list[int] = []

    def counting(symbols, start, end, today):
        calls.append(len(symbols))
        return {symbol: [] for symbol in symbols}

    monkeypatch.setattr(market, "daily_bars", counting)

    daily_bars_tolerating_gaps(["MO", "AGNC", "CAVA"], START, END, TODAY)

    assert calls == [3]


def test_a_request_refused_for_its_own_sake_is_raised(monkeypatch: pytest.MonkeyPatch):
    """Every symbol failing alone means the window is wrong, not the symbols."""
    monkeypatch.setattr(market, "daily_bars", _responding({"MO", "AGNC", "CAVA", "OAK-PA"}))

    with pytest.raises(SystemicRejectionError, match="points at the request"):
        daily_bars_tolerating_gaps(["MO", "AGNC", "CAVA", "OAK-PA"], START, END, TODAY)


def test_a_quiet_window_is_not_mistaken_for_a_broken_request(
    monkeypatch: pytest.MonkeyPatch,
):
    """A symbol that did not trade is empty too; only refusals count as refusals."""

    def silent(symbols, start, end, today):
        if "OAK-PA" in symbols:
            request = httpx.Request("GET", market.BARS_ENDPOINT)
            raise httpx.HTTPStatusError(
                "invalid symbol", request=request, response=httpx.Response(400, request=request)
            )
        return {symbol: [] for symbol in symbols}

    monkeypatch.setattr(market, "daily_bars", silent)

    bars = daily_bars_tolerating_gaps(["MO", "OAK-PA", "AGNC", "CAVA"], START, END, TODAY)

    assert all(series == [] for series in bars.values())


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_failures_other_than_rejected_symbols_are_not_split(
    monkeypatch: pytest.MonkeyPatch, status: int
):
    """Splitting a credential or rate-limit failure multiplies it and hides it."""
    calls: list[int] = []

    def failing(symbols, start, end, today):
        calls.append(len(symbols))
        request = httpx.Request("GET", market.BARS_ENDPOINT)
        raise httpx.HTTPStatusError(
            "refused", request=request, response=httpx.Response(status, request=request)
        )

    monkeypatch.setattr(market, "daily_bars", failing)

    with pytest.raises(httpx.HTTPStatusError):
        daily_bars_tolerating_gaps(["MO", "AGNC"], START, END, TODAY)

    assert calls == [2], "a non-symbol failure must not be retried per symbol"


def test_no_symbols_makes_no_request(monkeypatch: pytest.MonkeyPatch):
    def unreachable(symbols, start, end, today):
        raise AssertionError("a request was made for no symbols")

    monkeypatch.setattr(market, "daily_bars", unreachable)

    assert daily_bars_tolerating_gaps([], START, END, TODAY) == {}
