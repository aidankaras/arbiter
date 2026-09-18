"""What the price feed's response must mean, checked against a recorded one.

Every other test of the batching layer replaces `daily_bars`, so the function
that holds the pagination loop, the feed rule and the credential handling has
never executed under test. These run the real function against recorded bytes,
which is the only place a change in what the service *means* — rather than what
it returns — would be caught.

The contract that matters most is the bar timestamp. `entry_sessions` decides
which session an event could have traded by comparing a bar's timestamp to the
filing instant, so the entry rule rests entirely on that stamp marking the
session's start. Nothing else in the suite states it, and if the feed ever
stamped a bar at the session close instead, every intraday filing would enter
at an open that preceded it — a lookahead leak with no failing test.
"""

import json
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from arbiter.ingestion import market
from arbiter.ingestion.market import Bar, RecentSipWindowError, daily_bars, parse_bars
from arbiter.ingestion.timestamps import SEC_TIMEZONE

FIXTURE = Path(__file__).parents[1] / "fixtures" / "alpaca_bars.json"
WINDOW_START = date(2026, 7, 27)
WINDOW_END = date(2026, 8, 28)
TODAY = date(2026, 9, 18)


def _payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def _serving(
    pages: list[dict[str, Any]], calls: list[httpx.Request]
) -> Callable[..., httpx.Response]:
    """Return an `httpx.get` stand-in that serves recorded pages in order."""
    remaining = list(pages)

    def get(url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("GET", url, params=kwargs.get("params"))
        calls.append(request)
        return httpx.Response(200, json=remaining.pop(0), request=request)

    return get


def test_a_daily_bar_is_stamped_at_the_session_it_opens(monkeypatch: pytest.MonkeyPatch):
    """The entry rule rests on this and nothing else asserts it.

    `entry_sessions` keeps bars whose timestamp is after the filing instant. If
    a bar were stamped at the session close, a filing accepted mid-session would
    be matched to that same session and would enter at an open that had already
    happened.
    """
    calls: list[httpx.Request] = []
    monkeypatch.setattr(httpx, "get", _serving([_payload()], calls))
    monkeypatch.setattr(market, "_credentials", lambda: {})

    bars = daily_bars(["MO"], WINDOW_START, WINDOW_END, TODAY)

    assert bars["MO"], "the recorded fixture must contain bars"
    for bar in bars["MO"]:
        eastern = bar.timestamp.astimezone(SEC_TIMEZONE)
        assert eastern.time() <= time(9, 30), (
            f"bar stamped {eastern:%H:%M} ET — at or after the open it represents"
        )


def test_the_real_fetch_parses_a_recorded_response(monkeypatch: pytest.MonkeyPatch):
    """`daily_bars` itself, not a stand-in, against bytes the service actually sent."""
    calls: list[httpx.Request] = []
    monkeypatch.setattr(httpx, "get", _serving([_payload()], calls))
    monkeypatch.setattr(market, "_credentials", lambda: {})

    bars = daily_bars(["MO", "XLP"], WINDOW_START, WINDOW_END, TODAY)

    assert set(bars) == {"MO", "XLP"}
    assert all(isinstance(bar, Bar) for bar in bars["MO"])
    assert len(calls) == 1


def test_the_request_names_the_consolidated_feed_and_adjusts_for_splits(
    monkeypatch: pytest.MonkeyPatch,
):
    """Substituting a thinner feed would make a label depend on when it was computed."""
    calls: list[httpx.Request] = []
    monkeypatch.setattr(httpx, "get", _serving([_payload()], calls))
    monkeypatch.setattr(market, "_credentials", lambda: {})

    daily_bars(["MO"], WINDOW_START, WINDOW_END, TODAY)

    query = str(calls[0].url)
    assert "feed=sip" in query
    assert "adjustment=all" in query, "an unadjusted split would read as a 50% return"


def test_a_paginated_response_is_read_to_the_end(monkeypatch: pytest.MonkeyPatch):
    """Reading only the first page returns a short series that looks complete.

    `parse_bars` cannot distinguish a truncated series from a thinly traded
    ticker, so a dropped page becomes a wrong label rather than an error.
    """
    recorded = _payload()
    first_half = {"bars": {"MO": recorded["bars"]["MO"][:10]}, "next_page_token": "page-2"}
    second_half = {"bars": {"MO": recorded["bars"]["MO"][10:]}, "next_page_token": None}

    calls: list[httpx.Request] = []
    monkeypatch.setattr(httpx, "get", _serving([first_half, second_half], calls))
    monkeypatch.setattr(market, "_credentials", lambda: {})

    bars = daily_bars(["MO"], WINDOW_START, WINDOW_END, TODAY)

    assert len(calls) == 2, "the second page must be requested"
    assert len(bars["MO"]) == len(recorded["bars"]["MO"])
    assert "page_token=page-2" in str(calls[1].url)


def test_pages_are_concatenated_without_repeating_a_session(
    monkeypatch: pytest.MonkeyPatch,
):
    """A duplicated session would shift the exit bar and silently misprice the label."""
    recorded = _payload()
    first_half = {"bars": {"MO": recorded["bars"]["MO"][:10]}, "next_page_token": "page-2"}
    second_half = {"bars": {"MO": recorded["bars"]["MO"][10:]}, "next_page_token": None}

    monkeypatch.setattr(httpx, "get", _serving([first_half, second_half], []))
    monkeypatch.setattr(market, "_credentials", lambda: {})

    bars = daily_bars(["MO"], WINDOW_START, WINDOW_END, TODAY)
    stamps = [bar.timestamp for bar in bars["MO"]]

    assert len(stamps) == len(set(stamps))
    assert stamps == sorted(stamps)


def test_a_window_covering_the_current_session_is_refused_before_any_request(
    monkeypatch: pytest.MonkeyPatch,
):
    """The feed refuses it, and falling back to a thinner one would break comparability."""
    calls: list[httpx.Request] = []
    monkeypatch.setattr(httpx, "get", _serving([_payload()], calls))
    monkeypatch.setattr(market, "_credentials", lambda: {})

    with pytest.raises(RecentSipWindowError):
        daily_bars(["MO"], WINDOW_START, TODAY, TODAY)

    assert calls == [], "the window is checked before the request is made"


def test_a_symbol_absent_from_the_response_yields_no_bars_rather_than_an_error():
    """A ticker that did not trade is a real answer; the caller decides what it means."""
    assert parse_bars(_payload(), "NOTLISTED") == []


def test_prices_survive_the_response_as_exact_decimals():
    """A float round-trip would perturb a return in its final digits."""
    bars = parse_bars(_payload(), "MO")

    assert bars[0].close == Decimal(str(_payload()["bars"]["MO"][0]["c"]))


def test_timestamps_arrive_timezone_aware():
    """A naive timestamp compared against an aware `as_of` raises at the entry rule."""
    for bar in parse_bars(_payload(), "MO"):
        assert bar.timestamp.tzinfo is not None
        assert bar.timestamp.astimezone(UTC).date() >= datetime(2026, 1, 1, tzinfo=UTC).date()
