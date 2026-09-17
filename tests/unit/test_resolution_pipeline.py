"""Turning stored events into labels.

The pipeline's job is to decide which events can be measured yet, find the
sessions each one could have traded, and refuse the rest without inventing
anything. The price fetcher is injected so these tests describe that logic
rather than the network.
"""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from arbiter.evaluation.resolution import (
    ResolutionRequest,
    entry_sessions,
    plan_price_windows,
    resolvable_on,
)
from arbiter.ingestion.market import Bar

SESSION_OPEN_HOUR = 4  # daily bars are stamped at midnight Eastern


def _bar(day: int, close: str = "100") -> Bar:
    return Bar(
        timestamp=datetime(2026, 8, day, SESSION_OPEN_HOUR, tzinfo=UTC),
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


# --- Which sessions an event could have traded ------------------------------


def test_entry_is_the_first_session_after_the_filing():
    """A filing accepted after the close cannot trade that session."""
    bars = [_bar(3), _bar(4), _bar(5)]
    as_of = datetime(2026, 8, 3, 20, 47, tzinfo=UTC)  # 16:47 Eastern, after close

    tradeable = entry_sessions(bars, as_of)

    assert tradeable[0].timestamp.date() == date(2026, 8, 4)


def test_a_filing_before_the_session_still_waits_for_the_next_one():
    """The nightly batch decides after the close, so entry is always the next open."""
    bars = [_bar(3), _bar(4), _bar(5)]
    as_of = datetime(2026, 8, 3, 11, 0, tzinfo=UTC)  # 07:00 Eastern, pre-market

    assert entry_sessions(bars, as_of)[0].timestamp.date() == date(2026, 8, 4)


def test_sessions_before_the_filing_are_never_returned():
    bars = [_bar(3), _bar(4), _bar(5)]
    as_of = datetime(2026, 8, 4, 20, 47, tzinfo=UTC)

    assert [bar.timestamp.date() for bar in entry_sessions(bars, as_of)] == [date(2026, 8, 5)]


def test_an_event_with_no_later_session_yields_nothing():
    """A filing from last night has no tradeable session yet."""
    assert entry_sessions([_bar(3)], datetime(2026, 8, 3, 20, 47, tzinfo=UTC)) == []


# --- Which events can be measured yet ---------------------------------------


def test_an_event_whose_window_has_closed_is_resolvable():
    """Five sessions after 3 August, with a settled tape, is measurable."""
    event = ResolutionRequest(
        accession_no="a-1",
        as_of=datetime(2026, 8, 3, 20, 47, tzinfo=UTC),
        ticker="MO",
        cik=764180,
        horizon=5,
    )

    assert resolvable_on(event, today=date(2026, 9, 1))


def test_an_event_whose_window_is_still_open_is_not_resolvable():
    event = ResolutionRequest(
        accession_no="a-1",
        as_of=datetime(2026, 8, 28, 20, 47, tzinfo=UTC),
        ticker="MO",
        cik=764180,
        horizon=20,
    )

    assert not resolvable_on(event, today=date(2026, 9, 1))


def test_an_event_resolving_today_waits_for_the_tape():
    """The consolidated feed refuses the current session, so resolution waits a day."""
    event = ResolutionRequest(
        accession_no="a-1",
        as_of=datetime(2026, 8, 24, 20, 47, tzinfo=UTC),
        ticker="MO",
        cik=764180,
        horizon=5,
    )

    # Calendar-day arithmetic puts the window's end on 31 August.
    assert not resolvable_on(event, today=date(2026, 8, 31))
    assert resolvable_on(event, today=date(2026, 9, 2))


# --- Fetching prices once per symbol rather than once per event -------------


def test_one_window_is_planned_per_symbol():
    """Fifty events on one issuer must not become fifty price requests."""
    events = [
        ResolutionRequest("a-1", datetime(2026, 8, 3, 20, tzinfo=UTC), "MO", 764180, 5),
        ResolutionRequest("a-2", datetime(2026, 8, 10, 20, tzinfo=UTC), "MO", 764180, 5),
        ResolutionRequest("a-3", datetime(2026, 8, 5, 20, tzinfo=UTC), "AAPL", 320193, 5),
    ]

    windows = plan_price_windows(events)

    assert set(windows) == {"MO", "AAPL"}


def test_a_symbol_window_spans_all_of_its_events():
    events = [
        ResolutionRequest("a-1", datetime(2026, 8, 3, 20, tzinfo=UTC), "MO", 764180, 5),
        ResolutionRequest("a-2", datetime(2026, 8, 20, 20, tzinfo=UTC), "MO", 764180, 5),
    ]

    start, end = plan_price_windows(events)["MO"]

    assert start <= date(2026, 8, 3)
    assert end >= date(2026, 8, 27)


def test_the_window_starts_before_the_first_event_so_entry_is_available():
    """The entry session follows the filing, so the window cannot begin at it."""
    events = [ResolutionRequest("a-1", datetime(2026, 8, 3, 20, tzinfo=UTC), "MO", 764180, 5)]

    start, _ = plan_price_windows(events)["MO"]

    assert start <= date(2026, 8, 3)


def test_planning_no_events_asks_for_nothing():
    assert plan_price_windows([]) == {}


def test_requests_are_immutable():
    event = ResolutionRequest("a-1", datetime(2026, 8, 3, 20, tzinfo=UTC), "MO", 764180, 5)

    with pytest.raises((AttributeError, TypeError)):
        event.ticker = "AAPL"
