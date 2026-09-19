"""The price window must reach the session the horizon closes on.

Sessions are not calendar days, and the gap between them depends on the weekday
a filing landed on: five sessions after a Monday is eight calendar days, after a
Friday it is ten. Scaling calendar days by a fixed ratio is therefore wrong by
an amount that varies systematically across the week.

That is what makes the error dangerous rather than merely imprecise. A window
one session short leaves every event filed that day with no price covering its
exit, so the day contributes nothing at all — the measured population loses
whole days chosen by weekday rather than losing observations evenly. A real
backfill lost 675 of 675 insider events from one Friday this way, while the
Monday beside it resolved 544 of 546.
"""

from datetime import UTC, date, datetime, timedelta

import pytest

from arbiter.evaluation.resolution import (
    ResolutionRequest,
    plan_price_windows,
    sessions_after,
    window_closes_on,
)
from arbiter.ingestion.edgar import UncoveredCalendarError, is_trading_day
from arbiter.ingestion.timestamps import SEC_TIMEZONE


def _event(as_of: datetime, horizon: int = 5) -> ResolutionRequest:
    return ResolutionRequest(
        accession_no="a-1",
        as_of=as_of,
        ticker="MO",
        cik=764180,
        horizon=horizon,
    )


def test_sessions_are_counted_over_the_market_calendar():
    """Friday plus one session is Monday, not Saturday."""
    assert sessions_after(date(2026, 3, 6), 1) == date(2026, 3, 9)


def test_counting_steps_over_weekends():
    assert sessions_after(date(2026, 3, 2), 5) == date(2026, 3, 9)


def test_counting_steps_over_holidays():
    """Friday 2026-09-04, then a weekend and Labor Day before counting starts.

    The five sessions are 8, 9, 10, 11 and 14 September; without the holiday the
    fifth would be the 11th, which is what a weekend-only rule would return.
    """
    assert sessions_after(date(2026, 9, 4), 5) == date(2026, 9, 14)
    assert date(2026, 9, 7) not in {
        sessions_after(date(2026, 9, 4), step) for step in range(1, 6)
    }


def test_a_friday_filing_closes_a_full_week_and_a_half_later():
    """Entry is the Monday after; five sessions from there is the next Monday."""
    friday_evening = datetime(2026, 3, 6, 21, 11, tzinfo=UTC)

    assert window_closes_on(_event(friday_evening)) == date(2026, 3, 16)


def test_a_monday_filing_closes_the_following_tuesday():
    monday_evening = datetime(2026, 3, 2, 23, 0, tzinfo=UTC)

    assert window_closes_on(_event(monday_evening)) == date(2026, 3, 10)


@pytest.mark.parametrize(
    "as_of",
    [
        datetime(2026, 3, 2, 23, 0, tzinfo=UTC),
        datetime(2026, 3, 3, 21, 0, tzinfo=UTC),
        datetime(2026, 3, 4, 21, 0, tzinfo=UTC),
        datetime(2026, 3, 5, 21, 0, tzinfo=UTC),
        datetime(2026, 3, 6, 21, 11, tzinfo=UTC),
    ],
)
@pytest.mark.parametrize("horizon", [5, 20])
def test_the_fetched_window_holds_enough_sessions_to_measure_the_event(
    as_of: datetime, horizon: int
):
    """The property the calendar approximation violated, stated independently.

    Measuring an event needs the entry session plus `horizon` more, all of them
    after the filing became public. The window is checked by counting sessions
    against the market calendar rather than by comparing it to the function that
    produced it: an earlier version of this test asserted the window reached
    `window_closes_on`, which is what `plan_price_windows` defines it from, so
    it read `x >= x` and passed on the very defect it was written to catch.
    """
    event = _event(as_of, horizon)

    _, end = plan_price_windows([event])["MO"]

    published = as_of.astimezone(SEC_TIMEZONE).date()
    sessions = sum(
        1
        for offset in range(1, (end - published).days + 1)
        if is_trading_day(published + timedelta(days=offset))
    )

    assert sessions >= horizon + 1, (
        f"{sessions} sessions between {published} and {end}; measuring a "
        f"{horizon}-session horizon needs {horizon + 1}"
    )


def test_the_window_starts_before_the_filing_so_the_entry_session_is_inside_it():
    event = _event(datetime(2026, 3, 6, 21, 11, tzinfo=UTC))

    start, _ = plan_price_windows([event])["MO"]

    assert start < event.as_of.date()


def test_the_window_reaches_back_far_enough_for_the_history_a_packet_describes():
    """One window serves labelling and packet construction, so it must span both.

    Counted in sessions from the market calendar rather than compared against
    the constant the window is built from, which would assert the code against
    itself. The requirement is the packet's longest backward statistic: a volume
    percentile ranked over 63 sessions.

    A filing late in the year is used so the whole span stays inside the years
    the holiday table covers; the production window deliberately does not count
    sessions backwards, precisely because an early-January event would reach
    into a year the table does not list.
    """
    event = _event(datetime(2026, 8, 3, 20, 47, tzinfo=UTC))

    start, _ = plan_price_windows([event])["MO"]

    sessions = sum(
        1
        for offset in range((event.as_of.date() - start).days)
        if is_trading_day(start + timedelta(days=offset))
    )

    assert sessions >= 63, (
        f"{sessions} sessions between {start} and {event.as_of.date()}; a packet's "
        "volume percentile ranks over 63"
    )


def test_one_window_per_symbol_still_covers_every_events_horizon():
    """Merging windows must widen them, never clip one to fit another."""
    early = _event(datetime(2026, 3, 2, 23, 0, tzinfo=UTC))
    late = _event(datetime(2026, 3, 6, 21, 11, tzinfo=UTC))

    _, merged = plan_price_windows([early, late])["MO"]
    _, alone = plan_price_windows([late])["MO"]

    assert merged >= alone
    assert merged >= date(2026, 3, 16), "the later event's exit session"


def test_the_window_the_calendar_approximation_produced_is_now_refused():
    """A direct regression on the values that lost 675 of 675 events.

    The replaced rule scaled the horizon by 1.5 calendar days, giving Friday
    2026-03-06 a window ending 2026-03-14 — a Saturday, two sessions short of
    the 2026-03-16 exit. Stating the old output as a literal keeps this honest
    if the implementation is rewritten again.
    """
    event = _event(datetime(2026, 3, 6, 21, 11, tzinfo=UTC))

    _, end = plan_price_windows([event])["MO"]

    assert end > date(2026, 3, 14)


def test_a_year_the_holiday_table_does_not_cover_is_refused():
    """Answering would report every weekday holiday that year as a trading day.

    Christmas 2024 fell on a Wednesday. A table listing only 2026 and 2027
    called it open, and `sessions_after` walks that table to place every entry
    and exit session — so the error would shift the session index of every event
    whose window spanned it, in some years and not others.
    """
    with pytest.raises(UncoveredCalendarError, match="outside the holiday calendar"):
        is_trading_day(date(2024, 12, 25))

    with pytest.raises(UncoveredCalendarError):
        is_trading_day(date(2028, 7, 4))


def test_the_covered_years_are_answered_normally():
    assert is_trading_day(date(2026, 12, 25)) is False
    assert is_trading_day(date(2026, 3, 6)) is True
