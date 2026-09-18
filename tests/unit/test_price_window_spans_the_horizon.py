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

from datetime import UTC, date, datetime

import pytest

from arbiter.evaluation.resolution import (
    ResolutionRequest,
    plan_price_windows,
    sessions_after,
    window_closes_on,
)


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
def test_the_fetched_window_always_reaches_the_closing_session(as_of: datetime, horizon: int):
    """The property the calendar approximation violated, across every weekday.

    Parameterised over the week because the old error was invisible on Mondays
    and total on Fridays; a single example would have passed.
    """
    event = _event(as_of, horizon)

    _, end = plan_price_windows([event])["MO"]

    assert end >= window_closes_on(event)


def test_the_window_starts_before_the_filing_so_the_entry_session_is_inside_it():
    event = _event(datetime(2026, 3, 6, 21, 11, tzinfo=UTC))

    start, _ = plan_price_windows([event])["MO"]

    assert start < event.as_of.date()


def test_one_window_per_symbol_still_covers_every_events_horizon():
    """Merging windows must widen them, never clip one to fit another."""
    early = _event(datetime(2026, 3, 2, 23, 0, tzinfo=UTC))
    late = _event(datetime(2026, 3, 6, 21, 11, tzinfo=UTC))

    _, end = plan_price_windows([early, late])["MO"]

    assert end >= window_closes_on(late)
