"""Choosing which days a backfill covers.

The cost of a backfill is fixed per trading day — every Form 4 accepted that day
must be fetched to learn whether it qualifies — so the choice of days is the
only lever on how long a run takes, and it is made once, here, rather than
implicitly by whoever invokes it.

Sampling every Nth trading day rather than taking a contiguous block buys
statistical independence at the same wall-clock cost: events filed on one day
share a residual market factor even after the sector benchmark is subtracted,
so a contiguous month yields far fewer effective observations than its event
count suggests. Stepping by a fixed count also rotates through weekdays, which
picking named weekdays would not.
"""

from datetime import date

import pytest

from arbiter.ingestion.backfill import sampled_trading_days


def test_every_trading_day_is_taken_by_default():
    days = sampled_trading_days(date(2026, 8, 3), date(2026, 8, 7), every=1)

    assert days == [
        date(2026, 8, 3),
        date(2026, 8, 4),
        date(2026, 8, 5),
        date(2026, 8, 6),
        date(2026, 8, 7),
    ]


def test_weekends_are_never_selected():
    """August 8 and 9 2026 are a Saturday and Sunday."""
    days = sampled_trading_days(date(2026, 8, 7), date(2026, 8, 10), every=1)

    assert days == [date(2026, 8, 7), date(2026, 8, 10)]


def test_market_holidays_are_never_selected():
    """September 7 2026 is Labor Day; EDGAR holds no filings for it."""
    days = sampled_trading_days(date(2026, 9, 4), date(2026, 9, 8), every=1)

    assert date(2026, 9, 7) not in days
    assert days == [date(2026, 9, 4), date(2026, 9, 8)]


def test_sampling_steps_over_trading_days_not_calendar_days():
    """Stepping by calendar days would land on weekends and skip unevenly."""
    days = sampled_trading_days(date(2026, 8, 3), date(2026, 8, 14), every=3)

    assert days == [date(2026, 8, 3), date(2026, 8, 6), date(2026, 8, 11), date(2026, 8, 14)]


def test_sampling_rotates_through_weekdays():
    """A step coprime with the five-day week must not fix one weekday.

    Sampling only Tuesdays would confound any weekday effect in filing
    behaviour with whatever the model learns.
    """
    days = sampled_trading_days(date(2026, 3, 2), date(2026, 9, 4), every=3)

    assert len({day.weekday() for day in days}) == 5


def test_the_range_is_inclusive_at_both_ends():
    days = sampled_trading_days(date(2026, 8, 3), date(2026, 8, 3), every=1)

    assert days == [date(2026, 8, 3)]


def test_a_range_containing_no_trading_day_yields_nothing():
    days = sampled_trading_days(date(2026, 8, 8), date(2026, 8, 9), every=1)

    assert days == []


def test_a_reversed_range_is_refused_rather_than_silently_empty():
    """Swapped arguments would otherwise look like a range with no trading days."""
    with pytest.raises(ValueError, match="starts after"):
        sampled_trading_days(date(2026, 8, 7), date(2026, 8, 3), every=1)


@pytest.mark.parametrize("every", [0, -1])
def test_a_step_below_one_is_refused(every: int):
    with pytest.raises(ValueError, match="at least 1"):
        sampled_trading_days(date(2026, 8, 3), date(2026, 8, 7), every=every)
