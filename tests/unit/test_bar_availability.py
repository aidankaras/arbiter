"""Which bars a packet is allowed to contain.

A packet states what was knowable at one instant, so the question for every bar
is not which session it belongs to but whether its closing price had been
published yet. Those differ for exactly the filings that matter most: a filing
accepted at 10:00 Eastern lands in the middle of a session whose close is still
six hours away, and including that bar would hand every arm the answer to part
of its own question.

The boundary moves with the calendar, which is why it is pinned here rather than
written as a constant. A close fixed at 20:00 UTC is right from March to
November and an hour early for the rest of the year, so a packet built in
January would carry a bar that had not closed. That error would appear only in
winter — correlated with the season, invisible to any check on its size.
"""

from datetime import UTC, datetime
from decimal import Decimal

from arbiter.ingestion.market import Bar, closed_bars, session_close_instant


def _bar(year: int, month: int, day: int) -> Bar:
    """One daily bar, timestamped as the vendor stamps them: session start."""
    return Bar(
        timestamp=datetime(year, month, day, 5, 0, tzinfo=UTC),
        open=Decimal("10"),
        high=Decimal("11"),
        low=Decimal("9"),
        close=Decimal("10.5"),
        volume=Decimal("1000"),
    )


def test_a_session_closes_at_four_eastern_in_summer():
    """Daylight time: 16:00 Eastern is 20:00 UTC."""
    assert session_close_instant(datetime(2026, 7, 15).date()) == datetime(
        2026, 7, 15, 20, 0, tzinfo=UTC
    )


def test_a_session_closes_an_hour_later_in_utc_in_winter():
    """Standard time: the same 16:00 Eastern is 21:00 UTC.

    The case a fixed-offset constant gets wrong, and gets wrong only in winter.
    """
    assert session_close_instant(datetime(2026, 1, 15).date()) == datetime(
        2026, 1, 15, 21, 0, tzinfo=UTC
    )


def test_a_filing_after_the_close_may_read_that_session():
    """16:47 Eastern is after the close, so the day's own bar is knowable."""
    as_of = datetime(2026, 7, 15, 20, 47, tzinfo=UTC)

    assert closed_bars([_bar(2026, 7, 15)], as_of) == [_bar(2026, 7, 15)]


def test_a_filing_during_the_session_may_not_read_it():
    """10:00 Eastern: the session is open and its close does not exist yet."""
    as_of = datetime(2026, 7, 15, 14, 0, tzinfo=UTC)

    assert closed_bars([_bar(2026, 7, 15)], as_of) == []


def test_the_boundary_instant_counts_as_closed():
    """A price published at the close is available at the close."""
    as_of = datetime(2026, 7, 15, 20, 0, tzinfo=UTC)

    assert closed_bars([_bar(2026, 7, 15)], as_of) == [_bar(2026, 7, 15)]


def test_one_second_before_the_close_does_not():
    as_of = datetime(2026, 7, 15, 19, 59, 59, tzinfo=UTC)

    assert closed_bars([_bar(2026, 7, 15)], as_of) == []


def test_a_winter_filing_at_twenty_thirty_utc_may_not_read_the_session():
    """The regression that a fixed 20:00 UTC close would let through.

    20:30 UTC in January is 15:30 Eastern — half an hour before the close. A
    constant-offset rule would call the session closed and leak that day's
    return into the packet, every winter and never in summer.
    """
    as_of = datetime(2026, 1, 15, 20, 30, tzinfo=UTC)

    assert closed_bars([_bar(2026, 1, 15)], as_of) == []


def test_earlier_sessions_remain_available():
    """Truncation removes the future, not the history."""
    bars = [_bar(2026, 7, 13), _bar(2026, 7, 14), _bar(2026, 7, 15)]
    as_of = datetime(2026, 7, 15, 14, 0, tzinfo=UTC)

    assert closed_bars(bars, as_of) == [_bar(2026, 7, 13), _bar(2026, 7, 14)]


def test_nothing_is_available_before_any_session_closed():
    bars = [_bar(2026, 7, 14), _bar(2026, 7, 15)]
    as_of = datetime(2026, 7, 13, 12, 0, tzinfo=UTC)

    assert closed_bars(bars, as_of) == []


def test_an_early_close_errs_toward_exclusion():
    """Half-days close at 13:00 Eastern and are not tabulated.

    Requiring 16:00 on such a day withholds a bar that was in fact published.
    That is the safe direction: a packet missing a bar is weaker evidence, while
    a packet holding an unclosed one is wrong evidence.
    """
    # 2026-11-27, the session after Thanksgiving, closes at 13:00 Eastern.
    as_of = datetime(2026, 11, 27, 18, 30, tzinfo=UTC)  # 13:30 Eastern

    assert closed_bars([_bar(2026, 11, 27)], as_of) == []
