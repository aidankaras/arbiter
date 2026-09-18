from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from arbiter.evaluation.resolution import (
    HORIZONS,
    Label,
    UnresolvableEventError,
    resolve_label,
)
from arbiter.ingestion.market import Bar


def _bars(closes: list[str], opens: list[str] | None = None) -> list[Bar]:
    """Build a bar series over consecutive sessions from 2026-08-03."""
    opens = opens or closes
    return [
        Bar(
            timestamp=datetime(2026, 8, 3 + index, 4, 0, tzinfo=UTC),
            open=Decimal(opens[index]),
            high=Decimal(closes[index]),
            low=Decimal(opens[index]),
            close=Decimal(closes[index]),
            volume=Decimal("1000"),
        )
        for index in range(len(closes))
    ]


def _bar(day: int, price: str) -> Bar:
    """One bar on a named August 2026 session, so gaps can be expressed."""
    return Bar(
        timestamp=datetime(2026, 8, day, 4, 0, tzinfo=UTC),
        open=Decimal(price),
        high=Decimal(price),
        low=Decimal(price),
        close=Decimal(price),
        volume=Decimal("1000"),
    )


def test_the_horizons_differ_by_domain():
    """Insider signal resolves in a week; red-flag drift runs for a month."""
    assert HORIZONS["insider"] == 5
    assert HORIZONS["redflag"] == 20


def test_a_label_carries_the_benchmark_it_used():
    """A later change to sector mapping must be visible in the data."""
    issuer = _bars(["100", "101", "102", "103", "104", "110"])
    benchmark = _bars(["50", "50", "50", "50", "50", "52"])

    label = resolve_label(
        issuer_bars=issuer,
        benchmark_bars=benchmark,
        benchmark_symbol="XLP",
        horizon=5,
    )

    assert isinstance(label, Label)
    assert label.benchmark_symbol == "XLP"
    assert label.horizon_days == 5


def test_the_label_is_the_excess_return_over_the_window():
    """Entry at the first open, exit at the close of the fifth session."""
    issuer = _bars(["100", "101", "102", "103", "104", "110"], opens=["100"] * 6)
    benchmark = _bars(["50", "50", "50", "50", "50", "52"], opens=["50"] * 6)

    label = resolve_label(
        issuer_bars=issuer,
        benchmark_bars=benchmark,
        benchmark_symbol="XLP",
        horizon=5,
    )

    # Issuer 100 -> 110 is +10%; benchmark 50 -> 52 is +4%; abnormal is +6%.
    assert abs(label.abnormal_return - Decimal("0.06")) < Decimal("0.000001")


def test_entry_uses_the_open_not_the_prior_close():
    """Entering at the prior close would credit the event with overnight movement."""
    issuer = _bars(["100", "101", "102", "103", "104", "110"], opens=["105"] * 6)
    benchmark = _bars(["50", "50", "50", "50", "50", "50"], opens=["50"] * 6)

    label = resolve_label(
        issuer_bars=issuer,
        benchmark_bars=benchmark,
        benchmark_symbol="XLP",
        horizon=5,
    )

    # Entry 105, exit 110 is +4.76%, not the +10% a close-to-close read would give.
    assert abs(label.abnormal_return - Decimal("0.047619")) < Decimal("0.00001")


def test_too_few_sessions_makes_the_event_unresolvable():
    """A window that has not closed yet yields no label rather than a partial one."""
    issuer = _bars(["100", "101", "102"])
    benchmark = _bars(["50", "50", "50"])

    with pytest.raises(UnresolvableEventError, match="sessions"):
        resolve_label(
            issuer_bars=issuer,
            benchmark_bars=benchmark,
            benchmark_symbol="XLP",
            horizon=5,
        )


def test_a_missing_benchmark_series_makes_the_event_unresolvable():
    with pytest.raises(UnresolvableEventError, match="benchmark"):
        resolve_label(
            issuer_bars=_bars(["100", "101", "102", "103", "104", "110"]),
            benchmark_bars=[],
            benchmark_symbol="XLP",
            horizon=5,
        )


def test_the_label_records_its_entry_and_exit_sessions():
    """Every published number must be traceable to the sessions that produced it."""
    issuer = _bars(["100", "101", "102", "103", "104", "110"])
    benchmark = _bars(["50", "50", "50", "50", "50", "52"])

    label = resolve_label(
        issuer_bars=issuer,
        benchmark_bars=benchmark,
        benchmark_symbol="XLP",
        horizon=5,
    )

    assert label.entry_session == date(2026, 8, 3)
    assert label.exit_session == date(2026, 8, 8)


def test_a_halted_issuer_is_measured_against_the_same_sessions_it_traded():
    """The two series are otherwise indexed by position, which assumes they match.

    An issuer halted mid-window holds fewer bars than its benchmark, so the same
    index reaches a later date in one series than the other and the abnormal
    return subtracts a benchmark move measured over a different span. Halts
    cluster on the bad news these events are about, so the error would bias the
    measurement rather than scatter it.
    """
    # The issuer misses 6 and 7 August; the benchmark trades every session.
    issuer = [
        _bar(day, price)
        for day, price in [(3, "100"), (4, "101"), (5, "102"), (10, "103"), (11, "110")]
    ]
    benchmark = [_bar(day, "50") for day in (3, 4, 5, 6, 7, 10, 11)]

    label = resolve_label(
        issuer_bars=issuer,
        benchmark_bars=benchmark,
        benchmark_symbol="XLP",
        horizon=4,
    )

    assert label.entry_session == date(2026, 8, 3)
    assert label.exit_session == date(2026, 8, 11), "the issuer's fifth traded session"


def test_an_issuer_and_benchmark_sharing_too_few_sessions_is_unresolvable():
    """Long enough separately, too little overlap to measure the horizon."""
    issuer = [_bar(day, "100") for day in (3, 4, 5, 6, 7, 10)]
    benchmark = [_bar(day, "50") for day in (17, 18, 19, 20, 21, 24)]

    with pytest.raises(UnresolvableEventError, match="overlapping"):
        resolve_label(
            issuer_bars=issuer,
            benchmark_bars=benchmark,
            benchmark_symbol="XLP",
            horizon=5,
        )
