"""Building a packet must not depend on the schema catching its mistakes.

The validator refuses a packet holding future information, so a builder that
truncates wrongly produces exceptions rather than bad packets — which is the
safe failure but not a correct one: a backfill would lose every event instead of
labelling it. These tests pin the truncation at the point it is performed, and
the statistics computed from whatever survives it.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from itertools import pairwise
from math import log, sqrt
from statistics import stdev

import pytest

from arbiter.ingestion.market import Bar
from arbiter.packets.build import build_packet, market_summary, to_session_bar
from arbiter.packets.schema import SessionBar

ACCEPTED = datetime(2026, 7, 13, 20, 47, tzinfo=UTC)


def _bar(day: int, close: str = "50", volume: str = "1000") -> Bar:
    """A vendor bar, stamped at the session's start as the feed publishes it."""
    return Bar(
        timestamp=datetime(2026, 7, day, 4, 0, tzinfo=UTC),
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal(volume),
    )


def _session(close: str = "50", volume: str = "1000", day: int = 1) -> SessionBar:
    return SessionBar(
        session=date(2026, 1, day),
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        volume=Decimal(volume),
    )


ROW = {
    "accession_no": "0000764180-26-000102",
    "as_of": ACCEPTED,
    "cik": 764180,
    "ticker": "MO",
    "issuer": "Altria Group Inc.",
    "insider_name": "Jane Roe",
    "position": "Officer",
    "transaction_code": "P",
    "shares": Decimal("1000"),
    "price": Decimal("50.00"),
    "value_usd": Decimal("50000.00"),
    "remaining_shares": Decimal("4000"),
    "is_10b5_1": False,
}


def _packet(bars: list[Bar] | None = None):
    return build_packet(
        ROW,
        [_bar(10), _bar(13)] if bars is None else bars,
        domain="insider",
        sector_etf="XLP",
    )


def test_a_bar_stamped_before_the_eastern_day_belongs_to_the_eastern_session():
    """05:00 UTC is midnight Eastern in summer: the session is the same day.

    Dating a bar by its UTC date rather than its Eastern one would move a third
    of the calendar by a day, which is the kind of error that shifts a window
    without changing any value in it.
    """
    bar = Bar(
        timestamp=datetime(2026, 7, 13, 4, 0, tzinfo=UTC),  # 2026-07-13 00:00 EDT
        open=Decimal("1"),
        high=Decimal("1"),
        low=Decimal("1"),
        close=Decimal("1"),
        volume=Decimal("1"),
    )

    assert to_session_bar(bar).session == date(2026, 7, 13)


def test_the_builder_truncates_rather_than_relying_on_the_validator():
    """Bars past the event are dropped here, so no packet has to be refused."""
    packet = _packet([_bar(10), _bar(13), _bar(14), _bar(15)])

    assert [bar.session for bar in packet.bars] == [date(2026, 7, 10), date(2026, 7, 13)]


def test_a_packet_carries_only_the_named_extracted_fields():
    """A new column in the event store must not silently change every hash."""
    packet = build_packet(
        {**ROW, "an_unrelated_new_column": "x"},
        [_bar(10)],
        domain="insider",
        sector_etf="XLP",
    )

    assert "an_unrelated_new_column" not in packet.extracted
    assert packet.extracted["transaction_code"] == "P"


def test_adding_a_column_to_the_event_store_does_not_change_the_hash():
    """The property the explicit field list exists to guarantee."""
    with_extra = build_packet(
        {**ROW, "later_addition": 1}, [_bar(10)], domain="insider", sector_etf="XLP"
    )
    without = build_packet(ROW, [_bar(10)], domain="insider", sector_etf="XLP")

    assert with_extra.content_hash == without.content_hash


def test_the_benchmark_is_carried_in_the_packet():
    """A label computed against another benchmark is a different measurement."""
    assert _packet().issuer.sector_etf == "XLP"


def test_comparables_are_empty_because_retrieval_is_not_built():
    """Pinned so that the field's meaning changes visibly when retrieval lands."""
    assert _packet().comparables == ()


def test_statistics_are_absent_when_the_window_is_too_short():
    """Two bars cannot produce a 21-session return, and must not claim zero."""
    summary = _packet().market

    assert summary.trailing_return_21d is None
    assert summary.realised_volatility_21d is None
    assert summary.volume_percentile_63d is None


def test_a_trailing_return_is_measured_over_twenty_one_sessions():
    """Twenty-two closes bound twenty-one returns; the window starts at the 22nd."""
    bars = [_session(close="100", day=1)] + [_session(close="110", day=2) for _ in range(21)]

    assert market_summary(bars).trailing_return_21d == Decimal("0.1")


def test_a_window_one_session_short_reports_nothing_rather_than_a_shorter_window():
    """Silently measuring 20 sessions and calling it 21 is the failure here."""
    bars = [_session(close="100")] + [_session(close="110") for _ in range(20)]

    assert market_summary(bars).trailing_return_21d is None


def test_a_flat_series_has_no_volatility():
    """The negative control: a constant price must not manufacture movement."""
    bars = [_session(close="50") for _ in range(30)]

    assert market_summary(bars).realised_volatility_21d == Decimal(0)


def test_a_moving_series_has_volatility_above_zero():
    bars = [_session(close=str(50 + (i % 2))) for i in range(30)]

    volatility = market_summary(bars).realised_volatility_21d
    assert volatility is not None
    assert volatility > 0


def test_the_heaviest_session_ranks_at_the_top_of_its_window():
    bars = [_session(volume="100") for _ in range(62)] + [_session(volume="999")]

    assert market_summary(bars).volume_percentile_63d == Decimal(1)


def test_the_lightest_session_ranks_at_the_bottom_of_its_window():
    """One sixty-third, not zero: the session ranks at or below itself."""
    bars = [_session(volume="100") for _ in range(62)] + [_session(volume="1")]

    assert market_summary(bars).volume_percentile_63d == Decimal(1) / Decimal(63)


def test_volatility_matches_the_sample_standard_deviation_of_its_log_returns():
    """Pins the estimator, not just that it produces a positive number.

    Checked against `statistics.stdev`, which is the standard library's sample
    standard deviation and divides by n-1. Using it as the oracle keeps this from
    restating the implementation: it is an independent definition of the same
    published quantity, so the test would still fail if this module's arithmetic
    were rewritten to agree with itself.

    Written because changing the denominator from n-1 to n+1 passed the entire
    suite. A flat series has zero volatility under either, and a moving one is
    positive under either, so every existing test was blind to the estimator.
    """
    closes = [
        50,
        51,
        49,
        52,
        48,
        53,
        47,
        54,
        46,
        55,
        45,
        56,
        44,
        57,
        43,
        58,
        42,
        59,
        41,
        60,
        40,
        61,
    ]
    bars = [_session(close=str(value)) for value in closes]

    expected = stdev([log(b / a) for a, b in pairwise(closes)]) * sqrt(252)
    measured = market_summary(bars).realised_volatility_21d

    assert measured is not None
    assert float(measured) == pytest.approx(expected, rel=1e-9)


def test_a_non_positive_close_reports_no_return_rather_than_a_meaningless_one():
    """A zero close is not a price; dividing by it would invent a figure."""
    bars = [_session(close="0")] + [_session(close="50") for _ in range(21)]

    assert market_summary(bars).trailing_return_21d is None
    assert market_summary(bars).realised_volatility_21d is None


def test_the_packet_built_from_a_row_hashes_stably():
    assert _packet().content_hash == _packet().content_hash
