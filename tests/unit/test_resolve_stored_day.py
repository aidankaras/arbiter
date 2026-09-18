"""Reading stored events back out and turning them into stored labels.

These tests cover the adapter between the two stores: stored events arrive as
plain rows, and what the resolver needs is a typed request carrying the horizon
its domain implies.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.evaluation.resolution import HORIZONS, requests_from_rows
from arbiter.ingestion.market import Bar


def _row(accession: str, ticker: str = "MO", day: int = 3) -> dict[str, object]:
    return {
        "accession_no": accession,
        "as_of": datetime(2026, 8, day, 20, 47, tzinfo=UTC),
        "ticker": ticker,
        "cik": 764180,
        "transaction_code": "P",
    }


def test_rows_become_requests_carrying_their_domain_horizon():
    requests = requests_from_rows([_row("a-1")], domain="insider")

    assert len(requests) == 1
    assert requests[0].accession_no == "a-1"
    assert requests[0].horizon == HORIZONS["insider"]


def test_red_flag_events_carry_the_longer_horizon():
    """Post-announcement drift runs about a month, so the domains differ."""
    requests = requests_from_rows([_row("r-1")], domain="redflag")

    assert requests[0].horizon == HORIZONS["redflag"]
    assert requests[0].horizon > HORIZONS["insider"]


def test_an_unknown_domain_is_refused_rather_than_guessed():
    """A silent default horizon would measure the wrong window forever."""
    with pytest.raises(KeyError, match="earnings"):
        requests_from_rows([_row("a-1")], domain="earnings")


def test_the_acceptance_instant_survives_the_round_trip():
    requests = requests_from_rows([_row("a-1")], domain="insider")

    assert requests[0].as_of == datetime(2026, 8, 3, 20, 47, tzinfo=UTC)
    assert requests[0].as_of.tzinfo is not None


def test_rows_without_a_ticker_are_refused():
    """A row that cannot be priced must fail loudly, not resolve to nothing."""
    row = _row("a-1")
    del row["ticker"]

    with pytest.raises(KeyError, match="ticker"):
        requests_from_rows([row], domain="insider")


def test_no_rows_yield_no_requests():
    assert requests_from_rows([], domain="insider") == []


def test_red_flag_rows_lacking_a_ticker_column_are_refused():
    """Red-flag events carry an issuer but no ticker; that gap must be visible."""
    row = {
        "accession_no": "r-1",
        "as_of": datetime(2026, 8, 3, 20, 47, tzinfo=UTC),
        "cik": 1423689,
        "issuer": "AGNC Investment Corp.",
    }

    with pytest.raises(KeyError, match="ticker"):
        requests_from_rows([row], domain="redflag")


def _bars(first_day: int, closes: list[str]) -> list[Bar]:
    return [
        Bar(
            timestamp=datetime(2026, 8, first_day + offset, 4, tzinfo=UTC),
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
            volume=Decimal("1000"),
        )
        for offset, close in enumerate(closes)
    ]


def test_a_day_with_no_resolvable_events_still_records_a_partition(tmp_path: Path):
    """A day that was processed and found nothing is not an unprocessed day."""
    from arbiter.evaluation.resolution import resolve_stored_day
    from arbiter.ingestion.store import read_events, write_events

    write_events([], tmp_path, "insider", date(2026, 8, 28))

    written = resolve_stored_day(
        day=date(2026, 8, 28),
        domain="insider",
        root=tmp_path,
        fetch_bars=lambda symbols, start, end: {symbol: [] for symbol in symbols},
        benchmark_for=lambda cik: "XLP",
        today=date(2026, 9, 1),
    )

    assert written == 0
    assert read_events(tmp_path, "labels-insider", date(2026, 8, 28)) == []
