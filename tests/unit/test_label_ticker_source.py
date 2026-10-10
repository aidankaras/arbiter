"""A stored label records where its ticker came from.

The marker is asserted on the label as written to the store, not on the
intermediate rows: an earlier version set it on a dict that the request builder
discarded, so every label silently lost it.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from arbiter.evaluation.resolution import resolve_stored_day
from arbiter.events.redflags import RedFlagEvent
from arbiter.ingestion.market import Bar
from arbiter.ingestion.store import read_events, write_events

DAY = date(2026, 8, 3)


def _bars() -> list[Bar]:
    return [
        Bar(
            timestamp=datetime(2026, 8, 4, 4, tzinfo=UTC) + timedelta(days=offset),
            open=Decimal(10),
            high=Decimal(10),
            low=Decimal(10),
            close=Decimal(10),
            volume=Decimal(1000),
        )
        for offset in range(30)
    ]


def _event(accession: str, ticker: str | None, source: str | None) -> RedFlagEvent:
    return RedFlagEvent(
        accession_no=accession,
        as_of=datetime(2026, 8, 3, 20, 47, tzinfo=UTC),
        cik=1423689,
        issuer="AGNC Investment Corp.",
        item_codes=("4.02",),
        item_text="Item 4.02",
        has_press_release=False,
        ticker=ticker,
        ticker_source=source,
    )


def test_stored_labels_say_whether_their_ticker_came_from_the_filing_or_the_map(tmp_path: Path):
    write_events(
        [_event("r-filing", "FBRX", "filing"), _event("r-map", None, None)],
        tmp_path,
        "redflag",
        DAY,
    )

    resolve_stored_day(
        day=DAY,
        domain="redflag",
        root=tmp_path,
        fetch_bars=lambda symbols, start, end: {symbol: _bars() for symbol in symbols},
        benchmark_for=lambda cik: "XLF",
        today=date(2026, 9, 30),
        ticker_for=lambda cik: "AGNC",
    )

    labels = {r["accession_no"]: r for r in read_events(tmp_path, "labels-redflag", DAY)}
    assert set(labels) == {"r-filing", "r-map"}, "premise: both events were labelled"
    assert labels["r-filing"]["ticker_source"] == "filing"
    assert labels["r-map"]["ticker_source"] == "current_map"


def test_a_label_without_a_source_is_still_readable(tmp_path: Path):
    """Insider events state their own ticker and carry no marker; so do old labels."""
    from tests.unit.stored_rows import event

    write_events([event("a-1")], tmp_path, "insider", DAY)

    resolve_stored_day(
        day=DAY,
        domain="insider",
        root=tmp_path,
        fetch_bars=lambda symbols, start, end: {symbol: _bars() for symbol in symbols},
        benchmark_for=lambda cik: "XLP",
        today=date(2026, 9, 30),
    )

    (label,) = read_events(tmp_path, "labels-insider", DAY)
    assert label["ticker_source"] is None
