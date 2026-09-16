from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.events.insider import InsiderEvent
from arbiter.ingestion.store import read_events, write_events

DAY = date(2026, 9, 11)


def _event(accession: str) -> InsiderEvent:
    return InsiderEvent(
        accession_no=accession,
        as_of=datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        cik=764180,
        ticker="MO",
        issuer="ALTRIA GROUP, INC.",
        insider_name="Kathryn B. Mcquade",
        position="Director",
        transaction_code="P",
        shares=Decimal("1500"),
        price=Decimal("67.62"),
        value_usd=Decimal("101430.0"),
        remaining_shares=Decimal("114929"),
        is_10b5_1=False,
    )


def test_events_round_trip_through_the_store(tmp_path: Path):
    write_events([_event("a-1"), _event("a-2")], tmp_path, "insider", DAY)
    rows = read_events(tmp_path, "insider", DAY)
    assert {row["accession_no"] for row in rows} == {"a-1", "a-2"}


def test_rerunning_a_day_replaces_that_day_exactly(tmp_path: Path):
    """The daily job is idempotent: re-running a date must not duplicate rows."""
    write_events([_event("a-1")], tmp_path, "insider", DAY)
    write_events([_event("a-1")], tmp_path, "insider", DAY)
    assert len(read_events(tmp_path, "insider", DAY)) == 1


def test_other_days_are_untouched(tmp_path: Path):
    write_events([_event("a-1")], tmp_path, "insider", date(2026, 9, 10))
    write_events([_event("b-1")], tmp_path, "insider", DAY)
    assert len(read_events(tmp_path, "insider", date(2026, 9, 10))) == 1
    assert len(read_events(tmp_path, "insider", DAY)) == 1


def test_domains_are_stored_separately(tmp_path: Path):
    write_events([_event("a-1")], tmp_path, "insider", DAY)
    write_events([], tmp_path, "redflag", DAY)
    assert len(read_events(tmp_path, "insider", DAY)) == 1
    assert read_events(tmp_path, "redflag", DAY) == []


def test_timestamps_survive_the_round_trip_as_the_same_instant(tmp_path: Path):
    """A timestamp that loses its zone in storage would misplace the event."""
    write_events([_event("a-1")], tmp_path, "insider", DAY)
    stored = read_events(tmp_path, "insider", DAY)[0]["as_of"]
    assert stored.tzinfo is not None
    assert stored == datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC)


def test_money_survives_the_round_trip_without_float_error(tmp_path: Path):
    write_events([_event("a-1")], tmp_path, "insider", DAY)
    stored = read_events(tmp_path, "insider", DAY)[0]
    assert stored["value_usd"] == Decimal("101430.0")
    assert stored["price"] == Decimal("67.62")


def test_writing_no_events_still_marks_the_day_as_processed(tmp_path: Path):
    """A quiet day and an unprocessed day must not look the same."""
    path = write_events([], tmp_path, "redflag", date(2026, 9, 12))
    assert path.exists()
    assert read_events(tmp_path, "redflag", date(2026, 9, 12)) == []


def test_reading_an_unprocessed_day_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="never processed"):
        read_events(tmp_path, "insider", date(2026, 9, 14))
