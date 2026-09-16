from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.ingestion.pipeline import ingest_day
from arbiter.ingestion.store import read_events

pytestmark = pytest.mark.integration

# A settled trading day: EDGAR does not revise the filings accepted on a
# completed day, so these assertions stay stable over time.
SETTLED_DAY = date(2026, 9, 11)
MIN_VALUE = Decimal("50000")


def test_ingesting_a_settled_day_produces_events_in_both_domains(tmp_path: Path):
    counts = ingest_day(SETTLED_DAY, tmp_path, MIN_VALUE)

    assert counts["insider"] > 0
    assert counts["redflag"] > 0
    assert set(counts) == {"insider", "redflag"}


def test_stored_insider_events_respect_the_filter_rules(tmp_path: Path):
    """Every stored trade must be an open-market trade above the size floor."""
    ingest_day(SETTLED_DAY, tmp_path, MIN_VALUE)

    rows = read_events(tmp_path, "insider", SETTLED_DAY)
    assert rows
    assert all(row["transaction_code"] in {"P", "S"} for row in rows)
    assert all(row["is_10b5_1"] is False for row in rows)
    assert all(row["value_usd"] >= MIN_VALUE for row in rows)


def test_stored_redflag_events_all_carry_a_trigger_item(tmp_path: Path):
    ingest_day(SETTLED_DAY, tmp_path, MIN_VALUE)

    rows = read_events(tmp_path, "redflag", SETTLED_DAY)
    assert rows
    for row in rows:
        assert row["item_codes"]
        assert set(row["item_codes"]) <= {"4.01", "4.02", "5.02"}


def test_events_carry_the_acceptance_instant_not_midnight(tmp_path: Path):
    """A date-only timestamp would place every event at midnight and mis-sequence them."""
    ingest_day(SETTLED_DAY, tmp_path, MIN_VALUE)

    rows = read_events(tmp_path, "insider", SETTLED_DAY)
    assert all(row["as_of"].tzinfo is not None for row in rows)
    assert any(row["as_of"].hour != 0 for row in rows)


def test_rerunning_the_day_leaves_the_same_counts(tmp_path: Path):
    """The daily job is idempotent, so a repeated run must not duplicate events."""
    first = ingest_day(SETTLED_DAY, tmp_path, MIN_VALUE)
    second = ingest_day(SETTLED_DAY, tmp_path, MIN_VALUE)

    assert first == second
    assert len(read_events(tmp_path, "insider", SETTLED_DAY)) == first["insider"]
