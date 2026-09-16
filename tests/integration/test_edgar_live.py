from datetime import date

import pytest

from arbiter.ingestion.edgar import daily_filings, filings_with_objects

pytestmark = pytest.mark.integration

# A settled trading day. Using a fixed past date keeps the assertions stable:
# EDGAR does not revise the filings accepted on a completed day.
SETTLED_DAY = date(2026, 9, 11)


def test_a_settled_day_returns_form_4_filings():
    records = daily_filings("4", SETTLED_DAY)
    assert len(records) > 100
    assert all(record.form.startswith("4") for record in records)


def test_every_record_carries_an_aware_timestamp():
    """A naive timestamp downstream would silently misplace an event in time."""
    records = daily_filings("4", SETTLED_DAY)
    assert all(record.as_of.tzinfo is not None for record in records)


def test_acceptance_times_fall_within_edgar_operating_hours():
    """EDGAR accepts filings from 06:00 to 22:00 Eastern; anything else signals a bug."""
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo("America/New_York")
    hours = {
        record.as_of.astimezone(eastern).hour for record in daily_filings("4", SETTLED_DAY)
    }
    assert hours
    assert min(hours) >= 6
    assert max(hours) <= 22


def test_filings_are_returned_with_their_source_document():
    pairs = filings_with_objects("8-K", SETTLED_DAY)
    assert pairs
    record, filing = pairs[0]
    assert record.accession_no == str(filing.accession_no)
