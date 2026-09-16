from datetime import UTC, date, datetime

import pytest

from arbiter.ingestion.edgar import FilingRecord, to_record
from arbiter.ingestion.timestamps import MissingAcceptanceTimeError


class _Header:
    acceptance_datetime = datetime(2026, 9, 11, 16, 47, 39)


class _Filing:
    accession_no = "0001610717-26-000414"
    form = "4"
    cik = 1770787
    company = "10x Genomics, Inc."
    filing_date = date(2026, 9, 11)
    header = _Header()


def test_a_filing_becomes_a_record_stamped_in_utc():
    assert to_record(_Filing()) == FilingRecord(
        accession_no="0001610717-26-000414",
        form="4",
        cik=1770787,
        company="10x Genomics, Inc.",
        as_of=datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        filing_date=date(2026, 9, 11),
    )


def test_the_record_carries_the_instant_not_the_calendar_day():
    """A filing date has no time of day, so it cannot decide which session trades."""
    record = to_record(_Filing())
    assert record.as_of.tzinfo is not None
    assert record.as_of.date() == date(2026, 9, 11)


def test_a_filing_without_an_acceptance_time_raises():
    class _NoTime(_Filing):
        header = type("_H", (), {"acceptance_datetime": None})()

    with pytest.raises(MissingAcceptanceTimeError):
        to_record(_NoTime())


def test_records_are_immutable():
    """A record's identity and timestamp must not drift after construction."""
    record = to_record(_Filing())
    with pytest.raises(AttributeError):
        record.as_of = datetime(2020, 1, 1, tzinfo=UTC)
