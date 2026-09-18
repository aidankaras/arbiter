"""A filing that cannot be placed in time must not cost the day around it.

An event's acceptance instant is what makes it point-in-time, so a filing
lacking one cannot enter the event store. The question is what happens to the
three thousand filings accepted alongside it. Building the day's records in one
pass meant the first untimestamped filing discarded all of them, which cost
entire days of a historical backfill.

Such a filing is now excluded and recorded by accession number, which is what
the exclusion was always documented to be. Recorded is not silent: the day's
rejection file names it, and enough of them still stops the day, because EDGAR
omitting acceptance times at scale is a break worth failing on.
"""

from datetime import UTC, date, datetime

import pytest

from arbiter.ingestion.edgar import records_from_filings


class _Filing:
    """The attributes `to_record` reads from a client filing."""

    def __init__(self, accession: str, acceptance: datetime | None) -> None:
        self.accession_no = accession
        self.form = "4"
        self.cik = 764180
        self.company = "ALTRIA GROUP, INC."
        self.filing_date = date(2026, 3, 6)
        self.header = type("Header", (), {"acceptance_datetime": acceptance})()


ACCEPTED = datetime(2026, 3, 6, 16, 12, 3)


def test_a_timestamped_filing_becomes_a_record():
    records, skipped = records_from_filings([_Filing("a-1", ACCEPTED)])

    assert skipped == []
    assert records[0][0].accession_no == "a-1"
    assert records[0][0].as_of == datetime(2026, 3, 6, 21, 12, 3, tzinfo=UTC)


def test_one_untimestamped_filing_does_not_discard_the_others():
    """The defect that cost whole days of a backfill."""
    filings = [
        _Filing("a-1", ACCEPTED),
        _Filing("a-2", None),
        _Filing("a-3", ACCEPTED),
    ]

    records, skipped = records_from_filings(filings)

    assert [record.accession_no for record, _ in records] == ["a-1", "a-3"]
    assert skipped == ["a-2"]


def test_the_skipped_filing_is_named_so_it_can_be_investigated():
    """A count alone cannot be checked against EDGAR afterwards."""
    _, skipped = records_from_filings([_Filing("0000320193-26-000042", None)])

    assert skipped == ["0000320193-26-000042"]


def test_the_source_filing_travels_with_its_record():
    """Extraction needs the parsed document, which only the client can produce."""
    filing = _Filing("a-1", ACCEPTED)

    records, _ = records_from_filings([filing])

    assert records[0][1] is filing


def test_a_day_where_every_filing_lacks_a_time_yields_nothing_and_says_so():
    records, skipped = records_from_filings([_Filing("a-1", None), _Filing("a-2", None)])

    assert records == []
    assert skipped == ["a-1", "a-2"]


def test_no_filings_yield_nothing():
    assert records_from_filings([]) == ([], [])


def test_an_acceptance_time_carrying_a_timezone_still_stops_the_day():
    """Only a missing time is tolerated, and only because it is a known gap.

    EDGAR publishes acceptance times naive and Eastern. One arriving with a
    timezone means the source's contract changed, which would silently shift
    every event in the day by hours if it were skipped like a missing one.
    """
    aware = _Filing("a-1", datetime(2026, 3, 6, 16, 12, 3, tzinfo=UTC))

    with pytest.raises(ValueError, match="naive Eastern"):
        records_from_filings([aware])
