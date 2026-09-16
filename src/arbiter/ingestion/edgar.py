"""EDGAR access.

Identity is configured from validated settings because the SEC throttles or
refuses requests that carry no contact string. Filings are converted into
records at this boundary, so that nothing downstream depends on the client
library's types and every event carries the instant it became public rather
than a calendar date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from edgar import get_filings, set_identity

from arbiter.config import get_settings
from arbiter.ingestion.timestamps import acceptance_time_utc


@dataclass(frozen=True)
class FilingRecord:
    """One filing's identity and the instant it became publicly available."""

    accession_no: str
    form: str
    cik: int
    company: str
    as_of: datetime
    filing_date: date


def configure_identity() -> None:
    """Register the contact string the SEC requires on every request."""
    set_identity(get_settings().sec_user_agent)


def to_record(filing: Any) -> FilingRecord:
    """Convert a client filing into a `FilingRecord`.

    Raises:
        MissingAcceptanceTimeError: the filing carries no acceptance time, so it
            cannot be placed in time and must not enter the event store.
    """
    return FilingRecord(
        accession_no=str(filing.accession_no),
        form=str(filing.form),
        cik=int(filing.cik),
        company=str(filing.company),
        as_of=acceptance_time_utc(filing.header.acceptance_datetime),
        filing_date=filing.filing_date,
    )


def filings_with_objects(form: str, on: date) -> list[tuple[FilingRecord, Any]]:
    """Return each filing of one form type accepted on one day, with its source.

    The source filing is returned alongside the record because the per-domain
    extractors need the parsed document, which only the client can produce.
    """
    configure_identity()
    filings = get_filings(form=form, filing_date=on.isoformat())
    return [(to_record(filing), filing) for filing in filings]


def daily_filings(form: str, on: date) -> list[FilingRecord]:
    """Return every filing of one form type accepted on one calendar day."""
    return [record for record, _ in filings_with_objects(form, on)]
