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


class EmptyTradingDayError(RuntimeError):
    """Raised when EDGAR returns no index for a day the market was open.

    An empty result is ambiguous: it means either that no filing of this form
    was accepted, or that the index could not be read. Accepting the ambiguity
    would let a transient failure be recorded as a complete, permanently empty
    day, which no later check could distinguish from a genuinely quiet one.
    """


# Days the US equity market is closed, so EDGAR legitimately holds no filings.
# Weekend closures are derived; these are the scheduled holidays through 2027.
_MARKET_HOLIDAYS = frozenset(
    {
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
        date(2027, 1, 1),
        date(2027, 1, 18),
        date(2027, 2, 15),
        date(2027, 3, 26),
        date(2027, 5, 31),
        date(2027, 6, 18),
        date(2027, 7, 5),
        date(2027, 9, 6),
        date(2027, 11, 25),
        date(2027, 12, 24),
    }
)


def is_trading_day(day: date) -> bool:
    """Report whether the US equity market was open on a calendar day.

    Public because a backfill decides which days are worth fetching before it
    reaches EDGAR at all, and duplicating the holiday calendar to do so would
    let the two drift apart.
    """
    return day.weekday() < 5 and day not in _MARKET_HOLIDAYS


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
    if filings is None:
        # A day EDGAR holds no filings for and a day whose index could not be
        # read are indistinguishable here, and treating both as empty would
        # record a failed fetch as a complete, permanently empty day. Only a
        # non-trading day is accepted as legitimately empty.
        if is_trading_day(on):
            msg = (
                f"EDGAR returned no {form} index for {on.isoformat()}, which is a "
                "trading day; treat this as a failed fetch and retry rather than "
                "recording the day as empty"
            )
            raise EmptyTradingDayError(msg)
        return []
    return [(to_record(filing), filing) for filing in filings]


def daily_filings(form: str, on: date) -> list[FilingRecord]:
    """Return every filing of one form type accepted on one calendar day."""
    return [record for record, _ in filings_with_objects(form, on)]
