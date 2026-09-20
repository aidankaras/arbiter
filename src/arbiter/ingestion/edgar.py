"""EDGAR access.

Identity is configured from validated settings because the SEC throttles or
refuses requests that carry no contact string. Filings are converted into
records at this boundary, so that nothing downstream depends on the client
library's types and every event carries the instant it became public rather
than a calendar date.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from edgar import get_filings, set_identity

from arbiter.config import get_settings
from arbiter.ingestion.timestamps import MissingAcceptanceTimeError, acceptance_time_utc


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


#: The years the holiday table actually covers, derived from the table so the
#: two cannot drift apart.
_CALENDAR_YEARS = range(
    min(holiday.year for holiday in _MARKET_HOLIDAYS),
    max(holiday.year for holiday in _MARKET_HOLIDAYS) + 1,
)


class UncoveredCalendarError(LookupError):
    """Raised when a date falls outside the years the holiday table lists.

    Answering anyway would mean reporting every weekday holiday in that year as
    a trading day: Christmas 2024 fell on a Wednesday, and a table listing only
    2026 and 2027 would call it open.

    That error is the worst kind this project has — it varies with the calendar,
    so it would not thin a sample evenly but shift the session index of every
    event whose window spans the misclassified day, and only in some years.
    Refusing is the honest answer until the table is extended.
    """


def is_trading_day(day: date) -> bool:
    """Report whether the US equity market was open on a calendar day.

    Public because a backfill decides which days are worth fetching before it
    reaches EDGAR at all, and duplicating the holiday calendar to do so would
    let the two drift apart.

    Raises:
        UncoveredCalendarError: the date falls outside the years the holiday
            table lists, where a weekday closure would be reported as open.
    """
    if day.year not in _CALENDAR_YEARS:
        msg = (
            f"{day.isoformat()} falls outside the holiday calendar, which covers "
            f"{_CALENDAR_YEARS.start}-{_CALENDAR_YEARS.stop - 1}. Extend "
            "_MARKET_HOLIDAYS rather than treating an unlisted year as fully open: "
            "every weekday holiday in it would be counted as a trading session."
        )
        raise UncoveredCalendarError(msg)
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


def records_from_filings(
    filings: Iterable[Any],
) -> tuple[list[tuple[FilingRecord, Any]], list[str]]:
    """Convert client filings into records, reporting those that carry no time.

    Returns the records that could be timestamped, each with its source filing,
    and the accession numbers of those that could not. A filing appearing more
    than once in the day's index — as a jointly filed Form 4 does, once per
    reporting owner — yields one record.

    A filing with no acceptance time cannot be placed in time and so cannot
    become an event, but it is one filing among thousands accepted the same day.
    Converting them in a single pass meant the first such filing discarded the
    whole day, which is how a historical backfill lost days at a time. It is
    excluded and named instead, which is what the exclusion was always
    documented to be — and naming it is what keeps the exclusion from being
    silent.

    Raises:
        ValueError: an acceptance time arrived carrying a timezone. Unlike a
            missing one, that means the source's contract changed, and skipping
            it would shift every event in the day by hours without saying so.
    """
    records: list[tuple[FilingRecord, Any]] = []
    untimestamped: list[str] = []
    seen: set[str] = set()

    for filing in filings:
        accession_no = str(filing.accession_no)
        # A day's index lists a Form 4 once per reporting-owner CIK, so a filing
        # submitted jointly by three related entities appears three times. They
        # are the same document describing the same transactions, and ingesting
        # each appearance turned one transaction into three events.
        #
        # The duplicates are not harmless beyond the count. The CIK on each entry
        # is that entry's filer rather than the issuer, and the sector benchmark
        # is looked up by CIK — so a reporting owner resolved to no sector and
        # fell back to the broad market, leaving duplicate rows measured against
        # SPY while the issuer resolved to its sector ETF.
        if accession_no in seen:
            continue
        seen.add(accession_no)

        try:
            records.append((to_record(filing), filing))
        except MissingAcceptanceTimeError:
            untimestamped.append(accession_no)

    return records, untimestamped


def filings_with_objects(
    form: str, on: date, on_untimestamped: Callable[[str], None] | None = None
) -> list[tuple[FilingRecord, Any]]:
    """Return each filing of one form type accepted on one day, with its source.

    The source filing is returned alongside the record because the per-domain
    extractors need the parsed document, which only the client can produce.

    `on_untimestamped` receives the accession number of any filing that carries
    no acceptance time. Those are excluded from the result, and a caller that
    passes nothing is choosing not to record them.
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

    records, untimestamped = records_from_filings(filings)
    if on_untimestamped is not None:
        for accession_no in untimestamped:
            on_untimestamped(accession_no)
    return records


def daily_filings(form: str, on: date) -> list[FilingRecord]:
    """Return every filing of one form type accepted on one calendar day."""
    return [record for record, _ in filings_with_objects(form, on)]
