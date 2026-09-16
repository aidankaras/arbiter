"""Daily ingestion.

Sequenced as fetch, extract, store, with no inference at any step. Each stage is
idempotent and keyed by date, so a failed run is re-run rather than repaired,
and re-running a completed day produces the same dataset.

A filing that cannot be timestamped raises rather than being skipped. An
untimestampable filing is a defect to investigate, and dropping it silently
would leave a gap that no later check could distinguish from a quiet day.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from arbiter.events.insider import InsiderEvent, extract_insider_events, is_candidate
from arbiter.events.redflags import RedFlagEvent, extract_redflag_event
from arbiter.ingestion.edgar import filings_with_objects
from arbiter.ingestion.store import write_events

INSIDER_FORM = "4"
REDFLAG_FORM = "8-K"


def summarize_counts(*, insider: int, redflag: int) -> dict[str, int]:
    """Report per-domain event counts, including zeros.

    Zero counts are reported rather than omitted: a domain missing from the
    summary would make a broken extractor look like a quiet day.
    """
    return {"insider": insider, "redflag": redflag}


def ingest_day(day: date, root: Path, min_value_usd: Decimal) -> dict[str, int]:
    """Ingest one calendar day of filings into the event store.

    Args:
        day: the calendar day whose accepted filings are processed.
        root: directory holding the date-partitioned event store.
        min_value_usd: size floor below which an insider trade is not studied.

    Returns:
        Event counts by domain, after filtering.
    """
    insider_events: list[InsiderEvent] = [
        event
        for record, filing in filings_with_objects(INSIDER_FORM, day)
        for event in extract_insider_events(record, filing.obj())
        if is_candidate(event, min_value_usd)
    ]

    redflag_events: list[RedFlagEvent] = []
    for record, filing in filings_with_objects(REDFLAG_FORM, day):
        event = extract_redflag_event(record, filing.obj())
        if event is not None:
            redflag_events.append(event)

    write_events(insider_events, root, "insider", day)
    write_events(redflag_events, root, "redflag", day)

    return summarize_counts(insider=len(insider_events), redflag=len(redflag_events))
