"""Daily ingestion.

Sequenced as fetch, extract, store, with no inference at any step. Each stage is
idempotent and keyed by date, so a failed run is re-run rather than repaired,
and re-running a completed day produces the same dataset.

A filing that cannot be timestamped raises rather than being skipped. An
untimestampable filing is a defect to investigate, and dropping it silently
would leave a gap that no later check could distinguish from a quiet day.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from pathlib import Path

from arbiter.events.insider import InsiderEvent, extract_insider_events, is_candidate
from arbiter.events.redflags import RedFlagEvent, extract_redflag_event
from arbiter.ingestion.edgar import filings_with_objects
from arbiter.ingestion.store import write_events

INSIDER_FORM = "4"
REDFLAG_FORM = "8-K"


#: A day where this share of filings fails to parse is a systemic breakage, not
#: a handful of odd filers, and is raised rather than quarantined.
_SYSTEMIC_REJECTION_RATE = 0.05


class SystemicParseFailureError(RuntimeError):
    """Raised when so many filings fail to parse that the day is untrustworthy."""


def _quarantine(
    rejected: list[dict[str, str]], root: Path, domain: str, day: date, attempted: int
) -> None:
    """Record filings that could not be parsed, and refuse a day that mostly failed.

    Rejections are written rather than logged and forgotten: a filing dropped
    without a trace is indistinguishable from one that never existed, and the
    difference decides whether a day's event count means anything.

    Raises:
        SystemicParseFailureError: too large a share of the day failed to parse.
    """
    path = root / "rejected" / domain / f"{day.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rejected, indent=1) + "\n")

    if attempted and len(rejected) / attempted > _SYSTEMIC_REJECTION_RATE:
        msg = (
            f"{len(rejected)} of {attempted} {domain} filings on {day.isoformat()} "
            "failed to parse; this is a format or client breakage rather than a few "
            f"odd filers, so the day is not recorded. See {path}"
        )
        raise SystemicParseFailureError(msg)


def ingest_day(day: date, root: Path, min_value_usd: Decimal) -> dict[str, int]:
    """Ingest one calendar day of filings into the event store.

    Args:
        day: the calendar day whose accepted filings are processed.
        root: directory holding the date-partitioned event store.
        min_value_usd: size floor below which an insider trade is not studied.

    Returns:
        Event counts by domain, after filtering.
    """
    insider_events: list[InsiderEvent] = []
    insider_rejected: list[dict[str, str]] = []
    insider_filings = filings_with_objects(INSIDER_FORM, day)
    for record, filing in insider_filings:
        # One malformed filing must not cost the other nine hundred. The failure
        # is recorded with its accession number so it can be investigated, which
        # a bare `continue` would not allow.
        try:
            for event in extract_insider_events(record, filing.obj()):
                if is_candidate(event, min_value_usd):
                    insider_events.append(event)
        except (ValueError, KeyError) as exc:
            insider_rejected.append({"accession_no": record.accession_no, "error": str(exc)})

    redflag_events: list[RedFlagEvent] = []
    redflag_rejected: list[dict[str, str]] = []
    redflag_filings = filings_with_objects(REDFLAG_FORM, day)
    for record, filing in redflag_filings:
        try:
            event = extract_redflag_event(record, filing.obj())
        except (ValueError, KeyError) as exc:
            redflag_rejected.append({"accession_no": record.accession_no, "error": str(exc)})
            continue
        if event is not None:
            redflag_events.append(event)

    _quarantine(insider_rejected, root, "insider", day, len(insider_filings))
    _quarantine(redflag_rejected, root, "redflag", day, len(redflag_filings))

    write_events(insider_events, root, "insider", day)
    write_events(redflag_events, root, "redflag", day)

    return {
        "insider": len(insider_events),
        "redflag": len(redflag_events),
        "rejected": len(insider_rejected) + len(redflag_rejected),
    }
