"""Daily ingestion.

Sequenced as fetch, extract, store, with no inference at any step. Each stage is
idempotent and keyed by date, so a failed run is re-run rather than repaired,
and re-running a completed day produces the same dataset.

A filing that cannot be timestamped is excluded and recorded by accession
number, never dropped silently and never dated by a substitute such as the
filing date, which carries no time of day and would place the event at
midnight. Recording is what keeps the exclusion honest: a gap nobody can
account for is indistinguishable from a defect, and enough of them still stops
the day, because EDGAR omitting acceptance times at scale is a break worth
failing on rather than absorbing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path

from arbiter.events.insider import (
    InsiderEvent,
    UnpriceableIssuerError,
    extract_insider_events,
    is_candidate,
)
from arbiter.events.redflags import RedFlagEvent, extract_redflag_event
from arbiter.ingestion.edgar import filings_with_objects
from arbiter.ingestion.store import write_events, write_unpriceable

INSIDER_FORM = "4"
REDFLAG_FORM = "8-K"

#: Failures that mean one filing could not be parsed, rather than that the
#: pipeline is broken. `AttributeError` belongs here because the filing client
#: raises it from inside its own rendering path for documents it cannot resolve,
#: and one such filing must not cost the day's other events. Anything outside
#: this set propagates: a quarantine that swallowed every exception would turn a
#: systemic break into a quiet day of zero events.
_PARSE_FAILURES = (ValueError, KeyError, AttributeError)


#: A day where this share of filings fails to parse is a systemic breakage, not
#: a handful of odd filers, and is raised rather than quarantined.
_SYSTEMIC_REJECTION_RATE = 0.05


class SystemicParseFailureError(RuntimeError):
    """Raised when so many filings fail to parse that the day is untrustworthy."""


def _untimestamped_into(rejected: list[dict[str, str]]) -> Callable[[str], None]:
    """Return a sink that records filings EDGAR gave no acceptance time for.

    They are counted as rejections rather than as an ordinary exclusion,
    because unlike an issuer with no listed stock this is the source failing to
    supply something it normally supplies — and the day's rejection rate is
    what catches that happening at scale.
    """

    def record(accession_no: str) -> None:
        rejected.append(
            {
                "accession_no": accession_no,
                "error": "EDGAR supplied no acceptance time; the filing cannot be "
                "placed in time and cannot become an event",
            }
        )

    return record


def _quarantine(
    rejected: list[dict[str, str]], root: Path, domain: str, day: date, attempted: int
) -> str | None:
    """Record filings that could not be parsed, and report a systemic share.

    Rejections are written rather than logged and forgotten: a filing dropped
    without a trace is indistinguishable from one that never existed, and the
    difference decides whether a day's event count means anything.

    Returns a description of the breakage rather than raising, so that every
    domain is judged before any of them stops the day. Raising here meant the
    first domain to fail masked the second, and a reader saw one problem where
    there were two.
    """
    path = root / "rejected" / domain / f"{day.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rejected, indent=1) + "\n")

    if attempted and len(rejected) / attempted > _SYSTEMIC_REJECTION_RATE:
        return (
            f"{len(rejected)} of {attempted} {domain} filings on {day.isoformat()} "
            f"failed to parse (see {path})"
        )
    return None


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
    insider_unpriceable: list[dict[str, str]] = []
    # Held apart from the other rejections only so that the day's denominator
    # can include them: they never reach the loop below, so counting them as
    # failures without counting them as attempts would overstate the rate.
    insider_untimestamped: list[dict[str, str]] = []
    insider_filings = filings_with_objects(
        INSIDER_FORM, day, _untimestamped_into(insider_untimestamped)
    )
    for record, filing in insider_filings:
        # One malformed filing must not cost the other nine hundred. The failure
        # is recorded with its accession number so it can be investigated, which
        # a bare `continue` would not allow.
        try:
            for event in extract_insider_events(record, filing.obj()):
                if is_candidate(event, min_value_usd):
                    insider_events.append(event)
        except UnpriceableIssuerError as exc:
            # Recorded apart from parse failures and excluded from the systemic
            # share below: an insider at a company with no listed common stock
            # files like any other, and that is a property of the population
            # rather than a sign the pipeline is broken.
            insider_unpriceable.append(
                {"accession_no": record.accession_no, "reason": str(exc)}
            )
        except _PARSE_FAILURES as exc:
            insider_rejected.append({"accession_no": record.accession_no, "error": str(exc)})

    redflag_events: list[RedFlagEvent] = []
    redflag_rejected: list[dict[str, str]] = []
    redflag_untimestamped: list[dict[str, str]] = []
    redflag_filings = filings_with_objects(
        REDFLAG_FORM, day, _untimestamped_into(redflag_untimestamped)
    )
    for record, filing in redflag_filings:
        try:
            event = extract_redflag_event(record, filing.obj())
        except _PARSE_FAILURES as exc:
            redflag_rejected.append({"accession_no": record.accession_no, "error": str(exc)})
            continue
        if event is not None:
            redflag_events.append(event)

    # A filing with no acceptance time was still a filing this day attempted, so
    # it belongs on both sides of the ratio. Unlistable issuers belong on
    # neither: leaving them in the denominator would understate the share of
    # genuinely unparseable filings and mask a real format break on a day that
    # happened to carry many of them.
    insider_rejected.extend(insider_untimestamped)
    redflag_rejected.extend(redflag_untimestamped)

    # Both domains are judged before either stops the day. They fail for
    # independent reasons, and raising inside the first check meant a day broken
    # in both was reported as broken in one.
    breakages = [
        _quarantine(
            insider_rejected,
            root,
            "insider",
            day,
            len(insider_filings) + len(insider_untimestamped) - len(insider_unpriceable),
        ),
        _quarantine(
            redflag_rejected,
            root,
            "redflag",
            day,
            len(redflag_filings) + len(redflag_untimestamped),
        ),
    ]
    named = [breakage for breakage in breakages if breakage is not None]
    if named:
        # Neither domain is written. A day is stored whole or not at all, so the
        # resume rule can treat a missing partition as work still to do; writing
        # the healthy domain would leave a half-day that looks complete to
        # nothing and costs a re-extraction anyway, since a retry reprocesses
        # both domains regardless.
        msg = (
            f"{'; '.join(named)}. This is a format or client breakage rather than "
            "a few odd filers, so the day is not recorded."
        )
        raise SystemicParseFailureError(msg)

    write_events(insider_events, root, "insider", day)
    write_events(redflag_events, root, "redflag", day)
    write_unpriceable(insider_unpriceable, root, "insider", day, stage="extraction")

    return {
        "insider": len(insider_events),
        "redflag": len(redflag_events),
        "rejected": len(insider_rejected) + len(redflag_rejected),
        "unpriceable": len(insider_unpriceable),
    }
