"""Corporate red-flag events.

Three 8-K items carry the signal this domain studies: a change of certifying
accountant, a statement that previously issued financial statements should no
longer be relied upon, and the departure or appointment of a principal officer.

The information lives in narrative text rather than in fields. Whether an
auditor resigned or was dismissed, whether a disagreement was disclosed, and
what a restatement covers are all expressed in prose with no standard
structure, which is what makes this domain the natural test of whether language
models hold an advantage over models reading tables.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from arbiter.ingestion.edgar import FilingRecord

TRIGGER_ITEMS = frozenset({"4.01", "4.02", "5.02"})


class RedFlagEvent(BaseModel):
    """One current report carrying at least one red-flag item."""

    model_config = ConfigDict(frozen=True)

    accession_no: str
    as_of: datetime
    cik: int
    issuer: str
    item_codes: tuple[str, ...]
    item_text: str
    has_press_release: bool


def triggered_items(items: Iterable[str]) -> tuple[str, ...]:
    """Return the trigger item codes present, normalized and ordered.

    Labels arrive as `"Item 5.02"`. Only the numeric code is retained, so that
    grouping and comparison do not depend on how a label happens to be
    formatted, and the order is fixed so that two reports carrying the same
    items compare equal.
    """
    codes = {item.replace("Item ", "").strip() for item in items}
    return tuple(sorted(codes & TRIGGER_ITEMS))


def extract_redflag_event(record: FilingRecord, report: Any) -> RedFlagEvent | None:
    """Convert a current report into an event, or `None` when nothing triggers.

    A report carrying no trigger item produces no event rather than an empty
    one, so that a count of events is a count of the phenomenon being studied.
    """
    codes = triggered_items(report.items)
    if not codes:
        return None

    return RedFlagEvent(
        accession_no=record.accession_no,
        as_of=record.as_of,
        cik=record.cik,
        issuer=record.company,
        item_codes=codes,
        item_text=str(report.text()),
        has_press_release=bool(report.has_press_release),
    )
