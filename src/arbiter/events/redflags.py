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

import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from arbiter.ingestion.edgar import CoverSecurity, FilingRecord
from arbiter.ingestion.market import tradeable_symbol

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
    # The symbol the filing's own cover page states, so labelling does not depend
    # on SEC's *current* ticker map: an issuer that delists after filing drops
    # out of that map, and auditor changes, restatements and officer departures
    # predict delisting, which would exclude events in correlation with the
    # outcome. `None` when the filing carries no usable symbol, in which case
    # labelling falls back to the current map.
    ticker: str | None = None
    # "filing" when `ticker` was read from the filing; `None` otherwise.
    ticker_source: str | None = None


#: Item labels vary in case, spacing, and sub-letter. Item 4.02 is nearly always
#: filed as 4.02(a) or 4.02(b), and 5.02 as 5.02(b) through (e), so matching the
#: numeric prefix rather than the whole label is what keeps those events in the
#: population at all.
_ITEM_LABEL = re.compile(r"\s*item\s*(\d+\.\d+)", re.IGNORECASE)


def triggered_items(items: Iterable[str]) -> tuple[str, ...]:
    """Return the trigger item codes present, normalized and ordered.

    Labels arrive in several shapes — `"Item 5.02"`, `"ITEM 5.02"`,
    `"Item 5.02(b)"` — and only the numeric code is retained, so that grouping
    and comparison do not depend on formatting. The order is fixed so that two
    reports carrying the same items compare equal.
    """
    codes: set[str] = set()
    for item in items:
        match = _ITEM_LABEL.match(str(item))
        if match:
            codes.add(match.group(1))
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
        item_text=_item_text(report),
        has_press_release=bool(report.has_press_release),
    )


#: Titles that name the common equity. A title naming a warrant, right, preferred
#: share or depositary interest in one also contains "common" ("Warrants to
#: purchase one share of Class A Common Stock"), and those are not the stock this
#: study prices.
_COMMON_TITLE = re.compile(r"\bcommon\b", re.IGNORECASE)
_NOT_COMMON_EQUITY = re.compile(
    r"warrant|right|preferred|depositary|note|debenture|subordinated", re.IGNORECASE
)


def choose_filing_ticker(securities: Iterable[CoverSecurity]) -> str | None:
    """Pick the symbol of the issuer's common equity from a cover page.

    A cover page lists one row per registered security, so an issuer with listed
    preferreds or warrants states several symbols, and the first is not reliably
    the common stock. Order of preference:

    1. a symbol whose paired security title names common stock;
    2. otherwise the first symbol `tradeable_symbol` accepts;
    3. otherwise `None`.

    Rule 2 is shape-only: `tradeable_symbol` accepts a preferred such as `AGNCN`
    because it cannot tell one from a common symbol by spelling. It is reached
    only when no title identifies the common stock, typically because the filer
    tagged symbols without titles.
    """
    candidates: list[tuple[str, str | None]] = []
    for security in securities:
        symbol = tradeable_symbol(security.symbol)
        if symbol is not None:
            candidates.append((symbol, security.title))

    for symbol, title in candidates:
        if title and _COMMON_TITLE.search(title) and not _NOT_COMMON_EQUITY.search(title):
            return symbol
    return candidates[0][0] if candidates else None


def with_filing_ticker(
    event: RedFlagEvent, securities: Iterable[CoverSecurity]
) -> RedFlagEvent:
    """Return the event carrying the ticker its filing states, if it states one."""
    ticker = choose_filing_ticker(securities)
    if ticker is None:
        return event
    return event.model_copy(update={"ticker": ticker, "ticker_source": "filing"})


def _item_text(report: Any) -> str:
    """Return the report's narrative text, or empty when it cannot be rendered.

    A small number of filings expose no primary document the client can render,
    and asking for their text raises from inside the library. The item codes are
    what make an event an event, and they come from the filing index rather than
    the body, so a report whose prose is unavailable is still a real event with
    a real timestamp.

    The absence is recorded as empty text rather than hidden: an arm reading a
    packet sees nothing to read, which is the truth, and the text-dependence of
    this domain means such events are worth counting separately before any claim
    rests on them.
    """
    try:
        return str(report.text())
    except (AttributeError, ValueError, TypeError):
        return ""
