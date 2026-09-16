"""Insider transaction events.

Only open-market purchases and sales carry information about a view. Trades
executed under a pre-established 10b5-1 plan were scheduled months earlier, and
option exercises, grants, gifts, and tax withholding reflect compensation
mechanics rather than a decision to buy or sell. Counting any of them as signal
is a documented route to apparent alpha that does not survive correction.

Every event inherits the filing's acceptance instant rather than the
transaction date: the market learns of an insider's trade when the filing is
accepted, which is often days later.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict

from arbiter.ingestion.edgar import FilingRecord

OPEN_MARKET_CODES = frozenset({"P", "S"})

# Columns describing a transaction. A filing that reports no transactions, such
# as an amendment carrying only a remark, has none of these while still carrying
# the filer's identity, so presence is tested against this group alone.
_TRANSACTION_COLUMNS = (
    "Code",
    "Shares",
    "Price",
    "Value",
    "Remaining Shares",
)

_IDENTITY_COLUMNS = (
    "Ticker",
    "Issuer",
    "Insider",
    "Position",
)

_REQUIRED_COLUMNS = _TRANSACTION_COLUMNS + _IDENTITY_COLUMNS


class InsiderEvent(BaseModel):
    """One reported insider transaction, stamped with when it became public."""

    model_config = ConfigDict(frozen=True)

    accession_no: str
    as_of: datetime
    cik: int
    ticker: str
    issuer: str
    insider_name: str
    position: str
    transaction_code: str
    shares: Decimal
    # Not every reported transaction carries a price. A gift, for instance, moves
    # shares with no consideration, and the filing leaves the price and value
    # blank. Those events are recorded as filed, with no price, rather than being
    # assigned a zero that would read as a free purchase.
    price: Decimal | None
    value_usd: Decimal | None
    # Derivative rows frequently report no post-transaction holding. Recording a
    # zero would assert that the insider now holds nothing, which is a different
    # and materially wrong statement from "the filing did not say".
    remaining_shares: Decimal | None
    is_10b5_1: bool


def is_candidate(event: InsiderEvent, min_value_usd: Decimal) -> bool:
    """Report whether an event belongs in the studied population.

    The rule is versioned with the code. Changing it changes which events every
    published comparison is computed over, so it is a reviewed change rather
    than a tuning knob.
    """
    if event.is_10b5_1:
        return False
    if event.transaction_code not in OPEN_MARKET_CODES:
        return False
    if event.value_usd is None:
        return False
    return event.value_usd >= min_value_usd


def _decimal(row: dict[str, Any], column: str) -> Decimal:
    """Read one required numeric cell as `Decimal`, preserving reported digits.

    Raises:
        KeyError: the column is absent, meaning the upstream format changed.
        decimal.InvalidOperation: the cell holds no usable number where one is
            always reported, which is a parsing defect rather than a data shape.
    """
    return Decimal(str(row[column]))


def _optional_decimal(row: dict[str, Any], column: str) -> Decimal | None:
    """Read one numeric cell that a filing may legitimately leave blank.

    Returns `None` when the cell is empty or not a number, which is how filings
    report transactions carrying no price, such as gifts. The absence is
    preserved rather than replaced, because a zero price would be indexed as a
    free acquisition and would distort any statistic computed over it.
    """
    raw = str(row[column]).strip()
    if raw == "" or raw.lower() in {"nan", "none", "null"}:
        return None
    try:
        return Decimal(raw)
    except InvalidOperation:
        return None


def extract_insider_events(record: FilingRecord, form4: Any) -> list[InsiderEvent]:
    """Convert one parsed Form 4 into events, one per reported transaction.

    Raises:
        KeyError: the parsed table lacks a column an event requires. The filing
            format changed and must be re-examined rather than worked around,
            because a missing column would otherwise yield events with
            substituted values.
    """
    frame = form4.to_dataframe()
    is_plan_trade = bool(form4.aff10b5_one)

    events: list[InsiderEvent] = []
    for row in frame.to_dict(orient="records"):
        reports_a_transaction = any(column in row for column in _TRANSACTION_COLUMNS)
        if not reports_a_transaction:
            # A Form 4 may report no transactions at all, carrying only the
            # filer's identity and a remark. Such a filing describes no event,
            # which is different from a filing whose format has changed.
            continue

        missing = [column for column in _REQUIRED_COLUMNS if column not in row]
        if missing:
            # Some transaction columns but not others: the format changed, and
            # continuing would silently emit events built from partial data.
            msg = f"Form 4 table is missing {missing}; the upstream format changed"
            raise KeyError(msg)

        events.append(
            InsiderEvent(
                accession_no=record.accession_no,
                as_of=record.as_of,
                cik=record.cik,
                ticker=str(row["Ticker"]),
                issuer=str(row["Issuer"]),
                insider_name=str(row["Insider"]),
                position=str(row["Position"]),
                transaction_code=str(row["Code"]),
                shares=_decimal(row, "Shares"),
                price=_optional_decimal(row, "Price"),
                value_usd=_optional_decimal(row, "Value"),
                remaining_shares=_optional_decimal(row, "Remaining Shares"),
                is_10b5_1=is_plan_trade,
            )
        )
    return events
