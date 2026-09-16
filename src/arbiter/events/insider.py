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
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict

from arbiter.ingestion.edgar import FilingRecord

OPEN_MARKET_CODES = frozenset({"P", "S"})

_REQUIRED_COLUMNS = (
    "Code",
    "Shares",
    "Price",
    "Value",
    "Ticker",
    "Issuer",
    "Insider",
    "Position",
    "Remaining Shares",
)


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
    price: Decimal
    value_usd: Decimal
    remaining_shares: Decimal
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
    return event.value_usd >= min_value_usd


def _decimal(row: dict[str, Any], column: str) -> Decimal:
    """Read one numeric cell as `Decimal`, preserving the reported digits.

    Raises:
        KeyError: the column is absent, meaning the upstream format changed.
    """
    return Decimal(str(row[column]))


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
        missing = [column for column in _REQUIRED_COLUMNS if column not in row]
        if missing:
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
                price=_decimal(row, "Price"),
                value_usd=_decimal(row, "Value"),
                remaining_shares=_decimal(row, "Remaining Shares"),
                is_10b5_1=is_plan_trade,
            )
        )
    return events
