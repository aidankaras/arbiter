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
from arbiter.ingestion.market import tradeable_symbol

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


class UnpriceableIssuerError(Exception):
    """Raised when a filing's issuer has no security this project can price.

    Deliberately not a parse failure. The filing was read correctly and the
    filer is real — it simply has no listed common stock, which is an ordinary
    property of the filing population rather than evidence that anything is
    broken. Insiders at companies with only registered debt, and at issuers
    between registration and listing, file Form 4 like anyone else.

    The distinction matters because ingestion aborts a day whose filings mostly
    fail to parse, on the reasoning that a format or client change has broken
    it. Counting unlistable issuers toward that share would let an unremarkable
    day trip a guard meant for breakage.
    """


_BLANK = frozenset({"", "nan", "none", "null", "<na>", "nat"})


def _is_blank(value: Any) -> bool:
    """Report whether a cell holds no value at all."""
    return str(value).strip().lower() in _BLANK


def _decimal(row: dict[str, Any], column: str, accession_no: str) -> Decimal:
    """Read one always-reported numeric cell as `Decimal`.

    Raises:
        KeyError: the column is absent, meaning the upstream format changed.
        ValueError: the cell is blank or unparseable where a number is always
            reported, or holds a non-finite value. `Decimal("nan")` parses
            without complaint, so finiteness is checked explicitly.
    """
    raw = str(row[column]).strip()
    if _is_blank(raw):
        msg = f"{column} is blank in {accession_no}, where a value is always reported"
        raise ValueError(msg)
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        msg = f"{column} in {accession_no} is not a number: {raw!r}"
        raise ValueError(msg) from exc
    if not value.is_finite():
        msg = f"{column} in {accession_no} is not finite: {raw!r}"
        raise ValueError(msg)
    return value


def _optional_decimal(row: dict[str, Any], column: str, accession_no: str) -> Decimal | None:
    """Read one numeric cell that a filing may legitimately leave blank.

    A blank cell yields `None`: that is how filings report a transaction with no
    price, such as a gift, and the absence is preserved rather than replaced,
    because a zero would be indexed as a free acquisition.

    A cell that is present but unparseable raises instead. Treating it as blank
    would quietly drop a real transaction from the studied population, and
    formatting anomalies are not distributed evenly across filers, so the loss
    would carry a bias no later test could detect.

    Raises:
        ValueError: the cell is present but not a finite number.
    """
    raw = str(row[column]).strip()
    if _is_blank(raw):
        return None
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        msg = f"{column} in {accession_no} is present but unparseable: {raw!r}"
        raise ValueError(msg) from exc
    if not value.is_finite():
        msg = f"{column} in {accession_no} is not finite: {raw!r}"
        raise ValueError(msg)
    return value


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
        # Presence of a *value*, not of a column: every row in a frame carries
        # the same keys, so a column-presence test can never filter the holdings
        # rows it is meant to skip.
        reports_a_transaction = any(
            column in row and not _is_blank(row[column]) for column in _TRANSACTION_COLUMNS
        )
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

        reported = str(row["Ticker"]).strip()
        ticker = tradeable_symbol(reported)
        if ticker is None:
            # Validated against what a symbol looks like rather than against a
            # list of placeholder spellings. The filing client renders a missing
            # ticker variously as "N/A", "None", or empty, and a denylist loses
            # to whichever spelling it has not met yet.
            msg = (
                f"Ticker {reported!r} in {record.accession_no} is not a symbol; "
                "the event cannot be priced"
            )
            raise UnpriceableIssuerError(msg)

        events.append(
            InsiderEvent(
                accession_no=record.accession_no,
                as_of=record.as_of,
                cik=record.cik,
                ticker=ticker,
                issuer=str(row["Issuer"]),
                insider_name=str(row["Insider"]),
                position=str(row["Position"]),
                transaction_code=str(row["Code"]).strip(),
                shares=_decimal(row, "Shares", record.accession_no),
                price=_optional_decimal(row, "Price", record.accession_no),
                value_usd=_optional_decimal(row, "Value", record.accession_no),
                remaining_shares=_optional_decimal(
                    row, "Remaining Shares", record.accession_no
                ),
                is_10b5_1=is_plan_trade,
            )
        )
    return events
