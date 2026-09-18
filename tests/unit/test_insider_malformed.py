"""Rows that are present but malformed must not quietly leave the population.

Every case here was observed or constructed from real Form 4 data. The common
thread is that each one previously produced a plausible-looking result — an
event with a substituted value, or no event at all — rather than an error.
"""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.events.insider import (
    UnpriceableIssuerError,
    extract_insider_events,
    is_candidate,
)
from arbiter.ingestion.edgar import FilingRecord

FIXTURE = Path(__file__).parents[1] / "fixtures" / "form4_open_market.json"
MIN_VALUE = Decimal("50000")


class _Form4:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.aff10b5_one = payload["aff10b5_one"]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self._payload["rows"])


def _record() -> FilingRecord:
    return FilingRecord(
        accession_no="0000764180-26-000102",
        form="4",
        cik=764180,
        company="ALTRIA GROUP, INC.",
        as_of=datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        filing_date=date(2026, 9, 11),
    )


def _payload(**row_overrides: object) -> dict:
    payload = json.loads(FIXTURE.read_text())
    payload["rows"] = [{**payload["rows"][0], **row_overrides}]
    return payload


def test_a_holdings_row_beside_a_real_trade_does_not_abort_the_filing():
    """One blank row used to raise and kill the whole day's ingestion."""
    payload = json.loads(FIXTURE.read_text())
    real = payload["rows"][0]
    holdings = dict.fromkeys(real, "")
    holdings.update({"Ticker": real["Ticker"], "Issuer": real["Issuer"]})
    holdings.update({"Insider": real["Insider"], "Position": real["Position"]})
    payload["rows"] = [real, holdings]

    events = extract_insider_events(_record(), _Form4(payload))

    assert len(events) == 1
    assert events[0].transaction_code == "P"


def test_a_thousands_separator_raises_rather_than_dropping_the_trade():
    """Treating it as blank would drop a qualifying purchase, with a filer bias."""
    with pytest.raises(ValueError, match="unparseable"):
        extract_insider_events(_record(), _Form4(_payload(Value="101,430.00")))


def test_a_currency_symbol_raises_rather_than_dropping_the_price():
    with pytest.raises(ValueError, match="unparseable"):
        extract_insider_events(_record(), _Form4(_payload(Price="$67.62")))


def test_a_non_finite_share_count_raises():
    """`Decimal("Infinity")` parses without complaint, so finiteness is checked.

    A bare "nan" is how pandas renders an empty cell, and is treated as blank
    above; this covers a value that is genuinely present and genuinely unusable.
    """
    with pytest.raises(ValueError, match="finite"):
        extract_insider_events(_record(), _Form4(_payload(Shares="Infinity")))


def test_a_blank_share_count_raises_where_a_value_is_always_reported():
    with pytest.raises(ValueError, match="Shares"):
        extract_insider_events(_record(), _Form4(_payload(Shares="", Code="P")))


def test_a_missing_ticker_raises_rather_than_becoming_a_string():
    """It previously became the string 'None' and entered the studied population.

    Raised as an unpriceable issuer rather than a parse failure: the row is
    well formed, and the filer simply has no listed common stock.
    """
    with pytest.raises(UnpriceableIssuerError, match="Ticker"):
        extract_insider_events(_record(), _Form4(_payload(Ticker=None)))


def test_a_well_formed_row_still_produces_a_candidate():
    """Regression guard: the stricter parsing must not reject valid filings."""
    events = extract_insider_events(_record(), _Form4(_payload()))

    assert len(events) == 1
    assert is_candidate(events[0], MIN_VALUE)
