"""A ticker is validated against what a symbol looks like, not against a list.

The filing client renders a missing ticker in several spellings. Each time one
was added to a denylist, reality supplied another: "None" reached the studied
population first, then "N/A" reached the price API and was rejected with a 400.
Matching the shape of a real symbol closes the whole class.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from arbiter.events.insider import extract_insider_events
from arbiter.ingestion.edgar import FilingRecord

FIXTURE = Path(__file__).parents[1] / "fixtures" / "form4_open_market.json"


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


def _with_ticker(value: object) -> _Form4:
    payload = json.loads(FIXTURE.read_text())
    payload["rows"] = [{**payload["rows"][0], "Ticker": value}]
    return _Form4(payload)


@pytest.mark.parametrize(
    "placeholder",
    ["N/A", "n/a", "None", "none", "", "  ", "nan", "NULL", "--", "N.A.", "UNKNOWN"],
)
def test_placeholders_are_refused_however_they_are_spelled(placeholder: str):
    """The 400 that killed a day's labels came from 'N/A' reaching the price API."""
    with pytest.raises(ValueError, match="not a symbol"):
        extract_insider_events(_record(), _with_ticker(placeholder))


def test_a_missing_ticker_is_refused():
    with pytest.raises(ValueError, match="not a symbol"):
        extract_insider_events(_record(), _with_ticker(None))


@pytest.mark.parametrize("symbol", ["MO", "AAPL", "XLP", "BRK.B", "F", "GOOGL"])
def test_real_symbols_are_accepted(symbol: str):
    """Validation must not reject the tickers the pipeline exists to price."""
    events = extract_insider_events(_record(), _with_ticker(symbol))

    assert events
    assert events[0].ticker == symbol


def test_surrounding_whitespace_is_trimmed_rather_than_refused():
    events = extract_insider_events(_record(), _with_ticker("  MO  "))

    assert events[0].ticker == "MO"


def test_the_refusal_names_the_filing_and_the_value():
    """A quarantined row must say which filing and what it contained."""
    with pytest.raises(ValueError, match="0000764180-26-000102"):
        extract_insider_events(_record(), _with_ticker("N/A"))
