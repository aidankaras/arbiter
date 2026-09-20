import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from arbiter.events.insider import extract_insider_events
from arbiter.ingestion.edgar import FilingRecord
from tests.unit.form4_stub import Form4Stub

FIXTURES = Path(__file__).parents[1] / "fixtures"
NO_TRANSACTIONS = FIXTURES / "form4_no_transactions.json"
OPEN_MARKET = FIXTURES / "form4_open_market.json"


def _record() -> FilingRecord:
    return FilingRecord(
        accession_no="0001922054-26-000007",
        form="4/A",
        cik=1922054,
        company="NUSCALE POWER Corp",
        as_of=datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        filing_date=date(2026, 9, 11),
    )


def test_a_filing_reporting_no_transactions_yields_no_events():
    """An amendment may carry only identity and a remark. That is not an event."""
    payload = json.loads(NO_TRANSACTIONS.read_text())

    assert extract_insider_events(_record(), Form4Stub(payload)) == []


def test_the_fixture_really_lacks_every_transaction_column():
    """Guards the premise: if this filing gains columns, the test above is vacuous."""
    row = json.loads(NO_TRANSACTIONS.read_text())["rows"][0]

    assert not {"Code", "Shares", "Price", "Value", "Remaining Shares"} & set(row)


def test_a_partially_present_schema_still_raises():
    """Some transaction columns but not others means the format changed."""
    payload = json.loads(OPEN_MARKET.read_text())
    payload["rows"] = [
        {key: value for key, value in payload["rows"][0].items() if key != "Shares"}
    ]

    with pytest.raises(KeyError, match="Shares"):
        extract_insider_events(_record(), Form4Stub(payload))


def test_ordinary_filings_still_produce_events():
    """Regression guard: skipping empty schemas must not skip real transactions."""
    payload = json.loads(OPEN_MARKET.read_text())

    assert extract_insider_events(_record(), Form4Stub(payload))
