import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.events.insider import InsiderEvent, extract_insider_events, is_candidate
from arbiter.ingestion.edgar import FilingRecord
from tests.unit.form4_stub import Form4Stub

FIXTURES = Path(__file__).parents[1] / "fixtures"
OPEN_MARKET = FIXTURES / "form4_open_market.json"
PLAN_TRADE = FIXTURES / "form4_plan_trade.json"

MIN_VALUE = Decimal("50000")


def _record() -> FilingRecord:
    return FilingRecord(
        accession_no="0000764180-26-000102",
        form="4",
        cik=764180,
        company="ALTRIA GROUP, INC.",
        as_of=datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        filing_date=date(2026, 9, 11),
    )


def _event(**overrides) -> InsiderEvent:
    base = {
        "accession_no": "0000764180-26-000102",
        "as_of": datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        "cik": 764180,
        "ticker": "MO",
        "issuer": "ALTRIA GROUP, INC.",
        "insider_name": "Kathryn B. Mcquade",
        "position": "Director",
        "transaction_code": "P",
        "shares": Decimal("1500"),
        "price": Decimal("67.62"),
        "value_usd": Decimal("101430.0"),
        "remaining_shares": Decimal("114929"),
        "is_10b5_1": False,
    }
    base.update(overrides)
    return InsiderEvent(**base)


def test_the_captured_fixture_has_the_fields_the_extractor_reads():
    """An upstream format change must fail here, not produce empty events."""
    rows = json.loads(OPEN_MARKET.read_text())["rows"]
    assert rows
    assert {"Code", "Shares", "Price", "Value", "Ticker", "Issuer", "Insider"} <= set(rows[0])


def test_extracting_an_open_market_purchase_from_a_real_filing():
    payload = json.loads(OPEN_MARKET.read_text())
    events = extract_insider_events(_record(), Form4Stub(payload))
    assert events
    purchase = events[0]
    assert purchase.transaction_code == "P"
    assert purchase.ticker == "MO"
    assert purchase.value_usd == Decimal("101430.0")
    assert purchase.is_10b5_1 is False
    assert purchase.as_of.tzinfo is not None


def test_events_inherit_the_filings_acceptance_instant():
    """The transaction date is not when the market learned of it."""
    payload = json.loads(OPEN_MARKET.read_text())
    events = extract_insider_events(_record(), Form4Stub(payload))
    assert all(event.as_of == _record().as_of for event in events)


def test_a_plan_filing_marks_every_event_as_scheduled():
    payload = json.loads(PLAN_TRADE.read_text())
    events = extract_insider_events(_record(), Form4Stub(payload))
    assert events
    assert all(event.is_10b5_1 for event in events)
    assert not any(is_candidate(event, MIN_VALUE) for event in events)


def test_open_market_trades_above_the_size_floor_are_candidates():
    assert is_candidate(_event(), MIN_VALUE)


def test_small_trades_are_not_candidates():
    assert not is_candidate(_event(value_usd=Decimal("900")), MIN_VALUE)


def test_scheduled_plan_trades_are_excluded():
    """A 10b5-1 trade was scheduled months earlier and carries no current view."""
    assert not is_candidate(_event(is_10b5_1=True), MIN_VALUE)


@pytest.mark.parametrize("code", ["M", "G", "A", "F"])
def test_non_open_market_codes_are_excluded(code: str):
    """Option exercises, gifts, grants, and tax withholding reflect compensation."""
    assert not is_candidate(_event(transaction_code=code), MIN_VALUE)


def test_events_are_immutable():
    event = _event()
    with pytest.raises(ValueError, match="frozen"):
        event.shares = Decimal("1")
