import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from arbiter.events.insider import extract_insider_events, is_candidate
from arbiter.ingestion.edgar import FilingRecord

FIXTURE = Path(__file__).parents[1] / "fixtures" / "form4_unpriced.json"
MIN_VALUE = Decimal("50000")


class _Form4:
    """Stands in for the parsed filing, exposing only what the extractor reads."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.aff10b5_one = payload["aff10b5_one"]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self._payload["rows"])


def _record() -> FilingRecord:
    return FilingRecord(
        accession_no="0001628280-26-061534",
        form="4",
        cik=1387467,
        company="Alpha & Omega Semiconductor Ltd",
        as_of=datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        filing_date=date(2026, 9, 11),
    )


def _events():
    return extract_insider_events(_record(), _Form4(json.loads(FIXTURE.read_text())))


def test_an_unpriced_transaction_does_not_crash_extraction():
    """A gift reports no consideration; parsing it must not raise."""
    assert _events()


def test_an_absent_value_is_recorded_as_absent_not_as_zero():
    """Zero dollars would read as a free acquisition and distort any aggregate."""
    gifts = [event for event in _events() if event.transaction_code == "G"]
    assert gifts
    assert all(event.value_usd is None for event in gifts)


def test_a_reported_zero_price_is_preserved_as_zero():
    """The filing states a zero price, which is data rather than a missing field."""
    gifts = [event for event in _events() if event.transaction_code == "G"]
    assert all(event.price == Decimal("0") for event in gifts)


def test_share_counts_are_still_required_and_parsed():
    gifts = [event for event in _events() if event.transaction_code == "G"]
    assert all(event.shares > 0 for event in gifts)


def test_unpriced_transactions_are_never_candidates():
    """Excluded twice over: a gift is not an open-market trade and has no value."""
    assert not any(is_candidate(event, MIN_VALUE) for event in _events())
