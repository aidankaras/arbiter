import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from arbiter.events.redflags import (
    TRIGGER_ITEMS,
    RedFlagEvent,
    extract_redflag_event,
    triggered_items,
)
from arbiter.ingestion.edgar import FilingRecord

FIXTURE = Path(__file__).parents[1] / "fixtures" / "current_report_redflag.json"


class _CurrentReport:
    """Stands in for the parsed 8-K, exposing only what the extractor reads."""

    def __init__(self, payload: dict) -> None:
        self.items = payload["items"]
        self.has_press_release = payload["has_press_release"]
        self._text = payload["text"]

    def text(self) -> str:
        return self._text


def _record() -> FilingRecord:
    return FilingRecord(
        accession_no="0001423689-26-000135",
        form="8-K",
        cik=1423689,
        company="AGNC Investment Corp.",
        as_of=datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        filing_date=date(2026, 9, 11),
    )


def test_the_trigger_set_matches_the_documented_domain():
    assert frozenset({"4.01", "4.02", "5.02"}) == TRIGGER_ITEMS


def test_item_codes_are_normalized_from_their_labels():
    """Grouping must not depend on how the label happens to be formatted."""
    assert triggered_items(["Item 5.02", "Item 9.01"]) == ("5.02",)


def test_multiple_triggers_are_all_reported_in_order():
    assert triggered_items(["Item 5.02", "Item 4.01"]) == ("4.01", "5.02")


def test_untriggered_reports_yield_no_codes():
    assert triggered_items(["Item 2.02", "Item 7.01", "Item 9.01"]) == ()


@pytest.mark.parametrize(
    "label",
    ["Item 5.02(b)", "Item 5.02(c)", "ITEM 5.02", "item 5.02", "Item  5.02", " Item 5.02(e)"],
)
def test_sub_lettered_and_differently_cased_labels_still_match(label: str):
    """Item 4.02 is nearly always filed as 4.02(a); exact matching lost the population."""
    assert triggered_items([label]) == ("5.02",)


def test_a_restatement_filed_with_its_usual_letter_suffix_is_caught():
    assert triggered_items(["Item 4.02(a)", "Item 9.01"]) == ("4.02",)


def test_a_similar_looking_item_is_not_matched():
    """5.02 must not swallow 5.021 or 15.02 if either ever appears."""
    assert triggered_items(["Item 15.02"]) == ()


def test_the_captured_fixture_carries_a_trigger_item():
    """An upstream change to item labels must fail here, not silently drop events."""
    payload = json.loads(FIXTURE.read_text())
    assert triggered_items(payload["items"])


def test_extracting_an_event_from_a_real_filing():
    payload = json.loads(FIXTURE.read_text())
    event = extract_redflag_event(_record(), _CurrentReport(payload))
    assert isinstance(event, RedFlagEvent)
    assert event.item_codes == ("5.02",)
    assert event.cik == 1423689
    assert event.issuer == "AGNC Investment Corp."
    assert event.as_of.tzinfo is not None
    assert event.item_text


def test_a_report_without_a_trigger_item_produces_no_event():
    payload = json.loads(FIXTURE.read_text()) | {"items": ["Item 2.02", "Item 9.01"]}
    assert extract_redflag_event(_record(), _CurrentReport(payload)) is None


def test_events_are_immutable():
    payload = json.loads(FIXTURE.read_text())
    event = extract_redflag_event(_record(), _CurrentReport(payload))
    assert event is not None
    with pytest.raises(ValueError, match="frozen"):
        event.item_text = "rewritten"
