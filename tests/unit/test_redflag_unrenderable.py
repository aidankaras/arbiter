"""A filing whose prose cannot be rendered is still an event.

A small number of current reports expose no primary document the client can
render, and asking for their text raises from inside the library. The item codes
come from the filing index rather than the body, so such a report is a real
event with a real timestamp and no readable text.
"""

from datetime import UTC, date, datetime

import pytest

from arbiter.events.redflags import extract_redflag_event
from arbiter.ingestion.edgar import FilingRecord


class _Report:
    """Stands in for a parsed current report with a controllable text path."""

    def __init__(self, text: object, items: list[str] | None = None) -> None:
        self.items = items if items is not None else ["Item 5.02", "Item 9.01"]
        self.has_press_release = False
        self._text = text

    def text(self) -> str:
        if isinstance(self._text, Exception):
            raise self._text
        return str(self._text)


def _record() -> FilingRecord:
    return FilingRecord(
        accession_no="0001423689-26-000135",
        form="8-K",
        cik=1423689,
        company="AGNC Investment Corp.",
        as_of=datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        filing_date=date(2026, 9, 11),
    )


def test_a_report_whose_text_cannot_be_rendered_still_yields_an_event():
    """The client raises from its own rendering path; the event survives it."""
    report = _Report(AttributeError("'NoneType' object has no attribute 'download'"))

    event = extract_redflag_event(_record(), report)

    assert event is not None
    assert event.item_codes == ("5.02",)
    assert event.item_text == ""


@pytest.mark.parametrize("failure", [ValueError("bad"), TypeError("bad")])
def test_other_rendering_failures_are_tolerated_the_same_way(failure: Exception):
    event = extract_redflag_event(_record(), _Report(failure))

    assert event is not None
    assert event.item_text == ""


def test_a_renderable_report_keeps_its_prose():
    """Regression guard: tolerating failure must not discard text that exists."""
    event = extract_redflag_event(_record(), _Report("Item 5.02 Departure of Directors"))

    assert event is not None
    assert "Departure of Directors" in event.item_text


def test_an_unrenderable_report_without_a_trigger_item_still_yields_nothing():
    """Text tolerance must not turn an untriggered report into an event."""
    report = _Report(AttributeError("unrenderable"), items=["Item 2.02", "Item 9.01"])

    assert extract_redflag_event(_record(), report) is None
