"""A red-flag event states the symbol its own 8-K's cover page carried.

Labelling used to resolve a ticker from SEC's current company map, so an issuer
that delisted after filing dropped out of it and its event was excluded. Auditor
changes, restatements and officer departures predict delisting, so the exclusion
was correlated with the outcome under study (issue #43).

The XBRL fixtures under `tests/fixtures/xbrl_cover/` are the cover-page rows the
SEC served, with the context each fact was tagged in, because pairing a symbol
with its title depends on that context:

- `0001628280-26-039277`  Cold storage REIT, one symbol (`COLD`).
- `0001193125-26-253436`  Single symbol of an issuer since delisted (`FBRX`).
- `0001423689-26-000124`  AGNC: common plus six listed depositary preferreds.
- `0001104659-26-105662`  Rigetti: common plus listed warrants, whose title
  itself contains the words "Common Stock".
"""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from arbiter.events.redflags import RedFlagEvent, choose_filing_ticker, with_filing_ticker
from arbiter.ingestion.edgar import CoverSecurity, FilingRecord, cover_securities
from arbiter.ingestion.pipeline import SystemicTickerReadError, ingest_day
from arbiter.ingestion.store import read_events

FIXTURES = Path(__file__).parent.parent / "fixtures" / "xbrl_cover"

SINGLE = "0001628280-26-039277"
DELISTED = "0001193125-26-253436"
PREFERREDS = "0001423689-26-000124"
WARRANTS = "0001104659-26-105662"


def _rows(accession: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURES / f"{accession}.json").read_text())["rows"]


class _Facts:
    """The slice of the XBRL facts API that cover-page reading touches."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self._concept = ""

    def query(self) -> "_Facts":
        return _Facts(self._rows)

    def by_concept(self, concept: str) -> "_Facts":
        selected = _Facts(self._rows)
        selected._concept = concept
        return selected

    def to_dataframe(self) -> pd.DataFrame:
        matching = [row for row in self._rows if row["concept"] == self._concept]
        return pd.DataFrame(matching, columns=["concept", "value", "context_ref"])


class _Xbrl:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.facts = _Facts(rows)


class _Filing:
    """A filing exposing `xbrl()` as the client does, or failing to."""

    def __init__(self, xbrl: object = None, raises: Exception | None = None) -> None:
        self._xbrl = xbrl
        self._raises = raises

    def xbrl(self) -> object:
        if self._raises is not None:
            raise self._raises
        return self._xbrl


def _filing(accession: str) -> _Filing:
    return _Filing(_Xbrl(_rows(accession)))


def _event() -> RedFlagEvent:
    return RedFlagEvent(
        accession_no="0001423689-26-000124",
        as_of=datetime(2026, 7, 20, 20, 0, tzinfo=UTC),
        cik=1423689,
        issuer="AGNC Investment Corp.",
        item_codes=("5.02",),
        item_text="Item 5.02",
        has_press_release=False,
    )


def _ticker(filing: _Filing) -> str | None:
    return choose_filing_ticker(cover_securities(filing))


def test_a_single_symbol_cover_page_yields_that_symbol():
    assert _ticker(_filing(SINGLE)) == "COLD"


def test_the_symbol_of_a_since_delisted_issuer_is_read_from_the_filing():
    """The case the issue was opened for: the current map no longer lists FBRX."""
    assert _ticker(_filing(DELISTED)) == "FBRX"


def test_common_stock_is_chosen_over_listed_preferreds():
    """AGNC states seven symbols; six are depositary preferreds such as AGNCN."""
    securities = cover_securities(_filing(PREFERREDS))

    assert len(securities) == 7
    assert choose_filing_ticker(securities) == "AGNC"


def test_the_common_stock_is_found_wherever_it_sits_on_the_cover_page():
    """Common is first in the recorded filing, which would pass a first-symbol rule.

    The recorded rows are reordered, not rewritten: the same facts, with the
    common stock's fact moved last, so the title pairing is the only thing that
    can still find it.
    """
    rows = _rows(PREFERREDS)
    common_context = next(r["context_ref"] for r in rows if r["value"] == "AGNC")
    moved = [r for r in rows if r["context_ref"] != common_context] + [
        r for r in rows if r["context_ref"] == common_context
    ]
    assert moved[0]["value"] != "AGNC", (
        "premise: the reordered filing no longer leads with AGNC"
    )
    assert next(r["value"] for r in moved if r["concept"] == "dei:TradingSymbol") == "AGNCN"

    assert _ticker(_filing_from(moved)) == "AGNC"


def test_a_warrant_whose_title_mentions_common_stock_is_not_the_common_stock():
    """Rigetti's warrant is titled 'Warrants ... exercisable for one share of Common Stock'.

    Moved first so that a rule matching the word "common" anywhere in a title
    would take the warrant symbol.
    """
    rows = _rows(WARRANTS)
    warrant_context = next(r["context_ref"] for r in rows if r["value"] == "RGTIW")
    warrant_title = next(
        r["value"]
        for r in rows
        if r["concept"] == "dei:Security12bTitle" and r["context_ref"] == warrant_context
    )
    assert "common" in warrant_title.lower(), "premise: the warrant's title names common stock"
    moved = [r for r in rows if r["context_ref"] == warrant_context] + [
        r for r in rows if r["context_ref"] != warrant_context
    ]
    assert next(r["value"] for r in moved if r["concept"] == "dei:TradingSymbol") == "RGTIW"

    assert _ticker(_filing_from(moved)) == "RGTI"


def _filing_from(rows: list[dict[str, Any]]) -> _Filing:
    return _Filing(_Xbrl(rows))


def test_two_untitled_acceptable_symbols_are_ambiguous():
    securities = [
        CoverSecurity("ABC.", None),
        CoverSecurity("XYZ", None),
        CoverSecurity("QRS", None),
    ]

    assert choose_filing_ticker(securities) is None


def test_a_title_pairs_by_context_not_by_position():
    """The first fact has no title; the second's title must not slide onto it."""
    rows = [
        {"concept": "dei:TradingSymbol", "value": "AAA", "context_ref": "c-1"},
        {"concept": "dei:TradingSymbol", "value": "BBB", "context_ref": "c-2"},
        {"concept": "dei:Security12bTitle", "value": "Common Stock", "context_ref": "c-2"},
    ]

    securities = cover_securities(_filing_from(rows))

    assert [(s.symbol, s.title) for s in securities] == [("AAA", None), ("BBB", "Common Stock")]
    assert choose_filing_ticker(securities) == "BBB"


def test_a_cover_page_with_only_unusable_symbols_yields_none():
    securities = [CoverSecurity("N/A", "Common Stock"), CoverSecurity("none", None)]

    assert choose_filing_ticker(securities) is None


def test_a_filing_without_xbrl_yields_no_ticker_and_no_error():
    assert cover_securities(_Filing(None)) == []
    assert with_filing_ticker(_event(), cover_securities(_Filing(None))).ticker is None


def test_a_filing_with_xbrl_but_no_symbol_yields_no_ticker():
    rows = [r for r in _rows(SINGLE) if r["concept"] != "dei:TradingSymbol"]
    assert rows, "premise: the title fact remains, so only the symbol is missing"

    assert cover_securities(_filing_from(rows)) == []


def test_the_event_records_where_its_ticker_came_from():
    event = with_filing_ticker(_event(), cover_securities(_filing(PREFERREDS)))

    assert (event.ticker, event.ticker_source) == ("AGNC", "filing")


def test_an_event_without_a_filing_ticker_has_neither_ticker_nor_source():
    event = with_filing_ticker(_event(), [])

    assert (event.ticker, event.ticker_source) == (None, None)


# --- the day-level behaviour: one filing's XBRL must not cost the others ------


class _Report:
    items = ("Item 5.02",)
    has_press_release = False

    def text(self) -> str:
        return "Item 5.02 text"


class _EightK:
    def __init__(self, xbrl_filing: _Filing) -> None:
        self._xbrl = xbrl_filing

    def obj(self) -> _Report:
        return _Report()

    def xbrl(self) -> object:
        return self._xbrl.xbrl()


def _record(accession: str, cik: int) -> FilingRecord:
    return FilingRecord(
        accession_no=accession,
        form="8-K",
        cik=cik,
        company=f"Issuer {cik}",
        as_of=datetime(2026, 9, 11, 20, 0, tzinfo=UTC),
        filing_date=date(2026, 9, 11),
    )


def _run_day(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, filings: list[Any]) -> None:
    def fake(form: str, day: date, on_untimestamped: Any) -> list[Any]:
        return filings if form == "8-K" else []

    monkeypatch.setattr("arbiter.ingestion.pipeline.filings_with_objects", fake)
    ingest_day(date(2026, 9, 11), tmp_path, min_value_usd=Decimal(1))


def test_an_xbrl_read_that_raises_costs_that_event_only_its_ticker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Three red-flag filings; the middle one's XBRL read blows up.

    All three events must still be stored, the neighbours with their tickers and
    the failed one without, and the failure must be on record by accession.
    """
    boom = RuntimeError("XBRL instance could not be parsed")
    filings = [
        (_record("0000000001-26-000001", 1), _EightK(_filing(SINGLE))),
        (_record("0000000002-26-000002", 2), _EightK(_Filing(raises=boom))),
        (_record("0000000003-26-000003", 3), _EightK(_filing(DELISTED))),
    ]

    _run_day(monkeypatch, tmp_path, filings)

    rows = {r["accession_no"]: r for r in read_events(tmp_path, "redflag", date(2026, 9, 11))}
    assert set(rows) == {"0000000001-26-000001", "0000000002-26-000002", "0000000003-26-000003"}
    assert rows["0000000001-26-000001"]["ticker"] == "COLD"
    assert rows["0000000003-26-000003"]["ticker"] == "FBRX"
    assert rows["0000000002-26-000002"]["ticker"] is None
    assert rows["0000000002-26-000002"]["ticker_source"] is None
    assert rows["0000000001-26-000001"]["ticker_source"] == "filing"

    failures = json.loads(
        (tmp_path / "rejected" / "redflag-ticker" / "2026-09-11.json").read_text()
    )
    assert [f["accession_no"] for f in failures] == ["0000000002-26-000002"]
    assert "XBRL instance could not be parsed" in failures[0]["error"]


def test_a_filing_without_xbrl_is_not_a_recorded_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """No XBRL is an ordinary property of some 8-Ks, not a defect to report."""
    filings = [(_record("0000000001-26-000001", 1), _EightK(_Filing(None)))]

    _run_day(monkeypatch, tmp_path, filings)

    (row,) = read_events(tmp_path, "redflag", date(2026, 9, 11))
    assert row["ticker"] is None
    failures = json.loads(
        (tmp_path / "rejected" / "redflag-ticker" / "2026-09-11.json").read_text()
    )
    assert failures == []


def test_xbrl_is_not_read_for_an_8k_that_is_not_an_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Most 8-Ks carry no trigger item; each XBRL fetch is a request."""

    class _Untriggered(_Report):
        items = ("Item 2.02", "Item 9.01")

    reads: list[str] = []

    class _Quiet(_EightK):
        def obj(self) -> _Report:
            return _Untriggered()

        def xbrl(self) -> object:
            # Recorded rather than raised: the pipeline isolates XBRL failures,
            # so an exception here would be caught and the test would pass.
            reads.append("read")
            return None

    filings = [(_record("0000000001-26-000001", 1), _Quiet(_Filing(None)))]

    _run_day(monkeypatch, tmp_path, filings)

    assert reads == [], "no XBRL read for a filing that carries no trigger item"
    assert read_events(tmp_path, "redflag", date(2026, 9, 11)) == []


def test_untitled_symbols_among_several_yield_none():
    """AGNC's recorded rows with every title removed: seven symbols, no way to tell
    the common stock from AGNCN. A guess could be a preferred."""
    rows = [r for r in _rows(PREFERREDS) if r["concept"] == "dei:TradingSymbol"]
    assert len(rows) == 7, "premise: several tradeable symbols remain"
    assert not any(r["concept"] == "dei:Security12bTitle" for r in rows), "premise: no titles"

    assert _ticker(_filing_from(rows)) is None


def test_a_single_untitled_symbol_is_taken():
    rows = [r for r in _rows(SINGLE) if r["concept"] == "dei:TradingSymbol"]
    assert len(rows) == 1, "premise: exactly one symbol, and no title"

    assert _ticker(_filing_from(rows)) == "COLD"


def test_only_tradeable_symbols_count_toward_the_single_symbol_rule():
    securities = [CoverSecurity("N/A", None), CoverSecurity("XYZ", None)]

    assert choose_filing_ticker(securities) == "XYZ"


# --- the day-level guard on failed cover-page reads ---------------------------


def _day_of(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, total: int, failing: int) -> None:
    boom = RuntimeError("XBRL instance could not be parsed")
    filings = [
        (
            _record(f"0000000001-26-{i:06d}", i),
            _EightK(_Filing(raises=boom) if i < failing else _filing(SINGLE)),
        )
        for i in range(total)
    ]
    _run_day(monkeypatch, tmp_path, filings)


def _stored(tmp_path: Path) -> bool:
    return (tmp_path / "redflag" / "2026-09-11.parquet").exists()


def test_exactly_half_failing_is_recorded_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Six reads, three failed: not more than half."""
    _day_of(monkeypatch, tmp_path, total=6, failing=3)

    assert _stored(tmp_path)


def test_more_than_half_failing_fails_the_day(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Five reads, three failed: a day stored like this would fall back to the map."""
    with pytest.raises(SystemicTickerReadError, match="3 of 5"):
        _day_of(monkeypatch, tmp_path, total=5, failing=3)

    assert not _stored(tmp_path), "a failed day must stay unrecorded so a rerun retries it"
    failures = json.loads(
        (tmp_path / "rejected" / "redflag-ticker" / "2026-09-11.json").read_text()
    )
    assert len(failures) == 3, "the causes are still on record"


def test_below_the_minimum_attempts_even_total_failure_is_only_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Four reads, all four failed: too few to say the client is broken."""
    _day_of(monkeypatch, tmp_path, total=4, failing=4)

    assert _stored(tmp_path)
