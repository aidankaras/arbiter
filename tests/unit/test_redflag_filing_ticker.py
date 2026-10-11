"""A red-flag event states the symbol its own 8-K's cover page carried.

Labelling used to resolve a ticker from SEC's current company map, so an issuer
that delisted after filing dropped out of it and its event was excluded. Auditor
changes, restatements and officer departures predict delisting, so the exclusion
was correlated with the outcome under study (issue #43).

The XBRL fixtures under `tests/fixtures/xbrl_cover/` are the cover-page rows the
SEC served, with the context each fact was tagged in, because pairing a symbol
with its title and its registrant depends on the context. Each test asserts the
shape its fixture exists for before asserting the outcome:

- `0001628280-26-039277`  Cold storage REIT, one symbol (`COLD`).
- `0001193125-26-253436`  Single symbol of an issuer since delisted (`FBRX`).
- `0001423689-26-000124`  AGNC: common plus six listed depositary preferreds.
- `0001104659-26-105662`  Rigetti: common plus listed warrants, whose title
  itself contains the words "Common Stock".
- `0001213900-26-103656`  A SPAC (CRAC): units, Class A ordinary shares,
  redeemable warrants and rights, every title naming the ordinary shares.
- `0000092122-26-000056`  Southern Company's combined filing: the parent's
  common stock and corporate units, subsidiaries' notes, and Alabama Power
  registered with no security of its own.
- `0000065984-26-000279`  Entergy's combined filing: the parent's common stock,
  and each subsidiary's mortgage bonds tagged to that subsidiary.

Titles with no recorded filing (a right attached to common stock, an isolated
exclusion term) are constructed, and each such test says so.
"""

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
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
SPAC = "0001213900-26-103656"
SOUTHERN = "0000092122-26-000056"
ENTERGY = "0000065984-26-000279"

# The CIKs the filings' own primary registrants carry.
COLD_CIK = 1455863
FBRX_CIK = 1419041
AGNC_CIK = 1423689
RIGETTI_CIK = 1838359
CRAC_CIK = 2070887
SOUTHERN_CIK = 92122
ENTERGY_CIK = 65984
# Index entries for the same combined filings under co-registrants.
ALABAMA_POWER_CIK = 3153
ENTERGY_ARKANSAS_CIK = 7323

TICKER = "dei:TradingSymbol"
TITLE = "dei:Security12bTitle"


def _rows(accession: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURES / f"{accession}.json").read_text())["rows"]


class _Facts:
    """The slice of the XBRL facts API that cover-page reading touches."""

    def __init__(self, rows: list[dict[str, Any]], concept: str = "") -> None:
        self._rows = rows
        self._concept = concept

    def query(self) -> "_Facts":
        return _Facts(self._rows)

    def by_concept(self, concept: str) -> "_Facts":
        return _Facts(self._rows, concept)

    def to_dataframe(self) -> pd.DataFrame:
        matching = [row for row in self._rows if row["concept"] == self._concept]
        return pd.DataFrame(
            matching, columns=["concept", "value", "context_ref", "entity_identifier"]
        )


class _Xbrl:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.facts = _Facts(rows)
        # Facts sharing a context share its dimensions, so a fact recorded
        # without them (a title) must not blank the ones a sibling carries.
        dimensions: dict[str, dict[str, str]] = {}
        for row in rows:
            dimensions.setdefault(row["context_ref"], {}).update(row.get("dimensions", {}))
        self.contexts = {
            ref: SimpleNamespace(dimensions=dims) for ref, dims in dimensions.items()
        }


class _Filing:
    """A filing exposing `xbrl()` as the client does, or failing to."""

    def __init__(self, xbrl: object = None, raises: Exception | None = None) -> None:
        self._xbrl = xbrl
        self._raises = raises

    def xbrl(self) -> object:
        if self._raises is not None:
            raise self._raises
        return self._xbrl


def _filing_from(rows: list[dict[str, Any]]) -> _Filing:
    return _Filing(_Xbrl(rows))


def _filing(accession: str) -> _Filing:
    return _filing_from(_rows(accession))


def _fact(
    concept: str, value: str, context: str, entity: str = "0000000001", **dims: str
) -> dict:
    """A constructed cover-page fact, for shapes no recorded filing exhibits."""
    return {
        "concept": concept,
        "value": value,
        "context_ref": context,
        "entity_identifier": entity,
        "dimensions": dims,
    }


def _event(cik: int = AGNC_CIK) -> RedFlagEvent:
    return RedFlagEvent(
        accession_no="0001423689-26-000124",
        as_of=datetime(2026, 7, 20, 20, 0, tzinfo=UTC),
        cik=cik,
        issuer="AGNC Investment Corp.",
        item_codes=("5.02",),
        item_text="Item 5.02",
        has_press_release=False,
    )


def _ticker(rows: list[dict[str, Any]], cik: int) -> str | None:
    return choose_filing_ticker(cover_securities(_filing_from(rows)), cik)


def _moved_last(rows: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    """The same facts with one symbol's context moved to the end of the filing."""
    contexts = {
        r["context_ref"] for r in rows if r["concept"] == TICKER and r["value"] == symbol
    }
    return [r for r in rows if r["context_ref"] not in contexts] + [
        r for r in rows if r["context_ref"] in contexts
    ]


def _moved_first(rows: list[dict[str, Any]], symbol: str) -> list[dict[str, Any]]:
    context = next(
        r["context_ref"] for r in rows if r["concept"] == TICKER and r["value"] == symbol
    )
    return [r for r in rows if r["context_ref"] == context] + [
        r for r in rows if r["context_ref"] != context
    ]


def _symbols(rows: list[dict[str, Any]]) -> list[str]:
    return [r["value"] for r in rows if r["concept"] == TICKER]


def _title_of(rows: list[dict[str, Any]], symbol: str) -> str:
    context = next(
        r["context_ref"] for r in rows if r["concept"] == TICKER and r["value"] == symbol
    )
    return next(
        r["value"] for r in rows if r["concept"] == TITLE and r["context_ref"] == context
    )


# --- single registrant, one or several securities ----------------------------


def test_a_single_symbol_cover_page_yields_that_symbol():
    assert _ticker(_rows(SINGLE), COLD_CIK) == "COLD"


def test_the_symbol_of_a_since_delisted_issuer_is_read_from_the_filing():
    """The case the issue was opened for: the current map no longer lists FBRX."""
    assert _ticker(_rows(DELISTED), FBRX_CIK) == "FBRX"


def test_common_stock_is_chosen_over_listed_preferreds():
    """AGNC states seven symbols; six are depositary preferreds such as AGNCN."""
    rows = _rows(PREFERREDS)
    assert len(_symbols(rows)) == 7, "premise: one common and six preferred symbols"

    assert _ticker(rows, AGNC_CIK) == "AGNC"


def test_the_common_stock_is_found_wherever_it_sits_on_the_cover_page():
    """Common is first in the recorded filing, which would pass a first-symbol rule.

    The recorded rows are reordered, not rewritten: the same facts, with the
    common stock's fact moved last, so the title pairing is the only thing that
    can still find it.
    """
    moved = _moved_last(_rows(PREFERREDS), "AGNC")
    assert _symbols(moved)[0] == "AGNCN", "premise: AGNC no longer leads the filing"

    assert _ticker(moved, AGNC_CIK) == "AGNC"


def test_a_warrant_whose_title_mentions_common_stock_is_not_the_common_stock():
    """Rigetti's warrant is titled 'Warrants ... exercisable for one share of Common Stock'."""
    rows = _rows(WARRANTS)
    assert "common" in _title_of(rows, "RGTIW").lower(), (
        "premise: the warrant names common stock"
    )
    moved = _moved_first(rows, "RGTIW")
    assert _symbols(moved)[0] == "RGTIW", "premise: the warrant leads the filing"

    assert _ticker(moved, RIGETTI_CIK) == "RGTI"


def test_a_spacs_class_a_ordinary_shares_are_chosen_over_units_warrants_and_rights():
    """CRAC lists units, Class A ordinary shares, warrants and rights.

    Every one of the four titles names the ordinary shares ("one Class A
    ordinary share" inside the unit, "exercisable for one Class A ordinary
    share" in the warrant), and none says 'common'. The unit leads the filing.
    """
    rows = _rows(SPAC)
    assert _symbols(rows) == ["CRACU", "CRAC", "CRACW", "CRACR"], "premise: the four securities"
    assert all("ordinary share" in t.lower() for t in _titles(rows)), "premise: all name shares"
    assert not any("common" in t.lower() for t in _titles(rows)), "premise: none says common"

    assert _ticker(rows, CRAC_CIK) == "CRAC"
    assert _ticker(_moved_last(rows, "CRAC"), CRAC_CIK) == "CRAC"


def _titles(rows: list[dict[str, Any]]) -> list[str]:
    return [r["value"] for r in rows if r["concept"] == TITLE]


def test_a_title_that_names_no_equity_is_not_common_equity():
    """Constructed: a share class titled without 'common', 'ordinary' or 'capital stock'.

    It is on no exclusion list either, so only the requirement that a title name
    common equity keeps its symbol from being taken ahead of the real common stock.
    """
    securities = [
        CoverSecurity("AAA", "Class A Shares, par value $0.01", 1),
        CoverSecurity("BBB", "Common Stock", 1),
    ]

    assert choose_filing_ticker(securities, 1) == "BBB"


# --- combined filings: a symbol belongs to one registrant ---------------------


def test_a_subsidiary_does_not_inherit_its_parents_ticker():
    """Southern Company's combined filing, indexed under Alabama Power too.

    The filing's symbols sit on a share-class axis only, so they all belong to the
    parent. Alabama Power's event previously came out as `SO`.
    """
    rows = _rows(SOUTHERN)
    legal_entities = {
        r["dimensions"]["dei:LegalEntityAxis"]
        for r in rows
        if r["concept"] == "dei:EntityCentralIndexKey" and r["dimensions"]
    }
    assert len(legal_entities) == 5, "premise: five co-registrants besides the parent"
    assert str(ALABAMA_POWER_CIK) in {
        str(int(r["value"])) for r in rows if r["concept"] == "dei:EntityCentralIndexKey"
    }, "premise: Alabama Power is a registrant on this filing"
    context = next(
        r["context_ref"] for r in rows if r["concept"] == TICKER and r["value"] == "SO"
    )
    assert "dei:LegalEntityAxis" not in next(
        r["dimensions"] for r in rows if r["context_ref"] == context
    ), "premise: the parent's common stock is tagged to no legal entity"
    assert any(
        "dei:LegalEntityAxis" in r["dimensions"] for r in rows if r["concept"] == TICKER
    ), "premise: some co-registrant securities are tagged to theirs"

    assert _ticker(rows, SOUTHERN_CIK) == "SO"
    assert _ticker(rows, ALABAMA_POWER_CIK) is None


def test_the_parent_keeps_its_common_stock_not_its_corporate_units():
    """Southern also lists `SOMN`, '2025 Series A Corporate Units'."""
    rows = _rows(SOUTHERN)
    assert "Corporate Units" in _title_of(rows, "SOMN"), "premise: a units symbol is listed"
    moved = _moved_first(rows, "SOMN")
    assert _symbols(moved)[0] == "SOMN", "premise: the units lead the filing"

    assert _ticker(moved, SOUTHERN_CIK) == "SO"


def test_entergy_arkansas_does_not_inherit_entergy_corporations_ticker():
    rows = _rows(ENTERGY)
    assert _symbols(rows).count("ETR") == 2, "premise: the parent's common is listed"

    arkansas = [
        s for s in cover_securities(_filing_from(rows)) if s.entity_cik == ENTERGY_ARKANSAS_CIK
    ]
    assert [s.symbol for s in arkansas] == ["EAI"], "premise: its only security is a bond"
    assert "Mortgage Bonds" in (arkansas[0].title or ""), "premise: titled as a bond"

    assert _ticker(rows, ENTERGY_CIK) == "ETR"
    assert _ticker(rows, ENTERGY_ARKANSAS_CIK) is None


def test_a_symbol_tagged_to_a_legal_entity_belongs_to_that_entity():
    """Constructed: no recorded filing tags a symbol under a legal-entity member."""
    rows = [
        _fact("dei:EntityCentralIndexKey", "0000000001", "c-1"),
        _fact(
            "dei:EntityCentralIndexKey",
            "0000000002",
            "c-2",
            **{"dei:LegalEntityAxis": "x:SubMember"},
        ),
        _fact(TICKER, "PAR", "c-3", **{"dei:LegalEntityAxis": "x:ParentMember"}),
        _fact(TITLE, "Common Stock", "c-3"),
        _fact(TICKER, "SUB", "c-4", **{"dei:LegalEntityAxis": "x:SubMember"}),
        _fact(TITLE, "Common Stock", "c-4"),
    ]
    securities = cover_securities(_filing_from(rows))
    assert [(s.symbol, s.entity_cik) for s in securities] == [("PAR", None), ("SUB", 2)], (
        "premise: the unmapped member resolves to no CIK"
    )

    assert choose_filing_ticker(securities, 2) == "SUB"
    assert choose_filing_ticker(securities, 1) is None


# --- the common-equity predicate, one term at a time (constructed titles) ----


@pytest.mark.parametrize(
    ("term", "title"),
    [
        ("warrant", "Warrants, each exercisable for one share of Class A Common Stock"),
        ("right", "Rights, each entitling the holder to one share of Common Stock"),
        ("preferred", "Series A Preferred Stock, convertible into Common Stock"),
        ("depositary", "American Depositary Shares, each representing one Ordinary Share"),
        ("note", "6.00% Senior Notes due 2030, convertible into Common Stock"),
        ("debenture", "Convertible Debentures due 2030, exchangeable for Common Stock"),
        ("unit", "Units, each consisting of one share of Common Stock"),
        ("bond", "Mortgage Bonds due 2030, convertible into Common Stock"),
    ],
)
def test_each_exclusion_term_disqualifies_a_title_that_names_common_equity(
    term: str, title: str
):
    """Constructed from the shapes real instruments take.

    Each title names common equity *and* contains exactly one exclusion term, so
    dropping that term from the list is the only way the instrument gets chosen
    over the plain common stock that follows it.
    """
    lowered = title.lower()
    assert term in lowered, "premise: the title carries the term under test"
    assert any(w in lowered for w in ("common", "ordinary")), "premise: it names equity"
    securities = [
        CoverSecurity("AAA", title, 1),
        CoverSecurity("BBB", "Common Stock, par value $0.01", 1),
    ]

    assert choose_filing_ticker(securities, 1) == "BBB"


@pytest.mark.parametrize("title", ["Common Units representing limited partner interests"])
def test_common_units_are_common_equity(title: str):
    """An MLP's primary security is titled 'Common Units', not 'Common Stock'."""
    securities = [
        CoverSecurity("AAA", "Mortgage Bonds due 2030", 1),
        CoverSecurity("BBB", title, 1),
    ]

    assert choose_filing_ticker(securities, 1) == "BBB"


def test_capital_stock_is_common_equity():
    """Constructed: 'Class C Capital Stock' is how some issuers title their only equity."""
    securities = [
        CoverSecurity("AAA", "Mortgage Bonds due 2030", 1),
        CoverSecurity("BBB", "Class C Capital Stock", 1),
    ]

    assert choose_filing_ticker(securities, 1) == "BBB"


def test_a_right_attached_to_the_common_stock_does_not_disqualify_it():
    """Constructed: 'Common Stock and associated Preferred Stock Purchase Rights'.

    No recorded 8-K titles its rights this way; the usual shape is a separate
    security. The title names a right and a preferred, both on the exclusion
    list, and is still the common stock.
    """
    title = "Common Stock and associated Preferred Stock Purchase Rights"
    securities = [
        CoverSecurity("AAA", "Mortgage Bonds due 2030", 1),
        CoverSecurity("BBB", title, 1),
    ]

    assert choose_filing_ticker(securities, 1) == "BBB"


def test_rights_that_are_their_own_security_are_not_the_common_stock():
    """Without 'associated', 'Common Stock Purchase Rights' is a different security."""
    securities = [
        CoverSecurity("AAA", "Common Stock Purchase Rights", 1),
        CoverSecurity("BBB", "Common Stock", 1),
    ]

    assert choose_filing_ticker(securities, 1) == "BBB"


def test_the_first_title_in_a_context_is_the_one_the_symbol_was_filed_with():
    """Constructed: a context carrying two titles, common first and a warrant second."""
    rows = [
        _fact(TICKER, "AAA", "c-1"),
        _fact(TITLE, "Common Stock", "c-1"),
        _fact(TITLE, "Warrants to purchase Common Stock", "c-1"),
    ]
    securities = cover_securities(_filing_from(rows))
    assert securities[0].title == "Common Stock"

    assert choose_filing_ticker(securities, 1) == "AAA"


def test_a_title_pairs_by_context_not_by_position():
    """The first fact has no title; the second's title must not slide onto it."""
    rows = [
        _fact(TICKER, "AAA", "c-1", "0000000001"),
        _fact(TICKER, "BBB", "c-2", "0000000001"),
        _fact(TITLE, "Common Stock", "c-2"),
    ]

    securities = cover_securities(_filing_from(rows))

    assert [(s.symbol, s.title) for s in securities] == [("AAA", None), ("BBB", "Common Stock")]
    assert choose_filing_ticker(securities, 1) == "BBB"


# --- the fallback when no title identifies common equity ----------------------


def test_untitled_symbols_among_several_yield_none():
    """AGNC's recorded rows with every title removed: seven symbols, no way to tell
    the common stock from AGNCN. A guess could be a preferred."""
    rows = [r for r in _rows(PREFERREDS) if r["concept"] == TICKER]
    assert len(rows) == 7, "premise: several tradeable symbols remain"
    assert not any(r["concept"] == TITLE for r in rows), "premise: no titles"

    assert _ticker(rows, AGNC_CIK) is None


def test_a_single_untitled_symbol_is_taken():
    rows = [r for r in _rows(SINGLE) if r["concept"] == TICKER]
    assert len(rows) == 1, "premise: exactly one symbol, and no title"

    assert _ticker(rows, COLD_CIK) == "COLD"


def test_two_untitled_acceptable_symbols_are_ambiguous():
    securities = [
        CoverSecurity("ABC.", None, 1),
        CoverSecurity("XYZ", None, 1),
        CoverSecurity("QRS", None, 1),
    ]

    assert choose_filing_ticker(securities, 1) is None


def test_only_tradeable_symbols_count_toward_the_single_symbol_rule():
    securities = [CoverSecurity("N/A", None, 1), CoverSecurity("XYZ", None, 1)]

    assert choose_filing_ticker(securities, 1) == "XYZ"


def test_a_lone_symbol_titled_as_another_security_is_not_taken():
    """A lone notes ticker (GECCZ, '8.00% Notes due 2028') is not the issuer's stock."""
    securities = [CoverSecurity("GECCZ", "8.00% Notes due 2028", 1)]

    assert choose_filing_ticker(securities, 1) is None


def test_a_lone_symbol_with_an_unrecognised_title_is_taken():
    securities = [CoverSecurity("ABC", "Class A Shares, par value $0.01", 1)]

    assert choose_filing_ticker(securities, 1) == "ABC"


def test_a_cover_page_with_only_unusable_symbols_yields_none():
    securities = [CoverSecurity("N/A", "Common Stock", 1), CoverSecurity("none", None, 1)]

    assert choose_filing_ticker(securities, 1) is None


def test_a_filing_without_xbrl_yields_no_ticker_and_no_error():
    assert cover_securities(_Filing(None)) == []
    assert with_filing_ticker(_event(), cover_securities(_Filing(None))).ticker is None


def test_a_filing_with_xbrl_but_no_symbol_yields_no_ticker():
    rows = [r for r in _rows(SINGLE) if r["concept"] != TICKER]
    assert any(r["concept"] == TITLE for r in rows), "premise: the title remains"

    assert cover_securities(_filing_from(rows)) == []


def test_the_event_records_where_its_ticker_came_from():
    event = with_filing_ticker(_event(), cover_securities(_filing(PREFERREDS)))

    assert (event.ticker, event.ticker_source) == ("AGNC", "filing")


def test_an_event_without_a_filing_ticker_has_neither_ticker_nor_source():
    event = with_filing_ticker(_event(), [])

    assert (event.ticker, event.ticker_source) == (None, None)


def test_the_events_own_cik_decides_whose_symbol_it_takes():
    event = with_filing_ticker(_event(ALABAMA_POWER_CIK), cover_securities(_filing(SOUTHERN)))

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
        (_record("0000000001-26-000001", COLD_CIK), _EightK(_filing(SINGLE))),
        (_record("0000000002-26-000002", 2), _EightK(_Filing(raises=boom))),
        (_record("0000000003-26-000003", FBRX_CIK), _EightK(_filing(DELISTED))),
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


# --- the day-level guard on failed cover-page reads ---------------------------


def _day_of(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, total: int, failing: int) -> None:
    boom = RuntimeError("XBRL instance could not be parsed")
    filings = [
        (
            _record(f"0000000001-26-{i:06d}", COLD_CIK),
            _EightK(_Filing(raises=boom) if i < failing else _filing(SINGLE)),
        )
        for i in range(total)
    ]
    _run_day(monkeypatch, tmp_path, filings)


def _stored(tmp_path: Path, domain: str = "redflag") -> bool:
    return (tmp_path / domain / "2026-09-11.parquet").exists()


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


def test_a_failed_red_flag_day_still_stores_the_insider_domain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """An 8-K XBRL outage must not withhold the domain the published report rests on.

    The insider partition is written; the red-flag partition is not, so the day
    does not count as stored and a rerun retries it.
    """
    with pytest.raises(SystemicTickerReadError):
        _day_of(monkeypatch, tmp_path, total=5, failing=5)

    assert _stored(tmp_path, "insider"), "the insider partition is written before the guard"
    assert not _stored(tmp_path, "redflag")


def test_below_the_minimum_attempts_even_total_failure_is_only_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Four reads, all four failed: too few to say the client is broken."""
    _day_of(monkeypatch, tmp_path, total=4, failing=4)

    assert _stored(tmp_path)


@pytest.mark.parametrize(
    ("title", "is_common"),
    [
        ("Common Stock (including associated preferred stock purchase rights)", True),
        (
            "Common Stock, par value $0.01 per share, and the associated Series A "
            "Junior Participating Preferred Stock Purchase Rights",
            True,
        ),
        ("Common Stock and related Warrants", False),
        ("Class A Common Stock together with the associated Contingent Value Rights", False),
        ("6.75% Non-Cumulative Preference Shares, Series A", False),
        ("Ordinary Preference Shares", False),
    ],
)
def test_attached_purchase_rights_are_stripped_and_other_attachments_are_not(
    title: str, is_common: bool
):
    """Constructed titles in the phrasings a review found on cover pages: only a
    stock purchase right travels with the common stock; warrants, contingent value
    rights and preference shares are different instruments."""
    from arbiter.events.redflags import _names_common_equity

    assert _names_common_equity(title) is is_common


def test_a_lone_preference_share_is_not_taken():
    """A constructed preference-only cover: the issuer's common stock is not listed."""
    securities = [CoverSecurity("ABCD", "6.75% Non-Cumulative Preference Shares, Series A", 1)]

    assert choose_filing_ticker(securities, issuer_cik=1) is None
