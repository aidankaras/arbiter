"""The insider-type breakdown at its seams: the loader's bad rows, the store, the CLI.

Loader cases are built from rows copied from SEC's 2024q1 Insider Transactions Data
Set (tests/fixtures/form345), changing one field of a copy so each case differs from a
real row in exactly the way it names.
"""

import csv
import io
import zipfile
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

import arbiter.cli
from arbiter.arms.evaluate import IssuerScore
from arbiter.cli import app
from arbiter.evaluation.insider_type import (
    MIXED,
    ROUTINE,
    UNCLASSIFIED,
    LoadedHistory,
    ic_by_insider_type,
    load_history,
    load_issuer_day_classes,
)
from arbiter.evaluation.report import render_baseline_report
from arbiter.ingestion.store import write_events
from tests.unit.test_report import GENERATED
from tests.unit.test_report import _evaluation as report_evaluation

FIXTURES = Path(__file__).parent.parent / "fixtures" / "form345"
#: A real one-owner, one-sale Form 4 in the fixtures, filed 29-MAR-2024.
BASE = "0001415889-24-009475"
MEMBERS = ("SUBMISSION", "REPORTINGOWNER", "NONDERIV_TRANS")


def _table(member: str) -> tuple[list[str], list[dict[str, str]]]:
    text = (FIXTURES / f"{member}.tsv").read_text(encoding="utf-8")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
    return list(reader.fieldnames or []), list(reader)


def _variant(member: str, accession: str, **changes: str) -> dict[str, str]:
    """BASE's row in `member` copied under a new accession, with fields changed."""
    _, rows = _table(member)
    original = next(r for r in rows if r["ACCESSION_NUMBER"] == BASE)
    return {**original, "ACCESSION_NUMBER": accession, **changes}


def _zip(path: Path, extra: dict[str, list[dict[str, str]]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for member in MEMBERS:
            header, rows = _table(member)
            out = io.StringIO()
            writer = csv.DictWriter(
                out, header, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows([*rows, *extra.get(member, [])])
            archive.writestr(f"{member}.tsv", out.getvalue())


def _filing(
    accession: str,
    trade_dates: list[str],
    owner: str = "999",
    issuer: str = "555",
    **bad: str,
) -> dict[str, list[dict[str, str]]]:
    """One synthetic filing: BASE's rows under `accession`, one sale per date in `trade_dates`.

    `bad` overrides one field in whichever member holds it.
    """
    submission = _variant("SUBMISSION", accession, ISSUERCIK=issuer)
    owners = _variant("REPORTINGOWNER", accession, RPTOWNERCIK=owner)
    trades = [_variant("NONDERIV_TRANS", accession, TRANS_DATE=d) for d in trade_dates]
    for row in (submission, owners, *trades):
        row.update({k: v for k, v in bad.items() if k in row})
    return {"SUBMISSION": [submission], "REPORTINGOWNER": [owners], "NONDERIV_TRANS": trades}


def _merge(*parts: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    return {m: [row for part in parts for row in part.get(m, [])] for m in MEMBERS}


def test_an_accession_trading_in_two_months_credits_both(tmp_path: Path):
    # Filed 29-MAR-2024; sales on 28-FEB and 27-MAR.
    _zip(tmp_path / "2024q1_form345.zip", _filing("X-1", ["28-FEB-2024", "27-MAR-2024"]))
    months = {(y, m) for y, m, _ in load_history(tmp_path).trades[(999, 555)]}
    assert months == {(2024, 2), (2024, 3)}


def test_rows_that_cannot_be_parsed_are_excluded_and_counted_not_fatal(tmp_path: Path):
    good = _filing("G-1", ["28-MAR-2024"])
    _zip(
        tmp_path / "2024q1_form345.zip",
        _merge(
            good,
            _filing("B-1", ["28-MAR-2024"], "1001", "601", ISSUERCIK="n/a"),
            _filing("B-2", ["28-MAR-2024"], "1002", "602", FILING_DATE="31-XYZ-2024"),
            _filing("B-3", ["not a date"], "1003", "603"),
            _filing("B-4", ["28-MAR-2024"], "1004", "604", RPTOWNERCIK=""),
        ),
    )
    loaded = load_history(tmp_path)
    assert loaded.unparseable == 4
    assert loaded.future_dated == 0
    # The good filing survives and the bad ones credit nobody.
    synthetic = {
        pair: t for pair, t in loaded.trades.items() if pair[0] in {999, 1001, 1002, 1003, 1004}
    }
    assert synthetic == {(999, 555): frozenset({(2024, 3, date(2024, 3, 29))})}
    # The counts survive the cache.
    assert load_history(tmp_path).unparseable == 4


def test_a_trade_dated_after_its_filing_is_excluded_and_counted(tmp_path: Path):
    # Filed 29-MAR-2024: the 15-APR sale is in the future, the 28-MAR sale is not.
    _zip(tmp_path / "2024q1_form345.zip", _filing("F-1", ["28-MAR-2024", "15-APR-2024"]))
    loaded = load_history(tmp_path)
    assert loaded.future_dated == 1
    assert {(y, m) for y, m, _ in loaded.trades[(999, 555)]} == {(2024, 3)}
    assert load_history(tmp_path).future_dated == 1


def test_the_loaded_quarters_are_read_from_the_zip_names(tmp_path: Path):
    for name in ("2025q4", "2023q1", "2026q1"):
        _zip(tmp_path / f"{name}_form345.zip", {})
    loaded = load_history(tmp_path)
    assert (loaded.first_quarter, loaded.last_quarter) == ("2023q1", "2026q1")


def test_skipped_rows_are_reported_only_when_there_are_some():
    scores = {date(2026, 8, 3): [IssuerScore(1, 0.5, 0.1)]}
    clean = ic_by_insider_type(scores, {}, history=LoadedHistory({}, "2023q1", "2025q4", 0, 0))
    dirty = ic_by_insider_type(scores, {}, history=LoadedHistory({}, "2023q1", "2025q4", 4, 1))
    texts = [
        " ".join(
            render_baseline_report(
                report_evaluation(0.01, 0.02, 0.5), GENERATED, insider_type=b
            ).split()
        )
        for b in (clean, dirty)
    ]
    assert "Excluded from the history" not in texts[0]
    assert (
        "Excluded from the history: 4 history rows had an unparseable date or CIK and "
        "1 history trades were dated after their filing."
    ) in texts[1]


# --- the store ---------------------------------------------------------------


class _Event(BaseModel):
    accession_no: str
    as_of: datetime
    cik: int
    owner_ciks: list[int]


class _Label(BaseModel):
    accession_no: str


def _routine(
    owner: int, issuer: int
) -> dict[tuple[int, int], frozenset[tuple[int, int, date]]]:
    return {(owner, issuer): frozenset((y, 3, date(y, 4, 1)) for y in (2023, 2024, 2025))}


def _opportunistic(
    owner: int, issuer: int
) -> dict[tuple[int, int], frozenset[tuple[int, int, date]]]:
    months = {2023: 1, 2024: 2, 2025: 3}
    return {(owner, issuer): frozenset((y, m, date(y, m, 28)) for y, m in months.items())}


DAY = date(2026, 8, 3)
#: 21:00 US/Eastern on DAY, which is already the next date in UTC.
LATE = datetime(2026, 8, 4, 1, 0, tzinfo=UTC)


def test_only_labelled_events_classify_an_issuer_day_and_it_is_keyed_by_the_store_day(
    tmp_path: Path,
):
    history = {**_routine(1, 7), **_opportunistic(2, 7)}
    write_events(
        [
            _Event(accession_no="A", as_of=LATE, cik=7, owner_ciks=[1]),
            _Event(accession_no="B", as_of=LATE, cik=7, owner_ciks=[2]),
        ],
        tmp_path,
        "insider",
        DAY,
    )
    write_events([_Label(accession_no="A")], tmp_path, "labels-insider", DAY)
    # A processed day with no labels contributes nothing.
    write_events(
        [_Event(accession_no="C", as_of=LATE, cik=8, owner_ciks=[1])],
        tmp_path,
        "insider",
        date(2026, 8, 4),
    )

    classes = load_issuer_day_classes(
        tmp_path, "insider", [DAY, date(2026, 8, 4), date(2026, 8, 5)], history
    )
    # Premise: B alone, or A and B together, would not be routine.
    both = load_issuer_day_classes(tmp_path, "insider", [DAY], {**history})
    assert both == classes
    assert classes == {(DAY, 7): ROUTINE}


def test_an_unlabelled_event_would_have_made_the_day_mixed_had_it_been_counted(tmp_path: Path):
    # Same events as above, both labelled: the disagreement is real, so the filter above
    # is what keeps the day routine.
    history = {**_routine(1, 7), **_opportunistic(2, 7)}
    write_events(
        [
            _Event(accession_no="A", as_of=LATE, cik=7, owner_ciks=[1]),
            _Event(accession_no="B", as_of=LATE, cik=7, owner_ciks=[2]),
        ],
        tmp_path,
        "insider",
        DAY,
    )
    write_events(
        [_Label(accession_no="A"), _Label(accession_no="B")], tmp_path, "labels-insider", DAY
    )
    assert load_issuer_day_classes(tmp_path, "insider", [DAY], history) == {(DAY, 7): MIXED}


# --- the CLI -----------------------------------------------------------------

runner = CliRunner()


def _cli_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    store = tmp_path / "events"
    write_events(
        [
            _Event(accession_no="A", as_of=LATE, cik=7, owner_ciks=[1]),
            _Event(accession_no="B", as_of=LATE, cik=8, owner_ciks=[]),
        ],
        store,
        "insider",
        DAY,
    )
    write_events(
        [_Label(accession_no="A"), _Label(accession_no="B")], store, "labels-insider", DAY
    )
    evaluation = replace(
        report_evaluation(0.01, 0.02, 0.5),
        test_days=[DAY],
        issuer_scores={DAY: [IssuerScore(7, 0.5, 0.1), IssuerScore(8, 0.4, 0.2)]},
    )
    monkeypatch.setattr(arbiter.cli, "stored_days", lambda *_a: [DAY])
    monkeypatch.setattr(arbiter.cli, "load_days", lambda *_a: [])
    monkeypatch.setattr(arbiter.cli, "evaluate_baseline", lambda *_a, **_k: evaluation)
    monkeypatch.setattr(arbiter.cli, "load_sizes", lambda *_a: ({}, 0))
    monkeypatch.setattr(arbiter.cli, "current_session_date", lambda: GENERATED)
    monkeypatch.setattr(arbiter.cli, "render_baseline_figure", lambda *_a: "<svg/>")
    return store


def test_report_reads_the_insider_history_directory_into_the_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = _cli_setup(tmp_path, monkeypatch)
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    (history_dir / "2023q1_form345.zip").write_bytes(b"")
    seen: list[Path] = []

    def fake_load(directory: Path) -> LoadedHistory:
        seen.append(directory)
        return LoadedHistory(_routine(1, 7), "2023q1", "2025q4", 3, 2)

    monkeypatch.setattr(arbiter.cli, "load_history", fake_load)
    result = runner.invoke(
        app,
        ["report", "--root", str(store), "--out", str(tmp_path / "out"),
         "--insider-history", str(history_dir)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert seen == [history_dir]
    text = " ".join((tmp_path / "out" / "baseline-insider.md").read_text().split())
    assert "## By insider type" in text
    # Issuer 7's owner is routine; issuer 8's event has no owners. One classified
    # issuer-day cannot reach the per-day minimum.
    assert "1 were unclassified, 0 were mixed" in text
    assert "1 were classified but fell in a type-day below the minimum" in text
    assert "from the 2023q1 to the 2025q4 data sets" in text
    assert "3 history rows had an unparseable date or CIK" in text
    assert UNCLASSIFIED in text.lower()


def test_report_without_history_zips_omits_the_section_with_a_line_saying_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = _cli_setup(tmp_path, monkeypatch)
    empty = tmp_path / "none"
    empty.mkdir()
    result = runner.invoke(
        app,
        ["report", "--root", str(store), "--out", str(tmp_path / "out"),
         "--insider-history", str(empty)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    text = (tmp_path / "out" / "baseline-insider.md").read_text()
    assert "Omitted: no *_form345.zip under" in text
    assert "| Routine |" not in text
