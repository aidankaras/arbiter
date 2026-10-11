"""Routine versus opportunistic insiders, after Cohen, Malloy and Pomorski (2012).

Expectations come from the stated definitions: for an event in month M of year Y,
an insider is routine if they traded in month M of each of Y-1, Y-2 and Y-3,
opportunistic if they traded in each of those years but not that month every time,
unclassified otherwise. Every history here is built by hand so a class can be
derived without running the code.
"""

import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from arbiter.arms.evaluate import IssuerScore
from arbiter.evaluation import insider_type
from arbiter.evaluation.insider_type import (
    MIXED,
    OPPORTUNISTIC,
    ROUTINE,
    UNCLASSIFIED,
    History,
    LoadedHistory,
    Trade,
    available_archives,
    classify_event,
    classify_pair,
    ic_by_insider_type,
    issuer_day_class,
    issuer_day_classes,
    load_history,
)
from arbiter.evaluation.metrics import MIN_EVENTS_PER_DAY
from arbiter.evaluation.report import render_baseline_report
from tests.unit.test_report import GENERATED
from tests.unit.test_report import _evaluation as report_evaluation

OWNER, ISSUER = 100, 7
#: Accepted in June 2026, Eastern. The three prior years are 2023, 2024 and 2025.
JUNE_2026 = datetime(2026, 6, 15, 15, 0, tzinfo=UTC)


def _trade(year: int, month: int, filed: date | None = None) -> Trade:
    """A trade in `year`-`month`, filed by default the first of the next month."""
    default = date(year + (month == 12), month % 12 + 1, 1)
    return (year, month, filed or default)


def _history(*trades: Trade, owner: int = OWNER, issuer: int = ISSUER) -> History:
    return {(owner, issuer): frozenset(trades)}


def test_the_same_month_in_each_of_three_prior_years_is_routine():
    history = _history(_trade(2023, 6), _trade(2024, 6), _trade(2025, 6))
    assert classify_pair(history, OWNER, ISSUER, JUNE_2026) == ROUTINE


@pytest.mark.parametrize("off_year", [2023, 2024, 2025])
def test_every_year_but_no_common_month_is_opportunistic(off_year: int):
    # Whichever single year breaks the pattern, two matching years are not enough.
    history = _history(*(_trade(y, 9 if y == off_year else 6) for y in (2023, 2024, 2025)))
    assert classify_pair(history, OWNER, ISSUER, JUNE_2026) == OPPORTUNISTIC


def test_months_shared_only_pairwise_are_not_a_routine_month():
    history = _history(
        _trade(2023, 1), _trade(2023, 2), _trade(2024, 2), _trade(2024, 3),
        _trade(2025, 1), _trade(2025, 3),
    )  # fmt: skip
    assert classify_pair(history, OWNER, ISSUER, JUNE_2026) == OPPORTUNISTIC


@pytest.mark.parametrize("event_month", [1, 3, 6, 12])
def test_a_routine_trader_is_routine_in_every_month_not_only_their_own(event_month: int):
    # The classification is of the trader: this one always trades in March.
    history = _history(_trade(2023, 3), _trade(2024, 3), _trade(2025, 3))
    event = datetime(2026, event_month, 15, 15, tzinfo=UTC)
    assert classify_pair(history, OWNER, ISSUER, event) == ROUTINE


def test_extra_trades_in_other_months_do_not_spoil_routine():
    history = _history(
        _trade(2023, 6), _trade(2023, 1), _trade(2024, 6), _trade(2025, 6), _trade(2025, 12)
    )
    assert classify_pair(history, OWNER, ISSUER, JUNE_2026) == ROUTINE


@pytest.mark.parametrize("missing", [2023, 2024, 2025])
def test_a_prior_year_without_any_trade_leaves_the_insider_unclassified(missing: int):
    history = _history(*(_trade(y, 6) for y in (2023, 2024, 2025) if y != missing))
    assert classify_pair(history, OWNER, ISSUER, JUNE_2026) == UNCLASSIFIED


def test_a_pair_with_no_history_is_unclassified():
    assert classify_pair({}, OWNER, ISSUER, JUNE_2026) == UNCLASSIFIED


def test_trades_from_before_the_three_year_window_do_not_stand_in_for_a_missing_year():
    history = _history(_trade(2022, 6), _trade(2024, 6), _trade(2025, 6))
    assert classify_pair(history, OWNER, ISSUER, JUNE_2026) == UNCLASSIFIED


def test_a_trade_in_the_events_own_year_is_ignored():
    history = _history(_trade(2024, 6), _trade(2025, 6), _trade(2026, 6, date(2026, 6, 1)))
    # Premise: a year later the 2026 trade is a prior year, and the class is routine.
    assert classify_pair(history, OWNER, ISSUER, datetime(2027, 6, 15, tzinfo=UTC)) == ROUTINE
    assert classify_pair(history, OWNER, ISSUER, JUNE_2026) == UNCLASSIFIED


@pytest.mark.parametrize("late_year", [2023, 2024, 2025])
def test_a_trade_filed_after_the_event_was_accepted_is_ignored_whatever_its_year(
    late_year: int,
):
    # A 4/A restating an old trade can be filed long after it: the filter applies to
    # every year of the window, not only the latest.
    history = _history(
        *(
            _trade(y, 6, filed=date(2026, 6, 20) if y == late_year else None)
            for y in (2023, 2024, 2025)
        )
    )
    # Premise: once the filing is public the same history is routine.
    assert (
        classify_pair(history, OWNER, ISSUER, datetime(2026, 6, 25, 15, tzinfo=UTC)) == ROUTINE
    )
    assert classify_pair(history, OWNER, ISSUER, JUNE_2026) == UNCLASSIFIED


def test_a_trade_filed_on_the_acceptance_date_is_not_yet_history():
    same_day = _trade(2025, 6, filed=date(2026, 6, 15))
    history = _history(_trade(2023, 6), _trade(2024, 6), same_day)
    assert classify_pair(history, OWNER, ISSUER, JUNE_2026) == UNCLASSIFIED
    assert classify_pair(history, OWNER, ISSUER, JUNE_2026 + timedelta(days=1)) == ROUTINE


def test_the_year_is_the_us_eastern_year_of_acceptance():
    history = _history(_trade(2023, 6), _trade(2024, 6), _trade(2025, 6))
    # 03:00 UTC on 1 January 2026 is 22:00 on 31 December 2025 in New York, where the
    # window is 2022-2024 and 2022 is missing.
    assert classify_pair(history, OWNER, ISSUER, datetime(2026, 1, 1, 3, tzinfo=UTC)) == (
        UNCLASSIFIED
    )
    assert classify_pair(history, OWNER, ISSUER, datetime(2026, 1, 1, 6, tzinfo=UTC)) == ROUTINE


def test_history_is_per_pair_not_per_person():
    elsewhere = _history(_trade(2023, 6), _trade(2024, 6), _trade(2025, 6), issuer=ISSUER + 1)
    assert classify_pair(elsewhere, OWNER, ISSUER, JUNE_2026) == UNCLASSIFIED
    # Premise: at that other issuer the same owner is routine.
    assert classify_pair(elsewhere, OWNER, ISSUER + 1, JUNE_2026) == ROUTINE
    # Nor does another owner's record at this issuer count.
    colleague = _history(_trade(2023, 6), _trade(2024, 6), _trade(2025, 6), owner=OWNER + 1)
    assert classify_pair(colleague, OWNER, ISSUER, JUNE_2026) == UNCLASSIFIED


def _two_owner_history() -> History:
    routine = [_trade(y, 6) for y in (2023, 2024, 2025)]
    opportunistic = [_trade(2023, 6), _trade(2024, 9), _trade(2025, 6)]
    return {
        (1, ISSUER): frozenset(routine),
        (2, ISSUER): frozenset(routine),
        (3, ISSUER): frozenset(opportunistic),
        (4, ISSUER): frozenset({_trade(2025, 6)}),
    }


def test_owners_that_agree_give_the_event_their_class():
    assert classify_event(_two_owner_history(), (1, 2), ISSUER, JUNE_2026) == ROUTINE


def test_owners_that_disagree_leave_the_event_unclassified():
    history = _two_owner_history()
    assert classify_event(history, (1, 3), ISSUER, JUNE_2026) == UNCLASSIFIED
    # Order must not decide: the first owner's class is not the event's.
    assert classify_event(history, (3, 1), ISSUER, JUNE_2026) == UNCLASSIFIED
    # Premise: each owner alone is classified.
    assert classify_event(history, (1,), ISSUER, JUNE_2026) == ROUTINE
    assert classify_event(history, (3,), ISSUER, JUNE_2026) == OPPORTUNISTIC


def test_an_unclassified_co_owner_leaves_the_event_unclassified():
    assert classify_event(_two_owner_history(), (1, 4), ISSUER, JUNE_2026) == UNCLASSIFIED


def test_an_event_with_no_recorded_owner_is_unclassified():
    assert classify_event(_two_owner_history(), (), ISSUER, JUNE_2026) == UNCLASSIFIED


@pytest.mark.parametrize(
    ("classes", "expected"),
    [
        ([ROUTINE, ROUTINE], ROUTINE),
        ([OPPORTUNISTIC], OPPORTUNISTIC),
        ([ROUTINE, OPPORTUNISTIC], MIXED),
        ([ROUTINE, UNCLASSIFIED], ROUTINE),
        ([UNCLASSIFIED, UNCLASSIFIED], UNCLASSIFIED),
        ([], UNCLASSIFIED),
    ],
)
def test_an_issuer_day_takes_the_class_its_classified_events_share(
    classes: list[str], expected: str
):
    assert issuer_day_class(classes) == expected


def test_stored_rows_without_owners_are_unclassified_and_others_are_aggregated():
    history = _two_owner_history()
    day = date(2026, 6, 15)

    def row(issuer: int, owners: tuple[int, ...] | None) -> dict[str, object]:
        base: dict[str, object] = {"cik": issuer, "as_of": JUNE_2026}
        if owners is not None:
            base["owner_ciks"] = owners
        return base

    rows = [
        (day, row(ISSUER, (1,))),  # routine
        (day, row(ISSUER, (2,))),  # routine
        (day, row(11, (1,))),  # issuer 11: owner 1 has no history there
        (day, row(12, None)),  # a partition written before owners were stored
    ]
    assert issuer_day_classes(history, rows) == {
        (day, ISSUER): ROUTINE,
        (day, 11): UNCLASSIFIED,
        (day, 12): UNCLASSIFIED,
    }
    rows.append((day, row(ISSUER, (3,))))  # opportunistic at the same issuer-day
    assert issuer_day_classes(history, rows)[(day, ISSUER)] == MIXED


# --- the breakdown -----------------------------------------------------------


def _scores_and_classes(
    days: int = 5,
) -> tuple[dict[date, list[IssuerScore]], dict[tuple[date, int], str]]:
    """Per day: opportunistic issuers rank perfectly, routine ones rank backwards.

    Plus two unclassified issuers, one mixed one, and on the last day a routine
    group too thin to count.
    """
    scores: dict[date, list[IssuerScore]] = {}
    classes: dict[tuple[date, int], str] = {}
    n = MIN_EVENTS_PER_DAY
    for offset in range(days):
        day = date(2026, 8, 3) + timedelta(days=offset)
        rows: list[IssuerScore] = []
        thin = offset == days - 1
        for k in range(n):
            rows.append(IssuerScore(1000 + k, k / 100, k / 100))
            classes[(day, 1000 + k)] = OPPORTUNISTIC
        for k in range(2 if thin else n):
            rows.append(IssuerScore(2000 + k, k / 100, -k / 100))
            classes[(day, 2000 + k)] = ROUTINE
        rows += [IssuerScore(3000, 0.5, 0.0), IssuerScore(3001, 0.5, 0.0)]
        rows.append(IssuerScore(4000, 0.5, 0.0))
        classes[(day, 4000)] = MIXED
        scores[day] = rows
    return scores, classes


def test_each_type_is_scored_within_itself_and_the_whole_sample_is_accounted_for():
    scores, classes = _scores_and_classes()
    result = ic_by_insider_type(scores, classes)
    opportunistic, routine = result.rows
    assert (opportunistic.label, routine.label) == (OPPORTUNISTIC, ROUTINE)
    assert opportunistic.ic is not None
    assert opportunistic.ic.mean == pytest.approx(1.0)
    assert opportunistic.days == 5
    assert routine.ic is not None
    assert routine.ic.mean == pytest.approx(-1.0)
    # The thin day's two routine issuers are dropped; the other four days count.
    assert routine.days == 4
    assert result.dropped == 2
    assert result.unclassified == 2 * 5
    assert result.mixed == 5
    assert result.scored_issuer_days == sum(len(rows) for rows in scores.values())
    in_rows = sum(row.issuer_days for row in result.rows)
    assert in_rows + result.dropped + result.unclassified + result.mixed == (
        result.scored_issuer_days
    )


# --- the report section ------------------------------------------------------


_LOADED = LoadedHistory({}, "2023q1", "2025q4", 0, 0)


def test_the_section_states_the_definitions_cites_the_paper_and_follows_the_size_section():
    scores, classes = _scores_and_classes()
    text = render_baseline_report(
        report_evaluation(0.01, 0.02, 0.5),
        GENERATED,
        insider_type=ic_by_insider_type(scores, classes, history=_LOADED),
    )
    flat = " ".join(text.split())
    assert text.index("## By issuer size") < text.index("## By insider type")
    assert text.index("## By insider type") < text.index("## Calibration")
    assert "| Opportunistic |" in text
    assert "| Routine |" in text
    assert "Cohen, Malloy and Pomorski (2012)" in flat
    assert '"Decoding Inside Information"' in flat
    assert "from the 2023q1 to the 2025q4 data sets" in flat
    assert "trades filed after 2025q4 are absent" in flat
    assert "the trader is classified, not the trade" in flat
    assert "the insider is the owner at one issuer, not the person" in flat
    assert "including trades under 10b5-1 plans, which the scored events exclude" in flat
    assert "history years come from transaction dates" in flat
    assert "Excluded from the history" not in flat
    assert "does not test whether the rows differ" in flat
    assert "5 were mixed" in flat
    assert "2 were classified but fell in a type-day below the minimum" in flat
    assert "10 were unclassified" in flat


def test_without_a_breakdown_the_section_says_why():
    text = render_baseline_report(
        report_evaluation(0.01, 0.02, 0.5),
        GENERATED,
        insider_type_omitted="Omitted: no history.",
    )
    assert "## By insider type" in text
    assert "Omitted: no history." in text
    assert "| Routine |" not in text


# --- the loader --------------------------------------------------------------

#: Rows copied from SEC's 2024q1 Insider Transactions Data Set (see the TSVs).
FIXTURES = Path(__file__).parent.parent / "fixtures" / "form345"
#: accession -> (issuer, owners, expected trades as (year, month, filed)); None if excluded.
EXPECTED: dict[str, tuple[int, tuple[int, ...], list[tuple[int, int, date]] | None]] = {
    # Form 4, one sale reported for three joint owners.
    "0001104659-24-040609": (
        1716951,
        (1607203, 1836609, 1883957),
        [(2024, 3, date(2024, 3, 28))],
    ),
    # Form 4, one sale.
    "0001415889-24-009475": (1604821, (2010019,), [(2024, 3, date(2024, 3, 29))]),
    # Form 4, one purchase.
    "0001437749-24-010117": (1490161, (2008040,), [(2024, 3, date(2024, 3, 29))]),
    # Form 4/A amending a 2022 sale.
    "0001104659-24-040732": (796343, (1674730,), [(2022, 10, date(2024, 3, 28))]),
    # Form 4 reporting only a grant (code A).
    "0001628280-24-013866": (912728, (2011491,), None),
    # Form 4 reporting only tax withholding (code F).
    "0000950170-24-038816": (899460, (1360928,), None),
    # Form 4 with a sale, an option exercise (M) and withholding (F): only the sale counts.
    "0000920148-24-000025": (920148, (1537208,), [(2024, 3, date(2024, 3, 29))]),
    # Form 5 reporting sales: not a Form 4, so not history.
    "0001156375-24-000016": (1156375, (1341358,), None),
}


def _build_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("SUBMISSION", "REPORTINGOWNER", "NONDERIV_TRANS"):
            archive.write(FIXTURES / f"{name}.tsv", f"{name}.tsv")


def test_the_loader_credits_every_owner_with_open_market_trades_on_forms_4_and_4a(
    tmp_path: Path,
):
    _build_zip(tmp_path / "2024q1_form345.zip")
    history = load_history(tmp_path).trades

    expected: dict[tuple[int, int], frozenset[tuple[int, int, date]]] = {}
    for issuer, owners, trades in EXPECTED.values():
        if trades is None:
            continue
        for owner in owners:
            expected[(owner, issuer)] = frozenset(trades)
    assert history == expected
    # Spelled out for the two shapes that matter most.
    assert {pair for pair in history if pair[1] == 1716951} == {
        (1607203, 1716951),
        (1836609, 1716951),
        (1883957, 1716951),
    }
    assert history[(1674730, 796343)] == {(2022, 10, date(2024, 3, 28))}
    for excluded in ((2011491, 912728), (1360928, 899460), (1341358, 1156375)):
        assert excluded not in history


def test_a_second_load_reads_the_cache_and_does_not_open_the_zips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _build_zip(tmp_path / "2024q1_form345.zip")
    first = load_history(tmp_path)
    assert list(tmp_path.glob("insider_history_*.parquet"))

    def refuse(*_args: object, **_kwargs: object) -> None:
        pytest.fail("the zip was reopened")

    monkeypatch.setattr(zipfile, "ZipFile", refuse)
    assert load_history(tmp_path) == first


def test_a_changed_zip_set_does_not_reuse_a_stale_cache(tmp_path: Path):
    _build_zip(tmp_path / "2024q1_form345.zip")
    before = load_history(tmp_path)
    _build_zip(tmp_path / "2024q2_form345.zip")
    # Same rows again, so only the cache key can tell the two sets apart.
    assert load_history(tmp_path).trades == before.trades
    assert len(list(tmp_path.glob("insider_history_*.parquet"))) == 2


def test_no_zips_is_an_error_not_an_empty_history(tmp_path: Path):
    assert available_archives(tmp_path) == []
    assert available_archives(tmp_path / "missing") == []
    with pytest.raises(FileNotFoundError):
        load_history(tmp_path)


def test_the_module_names_the_paper_it_replicates():
    doc = insider_type.__doc__ or ""
    assert "Cohen, Malloy and Pomorski (2012)" in doc
