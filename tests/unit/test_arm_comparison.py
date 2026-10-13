"""The agent is scored by the baseline's own machinery, on the same issuer-days."""

import random
from datetime import date, timedelta

import pytest

from arbiter.arms.evaluate import DayOfEvents, evaluate_baseline, forecast_walk_forward
from arbiter.arms.features import FEATURE_NAMES
from arbiter.evaluation.comparison import WindowNotScoredError, compare, daily_ic

START = date(2026, 3, 2)
DAYS = [START + timedelta(days=offset) for offset in range(40)]


def _day(day: date, issuers: int = 24) -> DayOfEvents:
    """A day where every third issuer filed twice, and purchases tend to rise."""
    generator = random.Random(day.toordinal())
    features, returns, owners, accessions = [], [], [], []
    for issuer in range(issuers):
        purchase = float(issuer % 2)
        issuer_return = 0.006 * (1 if purchase else -1) + generator.gauss(0, 0.006)
        for filing in range(2 if issuer % 3 == 0 else 1):
            row = dict.fromkeys(FEATURE_NAMES, 0.0)
            row["is_purchase"] = purchase
            row["log_value_usd"] = 12.0 + filing
            features.append(row)
            returns.append(issuer_return)
            owners.append(issuer)
            accessions.append(f"{day.isoformat()}-{issuer}-{filing}")
    return DayOfEvents(
        day=day,
        features=features,
        outcomes=[int(value > 0) for value in returns],
        returns=returns,
        issuers=owners,
        exit_sessions=[day] * len(returns),
        accessions=accessions,
    )


@pytest.fixture(scope="module")
def days() -> list[DayOfEvents]:
    return [_day(day) for day in DAYS]


@pytest.fixture(scope="module")
def scored(days: list[DayOfEvents]) -> dict[date, list[float]]:
    forecasts: dict[date, list[float]] = {}
    for _, _, block in forecast_walk_forward(days):
        forecasts.update(block)
    return forecasts


def test_scoring_the_baselines_forecasts_reproduces_its_own_evaluation(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    """The oracle: the agent's scoring path, fed the baseline, gives the baseline's numbers."""
    reference = evaluate_baseline(days, overlap_lags=5)
    ours, issuer_days = daily_ic([d for d in days if d.day in scored], scored)
    assert ours == reference.daily_ic
    assert issuer_days == reference.test_issuer_days
    # The premise: issuers with two filings exist, so crediting by issuer is exercised.
    assert any(len(set(d.issuers)) < len(d) for d in days)


def _window(scored: dict[date, list[float]]) -> tuple[date, date]:
    return min(scored), max(scored)


def _baseline_by_accession(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> dict[str, float]:
    return {
        accession: forecast
        for day in days
        if day.day in scored
        for accession, forecast in zip(day.accessions, scored[day.day], strict=True)
    }


def test_an_agent_that_forecasts_what_the_baseline_did_differs_by_nothing(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    agent = _baseline_by_accession(days, scored)
    result = compare(days, agent, _window(scored), overlap_lags=5)
    assert result.agent == result.baseline
    assert result.difference.mean == 0.0
    assert result.dropped_issuer_days == 0


def test_the_difference_is_agent_minus_baseline_day_by_day(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    # Reversing the baseline's forecasts reverses its ranking: every day's
    # coefficient changes sign, so the paired difference is twice the negative.
    agent = {k: 1.0 - v for k, v in _baseline_by_accession(days, scored).items()}
    result = compare(days, agent, _window(scored), overlap_lags=5)
    assert result.difference.mean == pytest.approx(-2 * result.baseline.mean)
    assert result.agent.mean == pytest.approx(-result.baseline.mean)
    # The premise: the baseline found a signal, so the reversal moved something.
    assert result.baseline.mean > 0.1


def test_an_issuer_the_agent_did_not_fully_forecast_leaves_both_arms(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    agent = _baseline_by_accession(days, scored)
    first_scored = min(scored)
    # Issuer 0 filed twice on every day; forget one of its filings on one day.
    missing = f"{first_scored.isoformat()}-0-1"
    assert missing in agent
    del agent[missing]
    result = compare(days, agent, _window(scored), overlap_lags=5)
    full = compare(days, _baseline_by_accession(days, scored), _window(scored), overlap_lags=5)
    assert result.dropped_issuer_days == 1
    assert result.issuer_days == full.issuer_days - 1
    # Dropped from both: the arms still agree exactly.
    assert result.agent == result.baseline


def test_a_window_reaching_into_the_unscored_start_is_refused(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    agent = _baseline_by_accession(days, scored)
    with pytest.raises(WindowNotScoredError):
        compare(days, agent, (DAYS[0], max(scored)), overlap_lags=5)


def test_a_day_too_thin_to_rank_contributes_no_coefficient() -> None:
    thin, full = _day(DAYS[0], issuers=9), _day(DAYS[1], issuers=24)
    # The premise: one day sits below the ten-issuer minimum and one above it.
    assert len(set(thin.issuers)) < 10 <= len(set(full.issuers))
    forecasts = {d.day: [float(i % 2) for i in d.issuers] for d in (thin, full)}
    daily, issuer_days = daily_ic([thin, full], forecasts)
    assert list(daily) == [full.day]
    assert issuer_days == 24


def _window_issuer_days(days: list[DayOfEvents], window: tuple[date, date]) -> int:
    return sum(len(set(d.issuers)) for d in days if window[0] <= d.day <= window[1])


def test_later_data_does_not_move_the_baseline_a_window_is_compared_against(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    # The walk-forward's initial cut and refit blocks are counted in stored days;
    # days stored after the window must not change what the window is scored on.
    agent = _baseline_by_accession(days, scored)
    later = [_day(DAYS[-1] + timedelta(days=k)) for k in range(1, 41)]
    window = _window(scored)
    assert compare(days + later, agent, window, overlap_lags=5) == compare(
        days, agent, window, overlap_lags=5
    )


def test_every_issuer_day_in_the_window_is_compared_or_counted(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    agent = _baseline_by_accession(days, scored)
    thin_day = sorted(scored)[3]
    # Forget every filing of 15 of the day's 24 issuers: the 9 left are too few to rank.
    for accession in [a for a in agent if a.startswith(thin_day.isoformat())]:
        if int(accession.split("-")[3]) < 15:
            del agent[accession]
    window = _window(scored)
    result = compare(days, agent, window, overlap_lags=5)
    assert (result.dropped_issuer_days, result.unranked_issuer_days) == (15, 9)
    total = result.issuer_days + result.dropped_issuer_days + result.unranked_issuer_days
    assert total == _window_issuer_days(days, window)


def test_a_day_the_agent_cannot_rank_leaves_every_line_of_the_report(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    agent = _baseline_by_accession(days, scored)
    flat_day = min(scored)
    for accession in agent:
        if accession.startswith(flat_day.isoformat()):
            agent[accession] = 0.5
    window = _window(scored)
    full = compare(days, _baseline_by_accession(days, scored), window, overlap_lags=5)
    result = compare(days, agent, window, overlap_lags=5)
    # Like for like: the agent, the baseline and the difference cover the same days.
    assert result.agent.days == result.baseline.days == result.difference.days
    assert result.difference.days == full.difference.days - 1
    assert result.agent_undefined_days == 1
    assert result.difference.mean == 0.0
    total = result.issuer_days + result.dropped_issuer_days + result.unranked_issuer_days
    assert total == _window_issuer_days(days, window)


def test_window_days_the_agent_has_not_run_are_counted(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    first_two = sorted(scored)[:2]
    agent = {
        accession: forecast
        for accession, forecast in _baseline_by_accession(days, scored).items()
        if accession[:10] in {d.isoformat() for d in first_two}
    }
    result = compare(days, agent, _window(scored), overlap_lags=5)
    assert result.window_days == len(scored)
    assert result.days_not_run == len(scored) - 2


def test_a_window_with_no_labelled_day_is_refused(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    agent = _baseline_by_accession(days, scored)
    after = DAYS[-1] + timedelta(days=10)
    with pytest.raises(WindowNotScoredError, match="no window day"):
        compare(days, agent, (after, after + timedelta(days=5)), overlap_lags=5)
    with pytest.raises(ValueError, match="ends before"):
        compare(days, agent, (max(scored), min(scored)), overlap_lags=5)


def test_dropping_an_issuer_equals_it_never_having_filed(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    # On the last day no later fold trains on it, so removing the issuer's rows
    # by hand is an independent reference for what the drop must produce.
    last = max(scored)
    agent = _baseline_by_accession(days, scored)
    del agent[f"{last.isoformat()}-0-1"]
    result = compare(days, agent, _window(scored), overlap_lags=5)
    never = [d if d.day != last else _without_issuer(d, 0) for d in days]
    expected = compare(
        never, _baseline_by_accession(days, scored), _window(scored), overlap_lags=5
    )
    assert (result.agent, result.baseline, result.difference) == (
        expected.agent,
        expected.baseline,
        expected.difference,
    )


def _without_issuer(day: DayOfEvents, issuer: int) -> DayOfEvents:
    rows = [i for i, owner in enumerate(day.issuers) if owner != issuer]
    return DayOfEvents(
        day=day.day,
        features=[day.features[i] for i in rows],
        outcomes=[day.outcomes[i] for i in rows],
        returns=[day.returns[i] for i in rows],
        issuers=[day.issuers[i] for i in rows],
        exit_sessions=[day.exit_sessions[i] for i in rows],
        accessions=[day.accessions[i] for i in rows],
    )


def test_every_summary_allows_for_the_overlap_it_was_given(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    result = compare(
        days, _baseline_by_accession(days, scored), _window(scored), overlap_lags=5
    )
    assert {s.overlap_lags for s in (result.agent, result.baseline, result.difference)} == {5}


def test_a_day_after_the_window_is_not_compared(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    window = (min(scored), sorted(scored)[-2])
    result = compare(days, _baseline_by_accession(days, scored), window, overlap_lags=5)
    assert result.issuer_days == _window_issuer_days(days, window)


def test_the_compared_baseline_is_the_published_baseline(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    result = compare(
        days, _baseline_by_accession(days, scored), _window(scored), overlap_lags=5
    )
    assert result.baseline == evaluate_baseline(days, overlap_lags=5).ic


def test_an_empty_day_in_the_window_is_skipped(
    days: list[DayOfEvents], scored: dict[date, list[float]]
) -> None:
    agent = _baseline_by_accession(days, scored)
    empty = _day(DAYS[-1] + timedelta(days=1), issuers=0)
    assert len(empty) == 0
    window = (min(scored), empty.day)
    assert compare([*days, empty], agent, window, overlap_lags=5) == compare(
        days, agent, _window(scored), overlap_lags=5
    )


def test_a_day_of_exactly_ten_issuers_is_ranked() -> None:
    ten = _day(DAYS[0], issuers=10)
    daily, _ = daily_ic([ten], {ten.day: [float(i % 2) for i in ten.issuers]})
    assert list(daily) == [ten.day]
