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
