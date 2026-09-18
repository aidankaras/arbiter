"""Evaluating an arm on data whose answer is known by construction.

The defect worth guarding against is an evaluation that reports skill for a
forecast that has none, so the central tests build days where the answer is
fixed in advance: a signal that persists across the split must be found, and
outcomes independent of the features must not be.

The other property tested here is the split itself. Scoring an arm on days it
was fitted on would report memory as skill, and no downstream check would catch
it.
"""

import random
from datetime import date

import pytest

from arbiter.arms.evaluate import DayOfEvents, evaluate_baseline, split_days
from arbiter.arms.features import FEATURE_NAMES
from arbiter.evaluation.metrics import InsufficientObservationsError

DAYS = [date(2026, 6, day) for day in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12)]


def _features(purchase: float) -> dict[str, float]:
    row = dict.fromkeys(FEATURE_NAMES, 0.0)
    row["is_purchase"] = purchase
    row["log_value_usd"] = 12.0
    return row


def _day(day: date, *, informative: bool, issuers: int = 40) -> DayOfEvents:
    """Build a day where purchases either do or do not precede positive returns."""
    generator = random.Random(day.toordinal())
    features: list[dict[str, float]] = []
    returns: list[float] = []

    # The signal's strength varies by day, as any real one does. Days that agree
    # exactly would leave no spread across days, and the summary would be
    # reporting a limit rather than an estimate.
    strength = 0.004 + 0.004 * generator.random()

    for index in range(issuers):
        purchase = float(index % 2)
        features.append(_features(purchase))
        if informative:
            returns.append(strength * (1 if purchase else -1) + generator.gauss(0, 0.004))
        else:
            returns.append(generator.gauss(0, 0.01))

    return DayOfEvents(
        day=day,
        features=features,
        outcomes=[int(value > 0) for value in returns],
        returns=returns,
        issuers=list(range(issuers)),
    )


def test_a_signal_persisting_across_the_split_is_found():
    days = [_day(day, informative=True) for day in DAYS]

    evaluation = evaluate_baseline(days)

    assert evaluation.ic.mean > 0.5
    assert evaluation.ic.is_significant


def test_outcomes_independent_of_the_features_yield_no_measurable_skill():
    """The test that matters: the evaluation must not manufacture a result."""
    days = [_day(day, informative=False) for day in DAYS]

    evaluation = evaluate_baseline(days)

    assert not evaluation.ic.is_significant


def test_scored_days_all_fall_after_fitted_days():
    """A random split would score the arm partly on days it had already seen."""
    days = [_day(day, informative=True) for day in DAYS]

    evaluation = evaluate_baseline(days)

    assert max(evaluation.train_days) < min(evaluation.test_days)


def test_no_day_is_both_fitted_on_and_scored():
    days = [_day(day, informative=True) for day in DAYS]

    evaluation = evaluate_baseline(days)

    assert not set(evaluation.train_days) & set(evaluation.test_days)


def test_several_filings_for_one_issuer_count_as_one_observation():
    """Insiders at one company on one day resolve to a single outcome."""
    duplicated = DayOfEvents(
        day=date(2026, 6, 15),
        features=[_features(1.0)] * 30,
        outcomes=[1] * 30,
        returns=[0.02] * 30,
        issuers=[1] * 10 + [2] * 10 + [3] * 10,
    )
    days = [_day(day, informative=True) for day in DAYS[:8]] + [duplicated]

    evaluation = evaluate_baseline(days)

    assert evaluation.test_events > evaluation.test_issuer_days
    assert evaluation.test_issuer_days <= sum(
        len(set(day.issuers)) for day in days if day.day in evaluation.test_days
    )


def test_a_day_with_too_few_issuers_is_left_out_of_the_average():
    """A rank correlation over a handful of issuers is noise, not a measurement."""
    thin = _day(date(2026, 6, 15), informative=True, issuers=4)
    days = [_day(day, informative=True) for day in DAYS] + [thin]

    evaluation = evaluate_baseline(days)

    assert thin.day not in evaluation.daily_ic


def test_the_split_reports_what_it_used():
    days = [_day(day, informative=True) for day in DAYS]

    evaluation = evaluate_baseline(days)

    assert len(evaluation.train_days) + len(evaluation.test_days) == len(DAYS)
    assert evaluation.train_events > 0


def test_days_are_divided_chronologically_whatever_order_they_arrive_in():
    shuffled = [DAYS[3], DAYS[0], DAYS[7], DAYS[1]]

    train, test = split_days(shuffled, train_fraction=0.5)

    assert train == sorted(train)
    assert max(train) < min(test)


def test_too_few_days_to_score_is_refused_rather_than_reported():
    """Two scored days is the minimum from which any spread can be estimated."""
    with pytest.raises(InsufficientObservationsError, match="at least 1 and 2"):
        split_days([date(2026, 6, 1), date(2026, 6, 2)], train_fraction=0.9)


def test_an_evaluation_needs_days_on_both_sides_of_the_split():
    with pytest.raises(InsufficientObservationsError):
        evaluate_baseline([_day(DAYS[0], informative=True)])


def test_the_reported_sample_counts_only_days_that_contributed_a_coefficient():
    """The report prints days-excluded beside observations, inviting a division.

    A day too thin to rank contributes no coefficient, so counting its issuers
    toward the reported sample describes an estimate resting on more data than
    it does.
    """
    thin = _day(date(2026, 6, 15), informative=True, issuers=4)
    days = [_day(day, informative=True) for day in DAYS] + [thin]

    evaluation = evaluate_baseline(days)

    assert thin.day not in evaluation.daily_ic
    assert evaluation.test_issuer_days == sum(
        len(set(by_day.issuers)) for by_day in days if by_day.day in evaluation.daily_ic
    )
