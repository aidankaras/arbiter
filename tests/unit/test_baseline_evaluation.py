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
from dataclasses import replace
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

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert evaluation.ic.mean > 0.5
    assert evaluation.ic.is_significant


def test_outcomes_independent_of_the_features_yield_no_measurable_skill():
    """The test that matters: the evaluation must not manufacture a result."""
    days = [_day(day, informative=False) for day in DAYS]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert not evaluation.ic.is_significant


def test_scored_days_all_fall_after_fitted_days():
    """A random split would score the arm partly on days it had already seen."""
    days = [_day(day, informative=True) for day in DAYS]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert max(evaluation.train_days) < min(evaluation.test_days)


def test_no_day_is_both_fitted_on_and_scored():
    days = [_day(day, informative=True) for day in DAYS]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

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

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert evaluation.test_events > evaluation.test_issuer_days
    assert evaluation.test_issuer_days <= sum(
        len(set(day.issuers)) for day in days if day.day in evaluation.test_days
    )


def test_a_day_with_too_few_issuers_is_left_out_of_the_average():
    """A rank correlation over a handful of issuers is noise, not a measurement."""
    thin = _day(date(2026, 6, 15), informative=True, issuers=4)
    days = [_day(day, informative=True) for day in DAYS] + [thin]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert thin.day not in evaluation.daily_ic


def test_the_split_reports_what_it_used():
    days = [_day(day, informative=True) for day in DAYS]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert len(evaluation.train_days) + len(evaluation.test_days) == len(DAYS)
    assert evaluation.train_events > 0


def test_days_are_divided_chronologically_whatever_order_they_arrive_in():
    shuffled = [DAYS[3], DAYS[0], DAYS[7], DAYS[1]]

    train, test = split_days(shuffled, train_fraction=0.5, embargo_sessions=0)

    assert train == sorted(train)
    assert max(train) < min(test)


def test_too_few_days_to_score_is_refused_rather_than_reported():
    """Two scored days is the minimum from which any spread can be estimated."""
    with pytest.raises(InsufficientObservationsError, match="at least 1 and 2"):
        split_days([date(2026, 6, 1), date(2026, 6, 2)], train_fraction=0.9, embargo_sessions=0)


def test_an_evaluation_needs_days_on_both_sides_of_the_split():
    with pytest.raises(InsufficientObservationsError):
        evaluate_baseline([_day(DAYS[0], informative=True)], embargo_sessions=0)


def test_the_reported_sample_counts_only_days_that_contributed_a_coefficient():
    """The report prints days-excluded beside observations, inviting a division.

    A day too thin to rank contributes no coefficient, so counting its issuers
    toward the reported sample describes an estimate resting on more data than
    it does.
    """
    thin = _day(date(2026, 6, 15), informative=True, issuers=4)
    days = [_day(day, informative=True) for day in DAYS] + [thin]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert thin.day not in evaluation.daily_ic
    assert evaluation.test_issuer_days == sum(
        len(set(by_day.issuers)) for by_day in days if by_day.day in evaluation.daily_ic
    )


def test_no_scored_day_opens_before_every_fitted_outcome_has_closed():
    """Fitting on outcomes still open when a scored forecast is made leaks the future.

    Fitted on 1 to 5 June with outcomes that can stay open two sessions past the
    last fitted day, the earliest admissible scored day is the session after
    9 June. Scoring 8 and 9 June would credit the model with returns that shared
    their market move with outcomes it had already been fitted on.
    """
    train, test = split_days(DAYS, train_fraction=0.5, embargo_sessions=2)

    assert train == DAYS[:5]
    assert test == [date(2026, 6, 10), date(2026, 6, 11), date(2026, 6, 12)]


def test_an_embargo_that_leaves_too_few_scored_days_is_refused():
    with pytest.raises(InsufficientObservationsError, match="at least 1 and 2"):
        split_days(DAYS, train_fraction=0.5, embargo_sessions=4)


def test_the_evaluation_states_which_days_the_embargo_removed():
    days = [_day(day, informative=True) for day in DAYS]

    evaluation = evaluate_baseline(days, train_fraction=0.5, embargo_sessions=2)

    assert evaluation.embargoed_days == [date(2026, 6, 8), date(2026, 6, 9)]
    assert len(evaluation.train_days) + len(evaluation.embargoed_days) + len(
        evaluation.test_days
    ) == len(DAYS)


def test_a_filing_repeated_across_rows_weighs_no_more_than_one_issuer_day():
    """Fitting and scoring must count the same unit.

    Scoring credits each issuer once per day. If fitting counted rows, one Form 4
    carrying forty transactions would pull the weights forty times as hard as a
    single-line filing, and the model would be fitted to a different population
    from the one it is judged on.
    """
    days = [_day(day, informative=True) for day in DAYS]
    first = days[0]
    repeated = DayOfEvents(
        day=first.day,
        features=first.features + [first.features[0]] * 9,
        outcomes=first.outcomes + [first.outcomes[0]] * 9,
        returns=first.returns + [first.returns[0]] * 9,
        issuers=first.issuers + [first.issuers[0]] * 9,
    )

    plain = evaluate_baseline(days, embargo_sessions=0).coefficients
    padded = evaluate_baseline([repeated, *days[1:]], embargo_sessions=0).coefficients

    assert padded == pytest.approx(plain, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("fraction", [-0.2, 0.0, 1.0, 1.5])
def test_a_fraction_outside_the_open_unit_interval_is_refused(fraction: float):
    """A negative fraction slices from the end and silently fits on most days."""
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_days(DAYS, train_fraction=fraction, embargo_sessions=0)


def test_calibration_is_pooled_over_every_scored_issuer_day():
    """A day too thin to rank still has outcomes worth calibrating against.

    The pooled count is stated separately from the ranked sample, and the
    calibration table must sum to the count stated beside it.
    """
    thin = _day(date(2026, 6, 15), informative=True, issuers=4)
    days = [_day(day, informative=True) for day in DAYS] + [thin]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert thin.day in evaluation.test_days
    assert thin.day not in evaluation.daily_ic
    assert evaluation.pooled_issuer_days == evaluation.test_issuer_days + 4
    assert sum(band.count for band in evaluation.calibration) == evaluation.pooled_issuer_days


def test_the_hindsight_reference_is_never_easier_than_the_fitted_one():
    """The scored period's own rate is the best constant in hindsight.

    Skill against it is therefore strictly below skill against any other
    constant, such as the fitted period's rate, whatever the forecasts are.
    """
    days = [_day(day, informative=False) for day in DAYS]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert evaluation.fitted_base_rate != evaluation.base_rate
    assert evaluation.brier_skill_scored_rate < evaluation.brier_skill


def test_unlabelled_events_on_scored_days_are_counted_and_fitted_ones_are_not():
    days = [replace(_day(day, informative=True), unlabelled=3) for day in DAYS]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert evaluation.test_events_unlabelled == 3 * len(evaluation.test_days)


def test_the_fitted_rate_counts_an_issuer_day_by_the_sign_of_its_mean_return():
    """The reference must decide "rose" exactly as scoring does.

    Each issuer that rose files a second row with a smaller opposite return. The
    issuer-day still rose; averaging its rows' signs would count it as half.
    """
    days: list[DayOfEvents] = []
    for day in DAYS:
        base = _day(day, informative=True)
        extra = [-value / 5 for value in base.returns if value > 0]
        positives = [index for index, value in enumerate(base.returns) if value > 0]
        days.append(
            DayOfEvents(
                day=day,
                features=base.features + [base.features[index] for index in positives],
                outcomes=base.outcomes + [int(value > 0) for value in extra],
                returns=base.returns + extra,
                issuers=base.issuers + [base.issuers[index] for index in positives],
            )
        )

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    fitted = [day for day in days if day.day in evaluation.train_days]
    rose = sum(sum(1 for value in day.returns[:40] if value > 0) for day in fitted)
    assert evaluation.fitted_base_rate == pytest.approx(rose / (40 * len(fitted)))


def test_a_scored_day_on_which_nothing_resolved_still_reports_its_unlabelled_events():
    """The day most fully lost to exclusions is the one the count exists for."""
    empty = DayOfEvents(
        day=date(2026, 6, 19), features=[], outcomes=[], returns=[], issuers=[], unlabelled=5
    )
    before = replace(empty, day=date(2026, 5, 29), unlabelled=7)
    days = [before, *(_day(day, informative=True) for day in DAYS), empty]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert evaluation.test_events_unlabelled == 5
