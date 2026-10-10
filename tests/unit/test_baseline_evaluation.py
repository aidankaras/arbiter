"""Evaluating an arm on data whose answer is known by construction.

The defect worth guarding against is an evaluation that reports skill for a
forecast that has none, so the central tests build days where the answer is
fixed in advance: a signal that persists across the walk must be found, and
outcomes independent of the features must not be.

The other property tested here is the walk itself. Scoring an arm on outcomes
it was fitted on, or fitting on outcomes that had not yet closed when a forecast
was made, would report memory or foresight as skill, and no downstream check
would catch it. Those tests state the rule from the requirement and check it
against a calendar written out by hand, not against the implementation's own
arithmetic.
"""

import math
import random
from dataclasses import replace
from datetime import date

import pytest

from arbiter.arms.evaluate import (
    DayOfEvents,
    _rose,
    evaluate_baseline,
    forecast_walk_forward,
)
from arbiter.arms.features import FEATURE_NAMES
from arbiter.evaluation.metrics import InsufficientObservationsError, brier_skill_score

#: Twenty consecutive sessions. 19 June 2026 is a market holiday, so the run
#: skips it; position in this list is therefore the session count, which lets
#: a test state "outcomes close N sessions later" without a calendar lookup.
DAYS = [
    date(2026, 6, day)
    for day in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 22, 23, 24, 25, 26, 29)
]
#: A session after the run, for fixtures that add one more day.
AFTER = date(2026, 6, 30)


def _features(purchase: float) -> dict[str, float]:
    row = dict.fromkeys(FEATURE_NAMES, 0.0)
    row["is_purchase"] = purchase
    row["log_value_usd"] = 12.0
    return row


def _day(day: date, *, informative: bool, issuers: int = 40, bias: float = 0.0) -> DayOfEvents:
    """Build a day where purchases either do or do not precede positive returns.

    `bias` shifts every return, moving the day's up-rate without touching how
    well the features rank it.
    """
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
            returns.append(
                strength * (1 if purchase else -1) + generator.gauss(0, 0.004) + bias
            )
        else:
            returns.append(generator.gauss(0, 0.01) + bias)

    return DayOfEvents(
        day=day,
        features=features,
        outcomes=[int(value > 0) for value in returns],
        returns=returns,
        issuers=list(range(issuers)),
    )


def _flipped(day: DayOfEvents) -> DayOfEvents:
    """The same day's events with every outcome reversed."""
    return replace(
        day,
        outcomes=[1 - outcome for outcome in day.outcomes],
        returns=[-value for value in day.returns],
    )


def _informative_days() -> list[DayOfEvents]:
    return [_day(day, informative=True) for day in DAYS]


def test_a_signal_persisting_across_the_walk_is_found():
    evaluation = evaluate_baseline(_informative_days(), embargo_sessions=0)

    assert evaluation.ic.mean > 0.5
    assert evaluation.ic.is_significant


def test_outcomes_independent_of_the_features_yield_no_measurable_skill():
    """The test that matters: the evaluation must not manufacture a result."""
    days = [_day(day, informative=False) for day in DAYS]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert not evaluation.ic.is_significant


@pytest.mark.parametrize("embargo", [0, 2, 3])
def test_no_fold_is_fitted_on_a_day_whose_outcomes_were_still_open(embargo: int):
    """The look-ahead property, checked against a hand-counted session calendar.

    A fitted day's outcomes close `embargo` sessions after it. A fold may fit on
    that day only if the close came before the fold's first scored day, and
    never on a day at or after any day it scores. DAYS is a run of consecutive
    sessions, so the close is `embargo` places along the list.
    """
    evaluation = evaluate_baseline(_informative_days(), embargo_sessions=embargo, refit_every=4)

    assert len(evaluation.folds) > 1
    for fold in evaluation.folds:
        first = DAYS.index(fold.scored_days[0])
        for fitted in fold.fit_days:
            assert DAYS.index(fitted) + embargo < first, (fitted, fold.scored_days[0])
            assert fitted < min(fold.scored_days)
        # The converse, so a fold fitted on nothing cannot pass: every day whose
        # outcomes had closed is used, not just a safe subset.
        closed = [day for position, day in enumerate(DAYS) if position + embargo < first]
        assert fold.fit_days == closed


def test_an_embargo_actually_removes_days_from_the_fit():
    """Premise of the property above: with an embargo there is something to exclude."""
    plain = evaluate_baseline(_informative_days(), embargo_sessions=0, refit_every=4)
    embargoed = evaluate_baseline(_informative_days(), embargo_sessions=3, refit_every=4)

    assert all(
        len(held.fit_days) == len(open_.fit_days) - 3
        for open_, held in zip(plain.folds, embargoed.folds, strict=True)
    )


def test_a_block_is_forecast_without_sight_of_its_own_outcomes():
    """Changing the outcomes inside a block must not move that block's forecasts.

    The oracle for look-ahead that does not depend on how folds are sliced: if
    any of block k's returns reached the fit that forecast block k, reversing
    them would change what was forecast.
    """
    days = _informative_days()
    baseline = forecast_walk_forward(days, embargo_sessions=0, refit_every=4)
    block = baseline[1][0].scored_days

    perturbed = [_flipped(day) if day.day in block else day for day in days]
    after = forecast_walk_forward(perturbed, embargo_sessions=0, refit_every=4)

    assert after[1][2] == baseline[1][2]
    # Premise: the perturbation is not inert. The next block is fitted on this
    # one, so it must have moved; otherwise equality above proves nothing.
    assert after[2][2] != baseline[2][2]


def test_a_realised_earlier_day_does_move_a_blocks_forecasts():
    days = _informative_days()
    baseline = forecast_walk_forward(days, embargo_sessions=0, refit_every=4)
    earlier = baseline[1][0].fit_days[-1]

    perturbed = [_flipped(day) if day.day == earlier else day for day in days]
    after = forecast_walk_forward(perturbed, embargo_sessions=0, refit_every=4)

    assert after[1][2] != baseline[1][2]


def test_an_unrealised_earlier_day_does_not_move_a_blocks_forecasts():
    """With a two-session embargo the day just before a block is still open."""
    days = _informative_days()
    baseline = forecast_walk_forward(days, embargo_sessions=2, refit_every=4)
    block_start = DAYS.index(baseline[1][0].scored_days[0])
    open_day = DAYS[block_start - 1]
    assert open_day not in baseline[1][0].fit_days

    perturbed = [_flipped(day) if day.day == open_day else day for day in days]
    after = forecast_walk_forward(perturbed, embargo_sessions=2, refit_every=4)

    assert after[1][2] == baseline[1][2]
    # Premise: that day is fitted on once it has closed, so it matters later.
    assert open_day in baseline[2][0].fit_days
    assert after[2][2] != baseline[2][2]


@pytest.mark.parametrize("refit_every", [1, 3, 5, 14, 100])
def test_every_populated_day_after_the_initial_window_is_scored_exactly_once(
    refit_every: int,
):
    """Coverage: 20 populated days at 0.3 leave six initial and fourteen scored."""
    nothing = DayOfEvents(
        day=date(2026, 6, 19), features=[], outcomes=[], returns=[], issuers=[]
    )
    days = [*_informative_days(), nothing]

    evaluation = evaluate_baseline(days, embargo_sessions=0, refit_every=refit_every)

    assert evaluation.initial_days == DAYS[:6]
    assert evaluation.test_days == DAYS[6:]
    assert not set(evaluation.initial_days) & set(evaluation.test_days)
    assert [day for fold in evaluation.folds for day in fold.scored_days] == DAYS[6:]
    assert len(evaluation.folds) == math.ceil(14 / refit_every)
    assert all(len(fold.scored_days) <= refit_every for fold in evaluation.folds)
    # Blocks are consecutive: only the last may be short.
    assert all(len(fold.scored_days) == refit_every for fold in evaluation.folds[:-1])


def test_training_expands_to_include_earlier_scored_blocks():
    evaluation = evaluate_baseline(_informative_days(), embargo_sessions=0, refit_every=4)

    sizes = [len(fold.fit_days) for fold in evaluation.folds]
    assert sizes == [6, 10, 14, 18]
    assert set(evaluation.folds[1].fit_days) >= set(evaluation.folds[0].scored_days)


def test_several_filings_for_one_issuer_count_as_one_observation():
    """Insiders at one company on one day resolve to a single outcome."""
    duplicated = DayOfEvents(
        day=AFTER,
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
    thin = _day(AFTER, informative=True, issuers=4)

    evaluation = evaluate_baseline([*_informative_days(), thin], embargo_sessions=0)

    assert thin.day not in evaluation.daily_ic


def test_the_evaluation_reports_the_last_folds_fit():
    evaluation = evaluate_baseline(_informative_days(), embargo_sessions=0, refit_every=4)

    assert evaluation.train_events == evaluation.folds[-1].train_events
    assert evaluation.train_events == 40 * len(evaluation.folds[-1].fit_days)


def test_days_are_walked_chronologically_whatever_order_they_arrive_in():
    ordered = evaluate_baseline(_informative_days(), embargo_sessions=0)
    shuffled = evaluate_baseline(list(reversed(_informative_days())), embargo_sessions=0)

    assert shuffled.initial_days == ordered.initial_days
    assert shuffled.test_days == sorted(shuffled.test_days)
    assert shuffled.ic.mean == pytest.approx(ordered.ic.mean)


def test_too_few_days_to_score_is_refused_rather_than_reported():
    """Two scored days is the minimum from which any spread can be estimated."""
    days = [_day(day, informative=True) for day in DAYS[:2]]

    with pytest.raises(InsufficientObservationsError, match="at least 1 and 2"):
        evaluate_baseline(days, embargo_sessions=0, initial_fraction=0.9)


def test_an_evaluation_needs_days_on_both_sides_of_the_walk():
    with pytest.raises(InsufficientObservationsError):
        evaluate_baseline([_day(DAYS[0], informative=True)], embargo_sessions=0)


def test_an_embargo_longer_than_the_initial_window_is_refused():
    """The first block would have no closed outcome to fit on."""
    with pytest.raises(InsufficientObservationsError, match="leaves nothing to fit on"):
        evaluate_baseline(_informative_days(), embargo_sessions=7)


def test_the_reported_sample_counts_only_days_that_contributed_a_coefficient():
    """The report prints days-excluded beside observations, inviting a division.

    A day too thin to rank contributes no coefficient, so counting its issuers
    toward the reported sample describes an estimate resting on more data than
    it does.
    """
    thin = _day(AFTER, informative=True, issuers=4)
    days = [*_informative_days(), thin]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert thin.day not in evaluation.daily_ic
    assert evaluation.test_issuer_days == sum(
        len(set(by_day.issuers)) for by_day in days if by_day.day in evaluation.daily_ic
    )


def test_a_filing_repeated_across_rows_weighs_no_more_than_one_issuer_day():
    """Fitting and scoring must count the same unit.

    Scoring credits each issuer once per day. If fitting counted rows, one Form 4
    carrying forty transactions would pull the weights forty times as hard as a
    single-line filing, and the model would be fitted to a different population
    from the one it is judged on.
    """
    days = _informative_days()
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
        evaluate_baseline(_informative_days(), embargo_sessions=0, initial_fraction=fraction)


@pytest.mark.parametrize("cadence", [0, -1])
def test_a_refit_cadence_below_one_is_refused(cadence: int):
    with pytest.raises(ValueError, match="at least 1"):
        evaluate_baseline(_informative_days(), embargo_sessions=0, refit_every=cadence)


def test_calibration_is_pooled_over_every_scored_issuer_day():
    """A day too thin to rank still has outcomes worth calibrating against.

    The pooled count is stated separately from the ranked sample, and the
    calibration table must sum to the count stated beside it.
    """
    thin = _day(AFTER, informative=True, issuers=4)

    evaluation = evaluate_baseline([*_informative_days(), thin], embargo_sessions=0)

    assert thin.day in evaluation.test_days
    assert thin.day not in evaluation.daily_ic
    assert evaluation.pooled_issuer_days == evaluation.test_issuer_days + 4
    assert sum(band.count for band in evaluation.calibration) == evaluation.pooled_issuer_days


def _drifting_days() -> list[DayOfEvents]:
    """Days whose up-rate falls from about 80% to about 20% across the run."""
    return [
        _day(day, informative=True, bias=0.012 - 0.024 * position / (len(DAYS) - 1))
        for position, day in enumerate(DAYS)
    ]


def test_brier_skill_is_measured_against_each_blocks_own_fitted_rate():
    """Skill is 1 - BS_model / BS_ref, with the reference chosen per forecast.

    The expectation is computed here from the definition: every scored
    issuer-day (all issuers are distinct, so one per row) is compared against the
    fitted up-rate of the fold that scored it.
    """
    days = _drifting_days()
    blocks = forecast_walk_forward(days, embargo_sessions=0, refit_every=7)
    by_day = {day.day: day for day in days}

    squared_model: list[float] = []
    squared_reference: list[float] = []
    for fold, _, forecasts in blocks:
        for scored in fold.scored_days:
            for probability, outcome in zip(
                forecasts[scored], by_day[scored].outcomes, strict=True
            ):
                squared_model.append((probability - outcome) ** 2)
                squared_reference.append((fold.fitted_base_rate - outcome) ** 2)
    expected = 1 - (sum(squared_model) / len(squared_model)) / (
        sum(squared_reference) / len(squared_reference)
    )

    evaluation = evaluate_baseline(days, embargo_sessions=0, refit_every=7)

    rates = [fold.fitted_base_rate for fold in evaluation.folds]
    assert len(rates) == 2
    assert max(rates) - min(rates) > 0.1, "the folds must disagree for this test to bite"
    assert evaluation.brier_skill == pytest.approx(expected)
    # Premise: a single global reference would have given a different answer.
    flat = brier_skill_score(
        [p for _, _, f in blocks for scored in f.values() for p in scored],
        [o for fold, _, _ in blocks for d in fold.scored_days for o in by_day[d].outcomes],
        reference_rate=evaluation.fitted_base_rate,
    )
    assert abs(flat - expected) > 1e-3


def test_the_reported_fitted_rate_is_the_issuer_day_weighted_mean_of_fold_rates():
    """Two folds of unequal size: a plain mean of the rates would be wrong."""
    days = _drifting_days()

    evaluation = evaluate_baseline(days, embargo_sessions=0, refit_every=10)

    sizes = [len(fold.scored_days) for fold in evaluation.folds]
    assert sizes == [10, 4]
    rates = [fold.fitted_base_rate for fold in evaluation.folds]
    expected = (rates[0] * 10 + rates[1] * 4) / 14
    assert evaluation.fitted_base_rate == pytest.approx(expected)
    assert evaluation.fitted_base_rate != pytest.approx(sum(rates) / 2)


def test_the_hindsight_reference_is_never_easier_than_the_fitted_one():
    """The scored period's own rate is the best constant in hindsight.

    Skill against it is therefore at or below skill against any other constants,
    such as the rates the refits had observed, whatever the forecasts are.
    """
    days = _drifting_days()

    evaluation = evaluate_baseline(days, embargo_sessions=0, refit_every=7)

    assert evaluation.fitted_base_rate != pytest.approx(evaluation.base_rate)
    assert evaluation.brier_skill_scored_rate < evaluation.brier_skill


def test_unlabelled_events_on_scored_days_are_counted_and_fitted_ones_are_not():
    days = [replace(day, unlabelled=3) for day in _informative_days()]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert evaluation.test_events_unlabelled == 3 * len(evaluation.test_days)


def test_each_folds_rate_counts_an_issuer_day_by_the_sign_of_its_mean_return():
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

    evaluation = evaluate_baseline(days, embargo_sessions=0, refit_every=4)

    by_day = {day.day: day for day in days}
    for fold in evaluation.folds:
        rose = sum(
            sum(1 for value in by_day[d].returns[:40] if value > 0) for d in fold.fit_days
        )
        assert fold.fitted_base_rate == pytest.approx(rose / (40 * len(fold.fit_days)))


def test_a_scored_day_on_which_nothing_resolved_still_reports_its_unlabelled_events():
    """The day most fully lost to exclusions is the one the count exists for."""
    empty = DayOfEvents(
        day=AFTER, features=[], outcomes=[], returns=[], issuers=[], unlabelled=5
    )
    before = replace(empty, day=date(2026, 5, 29), unlabelled=7)
    days = [before, *_informative_days(), empty]

    evaluation = evaluate_baseline(days, embargo_sessions=0)

    assert evaluation.test_events_unlabelled == 5


def test_an_issuer_day_with_no_abnormal_return_did_not_rise():
    """The outcome is whether the stock beat its benchmark; matching it is not a rise."""
    assert _rose([0.0, 0.01, -0.01]) == [0, 1, 0]
