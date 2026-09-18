"""The published numbers, tested against cases whose answer is known in advance.

These functions decide what the project claims, so each is checked against
inputs with an analytically known result rather than against whatever the
implementation happens to return. The failure mode being guarded is a metric
that looks plausible and is silently vacuous — one that reports skill for a
forecast that has none.
"""

import math

import pytest

from arbiter.evaluation.metrics import (
    InsufficientObservationsError,
    brier_score,
    brier_skill_score,
    calibration_curve,
    information_coefficient,
    summarise_ic,
)


def test_a_perfect_ranking_scores_one():
    assert information_coefficient([1, 2, 3, 4], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)


def test_an_exactly_reversed_ranking_scores_minus_one():
    assert information_coefficient([1, 2, 3, 4], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(
        -1.0
    )


def test_only_the_ordering_counts_not_the_magnitudes():
    """Returns are heavy-tailed; a linear measure would follow the outliers."""
    modest = information_coefficient([1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0])
    extreme = information_coefficient([1, 2, 3, 4], [1.0, 2.0, 3.0, 4000.0])

    assert modest == pytest.approx(extreme)


def test_a_forecast_that_says_the_same_thing_about_everything_is_undefined():
    """No ranking exists to score, which is not the same as ranking badly."""
    assert math.isnan(information_coefficient([0.5, 0.5, 0.5], [1.0, 2.0, 3.0]))


def test_outcomes_that_never_vary_are_undefined_too():
    assert math.isnan(information_coefficient([1.0, 2.0, 3.0], [7.0, 7.0, 7.0]))


def test_mismatched_lengths_are_refused():
    with pytest.raises(InsufficientObservationsError):
        information_coefficient([1.0, 2.0], [1.0])


def test_a_single_observation_cannot_produce_a_correlation():
    with pytest.raises(InsufficientObservationsError, match="at least two"):
        information_coefficient([1.0], [1.0])


def test_the_average_is_taken_across_days_not_across_events():
    """Events on one day react to the same market and are not independent."""
    summary = summarise_ic({"a": 0.10, "b": 0.20, "c": 0.30}, observations=900)

    assert summary.mean == pytest.approx(0.20)
    assert summary.days == 3
    assert summary.observations == 900


def test_the_standard_error_reflects_the_spread_between_days():
    steady = summarise_ic({"a": 0.20, "b": 0.20, "c": 0.20}, observations=900)
    volatile = summarise_ic({"a": -0.30, "b": 0.20, "c": 0.70}, observations=900)

    assert steady.standard_error == pytest.approx(0.0)
    assert volatile.standard_error > 0.2


def test_a_consistent_edge_is_significant_and_a_noisy_one_is_not():
    consistent = summarise_ic({str(i): 0.05 + 0.001 * i for i in range(20)}, observations=5000)
    noisy = summarise_ic({str(i): 0.4 * (-1) ** i for i in range(20)}, observations=5000)

    assert consistent.is_significant
    assert not noisy.is_significant


def test_days_without_a_defined_coefficient_are_excluded_not_counted_as_zero():
    """Counting them as zero would drag any real edge toward nothing."""
    summary = summarise_ic({"a": 0.2, "b": math.nan, "c": 0.2}, observations=100)

    assert summary.days == 2
    assert summary.mean == pytest.approx(0.2)


def test_days_that_agree_exactly_are_unboundedly_far_from_zero():
    """No spread to divide by is a limit, not a failure to measure."""
    summary = summarise_ic({"a": 0.3, "b": 0.3, "c": 0.3}, observations=100)

    assert summary.t_statistic == math.inf
    assert summary.is_significant


def test_days_that_agree_on_exactly_zero_are_undefined():
    """The one case with genuinely nothing to report."""
    summary = summarise_ic({"a": 0.0, "b": 0.0}, observations=100)

    assert math.isnan(summary.t_statistic)
    assert not summary.is_significant


def test_a_single_day_cannot_be_reported_as_an_average():
    with pytest.raises(InsufficientObservationsError, match="at least two days"):
        summarise_ic({"a": 0.5}, observations=100)


def test_a_perfect_probabilistic_forecast_scores_zero():
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


def test_a_confidently_wrong_forecast_scores_one():
    assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)


def test_forecasting_the_base_rate_has_no_skill_by_construction():
    """The property that makes the skill score readable at all."""
    outcomes = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    base_rate = sum(outcomes) / len(outcomes)

    assert brier_skill_score([base_rate] * len(outcomes), outcomes) == pytest.approx(0.0)


def test_a_forecast_worse_than_the_base_rate_scores_negative():
    outcomes = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    assert brier_skill_score([0.9] * len(outcomes), outcomes) < 0


def test_a_skilful_forecast_scores_positive():
    outcomes = [1, 1, 1, 1, 0, 0, 0, 0]
    confident = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]

    assert brier_skill_score(confident, outcomes) > 0.8


def test_a_perfectly_calibrated_forecast_lies_on_the_diagonal():
    """Thirty percent of the events forecast at thirty percent actually happen."""
    probabilities = [0.3] * 10 + [0.8] * 10
    outcomes = [*[1, 1, 1, 0, 0, 0, 0, 0, 0, 0], *[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]

    curve = calibration_curve(probabilities, outcomes, bins=10)

    for band in curve:
        assert band.forecast == pytest.approx(band.realised, abs=0.01)


def test_an_overconfident_forecast_sits_below_the_diagonal():
    probabilities = [0.9] * 10
    outcomes = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    curve = calibration_curve(probabilities, outcomes, bins=10)

    assert curve[0].forecast > curve[0].realised


def test_bands_containing_no_forecast_are_omitted_not_reported_as_wrong():
    """No forecast was made there, which is not the same as being wrong there."""
    curve = calibration_curve([0.95] * 5, [1, 1, 1, 1, 0], bins=10)

    assert len(curve) == 1
    assert curve[0].count == 5


def test_a_forecast_of_exact_certainty_is_scored_rather_than_dropped():
    curve = calibration_curve([1.0, 1.0], [1, 0], bins=10)

    assert sum(band.count for band in curve) == 2


def test_every_forecast_lands_in_exactly_one_band():
    """Overlapping edges would double-count and leave the curve unreadable."""
    probabilities = [index / 50 for index in range(51)]
    outcomes = [index % 2 for index in range(51)]

    curve = calibration_curve(probabilities, outcomes, bins=10)

    assert sum(band.count for band in curve) == 51


def test_a_curve_needs_at_least_two_bands():
    with pytest.raises(ValueError, match="at least two bins"):
        calibration_curve([0.5], [1], bins=1)


def test_an_empty_evaluation_is_refused_rather_than_scored():
    with pytest.raises(InsufficientObservationsError):
        calibration_curve([], [], bins=10)
