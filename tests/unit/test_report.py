"""The report is the claim, so what it says about a weak result is the test.

A rendered table invites the most flattering reading of itself, and the
temptation it creates is to let a null result look promising. These tests pin
the opposite behaviour: an estimate indistinguishable from zero has to say so
in words, and every figure needed to judge the estimate has to appear beside it.
"""

from dataclasses import replace
from datetime import date

import pytest

from arbiter.arms.evaluate import Evaluation
from arbiter.evaluation.metrics import CalibrationBin, ICSummary
from arbiter.evaluation.report import render_baseline_report

GENERATED = date(2026, 9, 18)


def _evaluation(mean: float, standard_error: float, t: float) -> Evaluation:
    return Evaluation(
        ic=ICSummary(
            mean=mean,
            standard_error=standard_error,
            t_statistic=t,
            days=12,
            observations=3400,
        ),
        daily_ic={date(2026, 8, 3): 0.02, date(2026, 8, 4): -0.01, date(2026, 8, 5): 0.01},
        calibration=[
            CalibrationBin(lower=0.4, upper=0.5, forecast=0.45, realised=0.44, count=900),
            CalibrationBin(lower=0.5, upper=0.6, forecast=0.55, realised=0.57, count=700),
        ],
        brier_skill=0.0012,
        base_rate=0.486,
        train_days=[date(2026, 3, 2), date(2026, 6, 1)],
        test_days=[date(2026, 6, 8), date(2026, 8, 11)],
        train_events=9000,
        test_events=6000,
        test_issuer_days=3400,
        coefficients={"is_purchase": 0.21, "is_10b5_1": -0.08, "is_sale": -0.19},
    )


def test_an_estimate_indistinguishable_from_zero_says_so_in_words():
    """The number a reader would otherwise round up into a finding."""
    rendered = render_baseline_report(_evaluation(0.004, 0.011, 0.36), GENERATED)

    assert "No measurable skill" in rendered


def test_a_null_result_is_not_reported_as_proof_that_no_signal_exists():
    """Absence of evidence stated as evidence of absence would be the wrong claim."""
    rendered = render_baseline_report(_evaluation(0.004, 0.011, 0.36), GENERATED)

    assert "not evidence that no signal is there" in rendered


def test_a_significant_result_is_still_qualified():
    rendered = render_baseline_report(_evaluation(0.06, 0.02, 3.0), GENERATED)

    assert "information coefficient of +0.0600" in rendered
    assert "weak bar" in rendered
    assert "should be expected to shrink" in rendered


def test_an_undefined_estimate_draws_no_conclusion():
    rendered = render_baseline_report(_evaluation(float("nan"), 0.0, float("nan")), GENERATED)

    assert "No conclusion" in rendered


@pytest.mark.parametrize(
    "expected",
    [
        "12 of 2",  # days contributing a rank correlation, against days scored
        "3,400",  # issuer-days the estimate rests on
        "48.6%",  # base rate
        "2026-06-08 to 2026-08-11",  # scoring period
        "2026-03-02 to 2026-06-01",  # fitting period
    ],
)
def test_the_sample_behind_the_estimate_is_stated(expected: str):
    """An information coefficient without its sample is not a finding."""
    rendered = render_baseline_report(_evaluation(0.004, 0.011, 0.36), GENERATED)

    assert expected in rendered


def test_days_scored_and_days_that_could_be_ranked_are_reported_separately():
    """A day with too few issuers to rank is still a day the arm was scored on.

    Reporting only one of the two counts would leave the verdict's day count
    contradicting the table's.
    """
    rendered = render_baseline_report(_evaluation(0.004, 0.011, 0.36), GENERATED)

    assert "Days contributing a rank correlation" in rendered


def test_the_chronological_split_is_stated_so_a_reader_can_check_it():
    rendered = render_baseline_report(_evaluation(0.004, 0.011, 0.36), GENERATED)

    assert "every scored day falls after every fitted day" in rendered


def test_calibration_bands_are_rendered_with_their_counts():
    rendered = render_baseline_report(_evaluation(0.004, 0.011, 0.36), GENERATED)

    assert "0.4 to 0.5" in rendered
    assert "900" in rendered


def test_weights_are_ordered_by_magnitude_not_by_name():
    """What the model leaned on should be readable without scanning the table."""
    rendered = render_baseline_report(_evaluation(0.004, 0.011, 0.36), GENERATED)
    weights = rendered.split("## Fitted weights")[1]

    assert weights.index("is_purchase") < weights.index("is_sale") < weights.index("is_10b5_1")


def test_the_limits_of_the_measurement_are_stated():
    """Ranking is not profit, and the report must not let that be assumed."""
    rendered = render_baseline_report(_evaluation(0.06, 0.02, 3.0), GENERATED)

    assert "measures ranking, not profit" in rendered
    assert "transaction costs" in rendered


def test_the_same_evaluation_renders_identically():
    """A committed report must diff cleanly when only the data has changed."""
    evaluation = _evaluation(0.004, 0.011, 0.36)

    assert render_baseline_report(evaluation, GENERATED) == render_baseline_report(
        evaluation, GENERATED
    )


def test_a_forecast_worse_than_the_base_rate_is_said_so_in_the_verdict():
    """Discrimination and calibration fail independently.

    A reader who stops at "no measurable skill" would otherwise take a forecast
    that is actively worse than a constant as merely uninformative. A real run
    produced an information coefficient indistinguishable from zero alongside a
    Brier skill of -0.22.
    """
    evaluation = _evaluation(0.004, 0.011, 0.36)
    worse = replace(evaluation, brier_skill=-0.2224)

    rendered = render_baseline_report(worse, GENERATED)
    verdict = rendered.split("## What this rests on")[0]

    assert "less useful than forecasting the base rate" in verdict
    assert "-0.2224" in verdict


def test_a_positive_brier_skill_is_still_qualified_in_the_verdict():
    better = replace(_evaluation(0.004, 0.011, 0.36), brier_skill=0.0130)

    verdict = render_baseline_report(better, GENERATED).split("## What this rests on")[0]

    assert "weaker claim than discrimination" in verdict


def test_the_direction_of_a_miscalibration_is_measured_not_assumed():
    """A negative Brier skill carries no direction; taking one from it was wrong.

    The first published report had forecasts running *low* — a band forecasting
    0.273 realised 0.831 — while the verdict asserted they ran high.
    """
    low = replace(
        _evaluation(0.004, 0.011, 0.36),
        brier_skill=-0.04,
        calibration=[
            CalibrationBin(lower=0.2, upper=0.3, forecast=0.273, realised=0.831, count=65),
            CalibrationBin(lower=0.4, upper=0.5, forecast=0.457, realised=0.452, count=1003),
        ],
    )

    verdict = render_baseline_report(low, GENERATED).split("## What this rests on")[0]

    assert "run low" in verdict
    assert "run high" not in verdict


def test_forecasts_that_run_high_are_described_as_such():
    high = replace(
        _evaluation(0.004, 0.011, 0.36),
        brier_skill=-0.04,
        calibration=[
            CalibrationBin(lower=0.7, upper=0.8, forecast=0.750, realised=0.300, count=400),
        ],
    )

    verdict = render_baseline_report(high, GENERATED).split("## What this rests on")[0]

    assert "run high" in verdict


def test_exactly_zero_skill_is_not_called_better_than_the_base_rate():
    """It falls in neither branch by sign, and was previously read as positive."""
    neutral = replace(_evaluation(0.004, 0.011, 0.36), brier_skill=0.0)

    verdict = render_baseline_report(neutral, GENERATED).split("## What this rests on")[0]

    assert "worth exactly what forecasting the base rate" in verdict
    assert "worth slightly more" not in verdict


def test_bands_agreeing_on_a_small_miss_are_not_called_contradictory():
    """A small aggregate can mean agreement, so the sign of each band is read.

    Two bands each running +0.004 high average to 0.004 — under the threshold
    for naming a shift, but they do not disagree, and saying they did would
    repeat the error this function was written to fix, one level down.
    """
    agreeing = replace(
        _evaluation(0.004, 0.011, 0.36),
        brier_skill=-0.01,
        calibration=[
            CalibrationBin(lower=0.4, upper=0.5, forecast=0.454, realised=0.450, count=500),
            CalibrationBin(lower=0.5, upper=0.6, forecast=0.554, realised=0.550, count=500),
        ],
    )

    verdict = render_baseline_report(agreeing, GENERATED).split("## What this rests on")[0]

    assert "same direction in every populated band" in verdict
    assert "disagree in direction" not in verdict


def test_bands_that_genuinely_disagree_are_reported_as_dispersion():
    mixed = replace(
        _evaluation(0.004, 0.011, 0.36),
        brier_skill=-0.01,
        calibration=[
            CalibrationBin(lower=0.2, upper=0.3, forecast=0.25, realised=0.60, count=500),
            CalibrationBin(lower=0.7, upper=0.8, forecast=0.75, realised=0.40, count=500),
        ],
    )

    verdict = render_baseline_report(mixed, GENERATED).split("## What this rests on")[0]

    assert "disagree in direction" in verdict
    assert "dispersion rather than a shift" in verdict
