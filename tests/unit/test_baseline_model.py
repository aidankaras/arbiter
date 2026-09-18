"""The baseline arm, checked on data whose answer is known by construction.

A control has to be trustworthy before anything can be compared against it, so
these tests establish that it learns a signal that is present, declines to
invent one that is not, and refuses inputs it cannot honestly fit.
"""

import random

import pytest

from arbiter.arms.baseline import (
    RANDOM_SEED,
    DegenerateTrainingSetError,
    fit_baseline,
    forecast,
)
from arbiter.arms.features import FEATURE_NAMES


def _features(purchase: float, value: float = 12.0, **overrides: float) -> dict[str, float]:
    row = dict.fromkeys(FEATURE_NAMES, 0.0)
    row["is_purchase"] = purchase
    row["is_sale"] = 1.0 - purchase
    row["log_value_usd"] = value
    row.update(overrides)
    return row


def _separable(count: int = 200) -> tuple[list[dict[str, float]], list[int]]:
    """Purchases precede positive outcomes, sales negative ones, without exception."""
    features = [_features(purchase=float(index % 2)) for index in range(count)]
    outcomes = [index % 2 for index in range(count)]
    return features, outcomes


def test_a_signal_that_is_present_is_learned():
    features, outcomes = _separable()

    model = fit_baseline(features, outcomes)
    probabilities = forecast(model, [_features(purchase=1.0), _features(purchase=0.0)])

    assert probabilities[0] > 0.9
    assert probabilities[1] < 0.1


def test_the_learned_weight_points_the_way_the_data_does():
    features, outcomes = _separable()

    weights = fit_baseline(features, outcomes).coefficients()

    assert weights["is_purchase"] > 0
    assert weights["is_sale"] < 0


def test_a_feature_carrying_no_information_earns_no_weight():
    """The property that keeps the control honest: it must not invent signal."""
    features, outcomes = _separable()
    for index, row in enumerate(features):
        row["insiders_trading_same_issuer"] = float(index % 7)

    weights = fit_baseline(features, outcomes).coefficients()

    assert abs(weights["insiders_trading_same_issuer"]) < abs(weights["is_purchase"]) / 5


def test_pure_noise_produces_forecasts_near_the_base_rate():
    """Given nothing to learn, the model must fall back on how often it happens.

    The outcomes are drawn independently of the features from a seeded
    generator. An arithmetic pattern is not usable here: a sequence such as
    `(7 * index + 3) % 2` reduces to `(index + 1) % 2`, which is perfectly
    anti-correlated with a feature alternating on the same period, and the
    model would be right to learn it.
    """
    generator = random.Random(RANDOM_SEED)
    features = [_features(purchase=float(index % 2)) for index in range(400)]
    outcomes = [generator.randint(0, 1) for _ in range(400)]

    model = fit_baseline(features, outcomes)
    probabilities = forecast(model, features)

    assert all(0.35 < probability < 0.65 for probability in probabilities)


def test_every_feature_is_weighted_so_none_is_silently_dropped():
    features, outcomes = _separable()

    weights = fit_baseline(features, outcomes).coefficients()

    assert set(weights) == set(FEATURE_NAMES)


def test_the_base_rate_and_sample_size_are_recorded_with_the_model():
    """A coefficient means nothing without knowing what it was fitted on."""
    features, outcomes = _separable(count=200)

    model = fit_baseline(features, outcomes)

    assert model.events == 200
    assert model.base_rate == pytest.approx(0.5)
    assert model.seed == RANDOM_SEED


def test_fitting_twice_on_the_same_data_gives_the_same_model():
    """A published number has to be reproducible from the recorded seed."""
    features, outcomes = _separable()

    first = fit_baseline(features, outcomes).coefficients()
    second = fit_baseline(features, outcomes).coefficients()

    assert first == second


def test_training_data_with_one_outcome_class_is_refused():
    """It would fit a constant probability that scores as perfect confidence."""
    features = [_features(purchase=1.0) for _ in range(50)]

    with pytest.raises(DegenerateTrainingSetError, match="nothing to"):
        fit_baseline(features, [1] * 50)


def test_no_training_data_is_refused():
    with pytest.raises(DegenerateTrainingSetError, match="no training events"):
        fit_baseline([], [])


def test_mismatched_features_and_outcomes_are_refused():
    features, outcomes = _separable(count=10)

    with pytest.raises(DegenerateTrainingSetError):
        fit_baseline(features, outcomes[:5])


def test_forecasting_nothing_returns_nothing():
    features, outcomes = _separable()

    assert forecast(fit_baseline(features, outcomes), []) == []


def test_forecasts_are_probabilities():
    features, outcomes = _separable()

    probabilities = forecast(fit_baseline(features, outcomes), features)

    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
