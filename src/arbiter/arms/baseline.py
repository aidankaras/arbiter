"""The conventional arm: logistic regression on filing characteristics.

This is the control the other three arms have to beat. It is deliberately
unremarkable — a linear model on a handful of features drawn from the filing —
because its job is to establish how much of any measured skill is available
without language understanding at all. An elaborate baseline would confound
that: if a gradient-boosted ensemble beat the agent, the finding would be about
model capacity rather than about reading the filing.

Probabilities rather than point estimates, because the comparison is scored on
calibration as well as ranking, and a regression on returns gives no calibrated
probability to score. The target is the sign of the abnormal return, which is
the coarsest question the system can be asked and therefore the one where a
weak signal is most likely to be visible at all.

Features are standardised before fitting. Their scales differ by orders of
magnitude — a log dollar value against a zero-or-one indicator — and without
scaling the regularisation penalty would fall almost entirely on the indicators,
shrinking them toward zero for a reason unrelated to how informative they are.
"""

# scikit-learn ships no type information, so strict checking reports every call
# into it as unknown. The suppressions are scoped to this module, the only place
# the estimator is touched, and the functions below declare their own types so
# the package's public surface stays fully checked.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from arbiter.arms.features import FEATURE_NAMES, as_matrix

#: Fixed so a published result can be reproduced exactly. Logistic regression
#: under this solver is deterministic given its data, but the seed is recorded
#: rather than assumed, because a later change of solver would not announce
#: itself.
RANDOM_SEED = 20260918

#: Inverse regularisation strength. Left at the library default deliberately:
#: tuning it against the evaluation data would report the best of many attempts
#: as though it were a single honest measurement.
REGULARISATION = 1.0


class DegenerateTrainingSetError(ValueError):
    """Raised when the training data cannot support a fitted model.

    A single outcome class means there is nothing to separate, and a model
    fitted on it would return one constant probability that scores as perfectly
    confident and is worthless.
    """


@dataclass(frozen=True)
class FittedBaseline:
    """A fitted model, with what it was fitted on recorded alongside it."""

    pipeline: Pipeline
    base_rate: float
    events: int
    seed: int = RANDOM_SEED

    def coefficients(self) -> dict[str, float]:
        """Return each feature's fitted weight, on the standardised scale.

        Comparable across features because the inputs were standardised, so the
        magnitudes say which characteristics the model actually leans on.
        """
        model = self.pipeline.named_steps["model"]
        weights: Sequence[float] = model.coef_[0]
        return dict(zip(FEATURE_NAMES, (float(weight) for weight in weights), strict=True))


def fit_baseline(
    features: Sequence[Mapping[str, float]], outcomes: Sequence[int]
) -> FittedBaseline:
    """Fit the baseline arm on features and realised outcome signs.

    Raises:
        DegenerateTrainingSetError: no events, mismatched lengths, or a single
            outcome class.
    """
    if len(features) != len(outcomes):
        msg = f"{len(features)} feature rows against {len(outcomes)} outcomes"
        raise DegenerateTrainingSetError(msg)
    if not features:
        msg = "no training events"
        raise DegenerateTrainingSetError(msg)
    if len(set(outcomes)) < 2:
        msg = (
            "every training event has the same outcome, so there is nothing to "
            "separate; a model fitted here would return one constant probability"
        )
        raise DegenerateTrainingSetError(msg)

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=REGULARISATION,
                    max_iter=1000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    pipeline.fit(np.asarray(as_matrix(features)), np.asarray(outcomes))

    return FittedBaseline(
        pipeline=pipeline,
        base_rate=float(np.mean(outcomes)),
        events=len(features),
    )


def forecast(model: FittedBaseline, features: Sequence[Mapping[str, float]]) -> list[float]:
    """Return the probability of a positive abnormal return for each event."""
    if not features:
        return []

    probabilities = model.pipeline.predict_proba(np.asarray(as_matrix(features)))
    return [float(row[1]) for row in probabilities]
