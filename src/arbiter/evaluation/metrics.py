"""The numbers a forecast is judged by.

Two properties are measured, and they are independent. Discrimination asks
whether the forecast ranks outcomes correctly, and is reported as the
information coefficient — the rank correlation between forecast and realised
return. Calibration asks whether a stated probability means what it says, and
is reported as a reliability curve and a Brier score. A forecast can rank well
and be badly calibrated, or be perfectly calibrated and rank no better than
chance, and a system that acts on stated probabilities needs both.

The information coefficient is computed within a day and then averaged across
days, never pooled. Events filed on one day react to the same market, so
pooling them would count correlated observations as independent and overstate
significance — the resulting standard error would be a claim the data cannot
support.

A forecast is also credited once per issuer per day, not once per filing.
Several insiders at one company on one day produce several events that share a
single outcome, so counting each would inflate the sample with copies of one
observation.
"""

# SciPy ships no type information, so strict checking reports its statistics
# calls as unknown. The suppressions are scoped to this module, which is the
# only place it is touched, and every function below declares its own types so
# the package's public surface stays fully checked.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

#: Below this many observations a within-day rank correlation is noise, and
#: including it would add variance to the average while carrying no signal.
MIN_EVENTS_PER_DAY = 10


class InsufficientObservationsError(ValueError):
    """Raised when a statistic is requested from too little data to support it."""


def information_coefficient(forecasts: Sequence[float], outcomes: Sequence[float]) -> float:
    """Return the rank correlation between forecasts and realised outcomes.

    Rank correlation rather than linear: the decision this feeds is which events
    to act on, which depends on the ordering, and returns are heavy-tailed
    enough that a linear measure would be dominated by a few observations.

    Returns `nan` when either side has no variation — a forecast that says the
    same thing about everything has no ranking to score, which is a different
    statement from ranking badly.

    Raises:
        InsufficientObservationsError: fewer than two paired observations, or
            the two sequences differ in length.
    """
    if len(forecasts) != len(outcomes):
        msg = f"{len(forecasts)} forecasts against {len(outcomes)} outcomes"
        raise InsufficientObservationsError(msg)
    if len(forecasts) < 2:
        msg = f"a correlation needs at least two observations, got {len(forecasts)}"
        raise InsufficientObservationsError(msg)

    if len(set(forecasts)) == 1 or len(set(outcomes)) == 1:
        return math.nan

    correlation = stats.spearmanr(np.asarray(forecasts), np.asarray(outcomes)).statistic
    return float(correlation)


@dataclass(frozen=True)
class ICSummary:
    """A forecast's discrimination, averaged over days."""

    mean: float
    standard_error: float
    t_statistic: float
    days: int
    observations: int

    @property
    def is_significant(self) -> bool:
        """Whether the mean is two standard errors clear of zero.

        A conventional threshold, and a weak one: it is reported so that a
        summary cannot be read as a finding without the reader seeing how far
        from zero it actually is.
        """
        return abs(self.t_statistic) >= 2.0


def summarise_ic[Key](daily: Mapping[Key, float], observations: int) -> ICSummary:
    """Average per-day information coefficients and report the uncertainty.

    The standard error is taken across days, which is the level at which the
    observations are close to independent. Days whose coefficient is undefined
    are excluded rather than counted as zero, since no ranking existed to score.

    Raises:
        InsufficientObservationsError: fewer than two days carry a defined
            coefficient, so no spread can be estimated.
    """
    values = [value for value in daily.values() if not math.isnan(value)]
    if len(values) < 2:
        msg = f"an average across days needs at least two days, got {len(values)}"
        raise InsufficientObservationsError(msg)

    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    # The sample standard deviation, so a single day cannot read as certainty.
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))

    return ICSummary(
        mean=mean,
        standard_error=standard_error,
        t_statistic=_t_statistic(mean, standard_error),
        days=len(array),
        observations=observations,
    )


def _t_statistic(mean: float, standard_error: float) -> float:
    """Return the mean in units of its own standard error.

    Days that agree exactly leave no spread to divide by. That is a limit rather
    than a failure — a non-zero mean with no disagreement is unboundedly far
    from zero — so it is reported as infinite. Only a zero mean with no spread
    is genuinely undefined, and it is the one case that returns `nan`.

    Exact agreement does not occur in measured data; it arises in constructed
    cases, where reporting it as undefined would make a perfect signal read as
    no signal at all.
    """
    if standard_error > 0:
        return mean / standard_error
    if mean == 0:
        return math.nan
    return math.inf if mean > 0 else -math.inf


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Return the mean squared error of probabilistic forecasts.

    Lower is better. The reference point that matters is the score of always
    forecasting the base rate, which `brier_skill_score` expresses directly:
    a raw Brier score cannot be read as good or bad without it.

    Raises:
        InsufficientObservationsError: the sequences differ in length or are empty.
    """
    if len(probabilities) != len(outcomes) or not probabilities:
        msg = f"{len(probabilities)} probabilities against {len(outcomes)} outcomes"
        raise InsufficientObservationsError(msg)

    predicted = np.asarray(probabilities, dtype=float)
    realised = np.asarray(outcomes, dtype=float)
    return float(np.mean((predicted - realised) ** 2))


def brier_skill_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    """Return the Brier score's improvement over forecasting the base rate.

    Zero means the forecast is worth exactly as much as knowing how often the
    outcome happens; negative means it is worse than that. This is the form
    worth reporting, because a Brier score alone looks impressive whenever the
    base rate is far from a half.
    """
    realised = np.asarray(outcomes, dtype=float)
    base_rate = float(realised.mean())
    reference = float(np.mean((base_rate - realised) ** 2))
    if reference == 0:
        return math.nan

    return 1.0 - brier_score(probabilities, outcomes) / reference


@dataclass(frozen=True)
class CalibrationBin:
    """One band of forecast probability, and what actually happened in it."""

    lower: float
    upper: float
    forecast: float
    realised: float
    count: int


def calibration_curve(
    probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> list[CalibrationBin]:
    """Group forecasts into probability bands and compare stated to realised.

    A band containing no forecast is omitted rather than reported as zero: no
    forecast was made there, which is not the same as forecasts there being
    wrong.

    Raises:
        InsufficientObservationsError: the sequences differ in length or are empty.
        ValueError: fewer than two bins, which cannot show a curve.
    """
    if len(probabilities) != len(outcomes) or not probabilities:
        msg = f"{len(probabilities)} probabilities against {len(outcomes)} outcomes"
        raise InsufficientObservationsError(msg)
    if bins < 2:
        msg = f"a calibration curve needs at least two bins, got {bins}"
        raise ValueError(msg)

    predicted = np.asarray(probabilities, dtype=float)
    realised = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)

    curve: list[CalibrationBin] = []
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        # The final band includes its upper edge so a forecast of exactly 1.0
        # is scored rather than dropped.
        in_bin = (
            (predicted >= lower) & (predicted <= upper)
            if index == bins - 1
            else (predicted >= lower) & (predicted < upper)
        )
        if not in_bin.any():
            continue

        curve.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                forecast=float(predicted[in_bin].mean()),
                realised=float(realised[in_bin].mean()),
                count=int(in_bin.sum()),
            )
        )

    return curve
