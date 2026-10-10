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
    #: The larger of the independent error and one allowing for `overlap_lags`
    #: days of correlation between neighbours; `t_statistic` and
    #: `is_significant` use it.
    standard_error: float
    t_statistic: float
    days: int
    observations: int
    #: Treating every day as independent; equal to `standard_error` when
    #: `overlap_lags` is 0. Reported so the effect of the overlap is visible.
    independent_standard_error: float = math.nan
    #: Neighbouring days whose outcome windows overlap, allowed for in
    #: `standard_error`.
    overlap_lags: int = 0

    @property
    def is_significant(self) -> bool:
        """Whether the mean is two standard errors clear of zero.

        A conventional threshold, and a weak one: it is reported so that a
        summary cannot be read as a finding without the reader seeing how far
        from zero it actually is.
        """
        return abs(self.t_statistic) >= 2.0


def defined_ic[Key](daily: Mapping[Key, float]) -> dict[Key, float]:
    """Return the days whose coefficient is defined.

    A day where every forecast or every outcome tied has no ranking to score.
    Such days are excluded rather than counted as zero, here and anywhere the
    daily values are summarised or drawn, so the two cannot disagree.
    """
    return {key: value for key, value in daily.items() if not math.isnan(value)}


def summarise_ic[Key](
    daily: Mapping[Key, float], observations: int, overlap_lags: int = 0
) -> ICSummary:
    """Average per-day information coefficients and report the uncertainty.

    The standard error is taken across days. Days are not independent when
    their outcome windows overlap: a five-session outcome scored on consecutive
    sessions shares up to four sessions with each neighbour, so their
    coefficients move together and an error treating them as independent is
    optimistic. `overlap_lags` neighbours are therefore allowed for with a
    Newey-West estimator (Bartlett weights), scaled by n/(n-1) so that with no
    lags it reduces exactly to the sample standard deviation over root n. Lags
    count positions in key order, which equal sessions when scored days are
    consecutive sessions; across gaps between sampled days the overlap they
    allow for is larger than the true one, which errs conservative. The error
    reported is the larger of this and the independent one.

    Days whose coefficient is undefined are excluded rather than counted as
    zero, since no ranking existed to score.

    Raises:
        InsufficientObservationsError: fewer than two days carry a defined
            coefficient, so no spread can be estimated.
    """
    # In key order: the overlap allowed for is between neighbouring days.
    values = [value for _, value in sorted(defined_ic(daily).items())]
    if len(values) < 2:
        msg = f"an average across days needs at least two days, got {len(values)}"
        raise InsufficientObservationsError(msg)

    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    # The sample standard deviation, so a single day cannot read as certainty.
    independent = float(array.std(ddof=1) / math.sqrt(len(array)))
    # Never narrower than the independent error. Newey-West is biased downward
    # at samples this small, so on its own it overstates precision (an iid
    # simulation at 60 days rejects at |t| > 2 about 7% of the time against a
    # nominal 5%); the floor keeps the headline from claiming more precision
    # than days treated as independent would support. It errs towards the
    # conclusion that there is no skill, and both errors are reported.
    standard_error = max(independent, _newey_west_standard_error(array, overlap_lags))

    return ICSummary(
        mean=mean,
        standard_error=standard_error,
        t_statistic=_t_statistic(mean, standard_error),
        days=len(array),
        observations=observations,
        independent_standard_error=independent,
        overlap_lags=overlap_lags,
    )


def _newey_west_standard_error(values: np.ndarray, lags: int) -> float:
    """Standard error of the mean of a series correlated over `lags` neighbours.

    Raises:
        ValueError: `lags` is negative.
    """
    if lags < 0:
        msg = f"overlap_lags must be non-negative, not {lags}"
        raise ValueError(msg)
    n = len(values)
    deviations = values - values.mean()

    def autocovariance(lag: int) -> float:
        return float((deviations[lag:] * deviations[: n - lag]).sum() / n)

    # Autocovariances at lags of n or more are zero (no pairs exist); the
    # Bartlett weights keep their bandwidth `lags` regardless.
    long_run = autocovariance(0) + 2 * sum(
        (1 - lag / (lags + 1)) * autocovariance(lag) for lag in range(1, min(lags, n - 1) + 1)
    )
    # Bartlett weights keep the estimate non-negative; the floor guards only
    # against rounding below zero on a series with no spread.
    return math.sqrt(max(long_run, 0.0) / n * n / (n - 1))


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


def brier_skill_score(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    reference_rate: float | Sequence[float] | None = None,
) -> float:
    """Return the Brier score's improvement over forecasting a constant rate.

    Zero means the forecast is worth exactly as much as forecasting that rate
    for every outcome; negative means it is worse than that. This is the form
    worth reporting, because a Brier score alone looks impressive whenever the
    base rate is far from a half.

    `reference_rate` is the constant compared against. Left out, it is the
    rate realised in `outcomes` themselves, which no forecaster could have
    known in advance — a stricter reference than any feasible constant. A
    rate fixed before the outcomes, such as the training period's, is the
    comparison a forecaster could actually have made.

    A sequence gives one reference per outcome, for forecasts made by a model
    refitted over time: each outcome is compared against the constant quoted
    when it was forecast, not against one rate fixed for the whole period.

    Raises:
        InsufficientObservationsError: a sequence of references whose length
            differs from the outcomes'.
    """
    realised = np.asarray(outcomes, dtype=float)
    base_rate = float(realised.mean()) if reference_rate is None else reference_rate
    if not isinstance(base_rate, float | int) and len(base_rate) != len(outcomes):
        msg = f"{len(base_rate)} reference rates against {len(outcomes)} outcomes"
        raise InsufficientObservationsError(msg)
    reference = float(np.mean((np.asarray(base_rate, dtype=float) - realised) ** 2))
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
