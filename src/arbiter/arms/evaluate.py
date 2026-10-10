"""Measuring an arm against the outcomes it did not see.

What a model is fitted on and what it is judged by are separated by date,
never at random. A random split would place events from the same day on both
sides, and since events filed on one day share a market, the model would be
scored partly on days it had already been shown — reporting as skill what is
partly memory.

The separation is walk-forward. After an initial fitting window, the remaining
days are scored in consecutive blocks, each forecast by a model refitted on
every earlier event whose outcome had closed before the block began. A single
chronological split put the whole verdict on one cut point: two reasonable cuts
of the same store gave t = 0.57 and t = 2.93. Walk-forward scores every day
after the initial window and lets the training set grow as outcomes realise,
the way a forecaster running live would.

Discrimination is scored one issuer per day rather than one filing per day.
Several insiders at one company on one day produce several events that resolve
to a single outcome, so scoring each would count one observation many times and
shrink the apparent standard error accordingly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from arbiter.arms.baseline import FittedBaseline, fit_baseline, forecast
from arbiter.arms.features import day_features
from arbiter.evaluation.metrics import MIN_EVENTS_PER_DAY as _MIN_PER_DAY
from arbiter.evaluation.metrics import (
    CalibrationBin,
    ICSummary,
    InsufficientObservationsError,
    brier_skill_score,
    calibration_curve,
    information_coefficient,
    summarise_ic,
)

#: The share of populated days in the initial fitting window. Those days are
#: fitted on and never scored: the first forecast needs something to rest on.
INITIAL_FRACTION = 0.3

#: Scored days per refit, counted in populated stored days rather than sessions.
REFIT_EVERY = 10


@dataclass(frozen=True)
class DayOfEvents:
    """One day's events, their features, and the outcomes they resolved to."""

    day: date
    features: list[dict[str, float]]
    outcomes: list[int]
    returns: list[float]
    issuers: list[int]
    #: The session each event's outcome window closed on, from its stored label.
    #: Taken from the label rather than inferred from the filing day: a filing
    #: accepted late in a day, or a window crossing a gap in the price
    #: sessions, closes later than a calendar count from the partition day says.
    exit_sessions: list[date]
    #: Events filed that day that resolved to no outcome, every one of them
    #: recorded as an exclusion by the labeling pass.
    unlabelled: int = 0

    def __len__(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class Fold:
    """One refit and the block of days it forecast."""

    #: Days contributing at least one fitted event: those filed before the
    #: block whose outcome had closed by the block's first day. A day whose
    #: events closed at different sessions may be fitted in part.
    fit_days: list[date]
    scored_days: list[date]
    #: Share of fitted issuer-days that rose; the feasible reference for the
    #: block's Brier skill.
    fitted_base_rate: float
    train_events: int


@dataclass(frozen=True)
class Evaluation:
    """What an arm scored on days it was not fitted on."""

    ic: ICSummary
    daily_ic: dict[date, float]
    calibration: list[CalibrationBin]
    #: Against forecasting the base rate of the model that made each forecast:
    #: the constant a forecaster could have quoted before any scored outcome in
    #: that block was known.
    brier_skill: float
    #: Against forecasting the scored period's own rate, which was not knowable
    #: in advance — a stricter reference than any feasible constant.
    brier_skill_scored_rate: float
    #: The reference rate averaged over scored issuer-days: each fold's fitted
    #: up-rate, weighted by the issuer-days that fold scored. This is the mean of
    #: the per-observation references `brier_skill` is measured against.
    fitted_base_rate: float
    #: Share of scored issuer-days that rose.
    base_rate: float
    #: Days fitted on before the first forecast and never scored.
    initial_days: list[date]
    #: Every scored day, in order: all populated days after the initial window.
    test_days: list[date]
    folds: list[Fold]
    #: Events in the last fold's fitting set.
    train_events: int
    test_events: int
    #: Events on scored days dropped for want of an outcome, each recorded as
    #: an exclusion. Reported so the labelled share of the sample is visible.
    test_events_unlabelled: int
    #: Issuer-days on days that contributed a rank correlation: the sample the
    #: information coefficient rests on.
    test_issuer_days: int
    #: Every scored issuer-day, which calibration, Brier skill and the base
    #: rate are pooled over. Larger than `test_issuer_days` whenever a scored
    #: day had too few issuers to rank.
    pooled_issuer_days: int
    #: The last fold's model, which saw the most data. Earlier folds' weights
    #: are not kept; the report labels this table as the final refit's.
    coefficients: dict[str, float]


class LabelCoverageError(ValueError):
    """Raised when a day's labels and events disagree beyond what was recorded.

    An unlabelled event is legitimate only when the labeling pass recorded why.
    One it did not record means the label partition predates the events it
    should describe — written before an outcome window closed, or left behind
    by a re-ingestion — and scoring the day would silently score a subset.
    """


def load_day(root: Path, domain: str, day: date) -> DayOfEvents | None:
    """Assemble one day's events with the outcomes they resolved to.

    Events without a label are dropped rather than assigned a neutral outcome:
    they were not measurable, and inventing a zero return for them would put
    fabricated observations into the evaluation. How many were dropped is kept,
    and each must have been recorded as an exclusion when the day was labelled.

    Returns `None` for a day that was never processed, so a caller can tell a
    missing day from a day that produced nothing.

    Raises:
        LabelCoverageError: a label names an event the day does not hold, or
            an event has neither a label nor a recorded exclusion.
    """
    from arbiter.ingestion.store import partition_exists, read_events, read_unpriceable

    if not partition_exists(root, domain, day) or not partition_exists(
        root, f"labels-{domain}", day
    ):
        return None

    events = read_events(root, domain, day)
    # Several rows of one filing share an accession number and resolve to the
    # same outcome, so the first is representative rather than arbitrary.
    outcomes: dict[str, float] = {}
    exits: dict[str, date] = {}
    for label in read_events(root, f"labels-{domain}", day):
        accession = str(label["accession_no"])
        outcomes.setdefault(accession, float(label["abnormal_return"]))
        exits.setdefault(accession, label["exit_session"])

    # Features are computed over the whole day and only then narrowed to the
    # events that resolved. One feature counts the insiders who had filed for
    # the same issuer by each event's acceptance, and computing it over the
    # labelled subset would make it depend on which events turned out to be
    # priceable — a fact from after the filing.
    accessions = {str(event["accession_no"]) for event in events}
    stray = sorted(set(outcomes) - accessions)
    excluded = {
        str(record["accession_no"])
        for record in read_unpriceable(root, domain, day, stage="labeling") or []
    }
    unaccounted = sorted(accessions - set(outcomes) - excluded)
    if stray or unaccounted:
        msg = (
            f"{day}: {len(stray)} labels name no stored event and {len(unaccounted)} "
            f"events have neither a label nor a recorded exclusion "
            f"(first: {(stray + unaccounted)[0]}); once its outcome windows have "
            f"closed, relabel it with `arbiter resolve {day} {domain}`"
        )
        raise LabelCoverageError(msg)

    features = day_features(events)
    kept = [
        index for index, event in enumerate(events) if str(event["accession_no"]) in outcomes
    ]
    unlabelled = len(events) - len(kept)
    if not kept:
        return DayOfEvents(
            day=day,
            features=[],
            outcomes=[],
            returns=[],
            issuers=[],
            exit_sessions=[],
            unlabelled=unlabelled,
        )

    returns = [outcomes[str(events[index]["accession_no"])] for index in kept]
    return DayOfEvents(
        day=day,
        features=[features[index] for index in kept],
        outcomes=[int(value > 0) for value in returns],
        returns=returns,
        issuers=[int(events[index]["cik"]) for index in kept],
        exit_sessions=[exits[str(events[index]["accession_no"])] for index in kept],
        unlabelled=unlabelled,
    )


def _rose(issuer_returns: Sequence[float]) -> list[int]:
    """Return whether each issuer-day rose, given its mean return.

    The one definition of the outcome, used for both the fitted reference rate
    and the scored outcomes: the sign of the issuer's mean return that day, not
    an average of its rows' signs. Two filings returning +0.05 and -0.01 are
    one issuer-day that rose.
    """
    return [int(realised > 0) for realised in issuer_returns]


def _issuer_day_outcomes(days: Sequence[DayOfEvents]) -> list[int]:
    """Return whether each issuer-day across `days` rose."""
    return [outcome for day in days for outcome in _rose(_by_issuer(day.returns, day)[1])]


def _by_issuer(forecasts: Sequence[float], day: DayOfEvents) -> tuple[list[float], list[float]]:
    """Collapse a day to one forecast and one outcome per issuer.

    Filings by several insiders at one company resolve to a single outcome, so
    the forecast for that issuer is the average of what was said about it.
    """
    grouped: dict[int, tuple[list[float], list[float]]] = {}
    for probability, realised, issuer in zip(forecasts, day.returns, day.issuers, strict=True):
        probabilities, returns = grouped.setdefault(issuer, ([], []))
        probabilities.append(probability)
        returns.append(realised)

    means = [
        (sum(probabilities) / len(probabilities), sum(returns) / len(returns))
        for probabilities, returns in grouped.values()
    ]
    return [mean for mean, _ in means], [realised for _, realised in means]


def _issuer_day_weights(day: DayOfEvents) -> list[float]:
    """Weight each row by one over the number of rows its issuer has that day."""
    rows_per_issuer: dict[int, int] = {}
    for issuer in day.issuers:
        rows_per_issuer[issuer] = rows_per_issuer.get(issuer, 0) + 1
    return [1.0 / rows_per_issuer[issuer] for issuer in day.issuers]


def _closed_before(day: DayOfEvents, cutoff: date) -> DayOfEvents | None:
    """Return the events of `day` whose outcomes closed before `cutoff`, if any."""
    kept = [index for index, exit_ in enumerate(day.exit_sessions) if exit_ < cutoff]
    if not kept:
        return None
    return replace(
        day,
        features=[day.features[i] for i in kept],
        outcomes=[day.outcomes[i] for i in kept],
        returns=[day.returns[i] for i in kept],
        issuers=[day.issuers[i] for i in kept],
        exit_sessions=[day.exit_sessions[i] for i in kept],
    )


def _fit_fold(train: Sequence[DayOfEvents]) -> tuple[FittedBaseline, float, int]:
    """Fit on `train`; return the model, its issuer-day up-rate and event count."""
    train_features = [row for day in train for row in day.features]
    train_outcomes = [outcome for day in train for outcome in day.outcomes]
    # Scoring credits each issuer once per day, so fitting does too: every row
    # is weighted by the share it holds of its issuer-day. Unweighted, one Form 4
    # carrying forty transactions would pull the fit forty times as hard as a
    # single-line filing, and the model would be fitted to a different
    # population from the one it is judged on.
    train_weights = [weight for day in train for weight in _issuer_day_weights(day)]
    model = fit_baseline(train_features, train_outcomes, train_weights)
    fitted_outcomes = _issuer_day_outcomes(train)
    return model, sum(fitted_outcomes) / len(fitted_outcomes), len(train_features)


def forecast_walk_forward(
    days: Sequence[DayOfEvents],
    *,
    initial_fraction: float = INITIAL_FRACTION,
    refit_every: int = REFIT_EVERY,
) -> list[tuple[Fold, FittedBaseline, dict[date, list[float]]]]:
    """Refit the baseline ahead of each block of days and forecast the block.

    The first `ceil(n * initial_fraction)` populated days form the initial
    window and are never scored. Every later populated day is scored, in
    consecutive blocks of `refit_every`. A block is forecast by a model fitted on
    each event filed before the block's first day whose outcome window had
    closed before it, judged per event from the stored label's exit session.
    Until a window closes its return shares market moves with the block's
    returns, so fitting on it would let the model see part of what it forecasts.
    Training expands to include earlier scored blocks.

    Returns each fold, its model, and the per-row forecasts of every scored day.

    Raises:
        ValueError: `initial_fraction` is not strictly between 0 and 1 (a
            negative one would slice from the end and fit on most of the days),
            or `refit_every` is below 1.
        InsufficientObservationsError: fewer than one initial day or two scored
            days, the latter being the minimum from which a spread across days
            can be estimated; or a block has no event whose outcome had closed
            before it, which a short initial window causes.
    """
    if not 0 < initial_fraction < 1:
        msg = f"initial_fraction must be strictly between 0 and 1, not {initial_fraction}"
        raise ValueError(msg)
    if refit_every < 1:
        msg = f"refit_every must be at least 1, not {refit_every}"
        raise ValueError(msg)

    by_day = {day.day: day for day in days if len(day) > 0}
    ordered = sorted(by_day)
    initial_count = math.ceil(len(ordered) * initial_fraction)
    scored = ordered[initial_count:]
    if initial_count < 1 or len(scored) < 2:
        msg = (
            f"{len(ordered)} days leave {initial_count} for the initial fit and "
            f"{len(scored)} for scoring; at least 1 and 2 are needed"
        )
        raise InsufficientObservationsError(msg)

    results: list[tuple[Fold, FittedBaseline, dict[date, list[float]]]] = []
    for offset in range(0, len(scored), refit_every):
        block = scored[offset : offset + refit_every]
        admitted = [
            closed
            for day in ordered
            if day < block[0]
            if (closed := _closed_before(by_day[day], block[0])) is not None
        ]
        if not admitted:
            msg = (
                f"no event had closed its outcome window by {block[0]}, so the "
                "first block has nothing to be fitted on"
            )
            raise InsufficientObservationsError(msg)
        model, fitted_rate, train_events = _fit_fold(admitted)
        forecasts = {day: forecast(model, by_day[day].features) for day in block}
        fold = Fold(
            fit_days=[day.day for day in admitted],
            scored_days=block,
            fitted_base_rate=fitted_rate,
            train_events=train_events,
        )
        results.append((fold, model, forecasts))
    return results


def evaluate_baseline(
    days: Sequence[DayOfEvents],
    *,
    initial_fraction: float = INITIAL_FRACTION,
    refit_every: int = REFIT_EVERY,
) -> Evaluation:
    """Score the baseline walk-forward; see `forecast_walk_forward` for the rule.

    Raises:
        ValueError: as `forecast_walk_forward`.
        InsufficientObservationsError: as `forecast_walk_forward`, or no
            labelled events on the scored days.
    """
    blocks = forecast_walk_forward(
        days,
        initial_fraction=initial_fraction,
        refit_every=refit_every,
    )
    by_day = {day.day: day for day in days if len(day) > 0}
    folds = [fold for fold, _, _ in blocks]
    test_days = [day for fold in folds for day in fold.scored_days]
    initial_days = sorted(by_day)[: len(by_day) - len(test_days)]

    daily: dict[date, float] = {}
    pooled_probabilities: list[float] = []
    pooled_outcomes: list[int] = []
    # The block's own fitted rate, repeated for each issuer-day it scored: the
    # constant a forecaster refitting on the same schedule could have quoted.
    references: list[float] = []
    issuer_days = 0
    # Counted over every day from the first scored one, including days on which
    # nothing resolved: those never enter the walk, but their events were filed
    # in the scoring period all the same.
    unlabelled = sum(day.unlabelled for day in days if day.day >= test_days[0])
    # Kept apart from the pooled counts, which are issuer-level: the report
    # states both, and conflating them would understate the data behind a run.
    scored_events = 0

    for fold, _, block_forecasts in blocks:
        for scored_day in fold.scored_days:
            day = by_day[scored_day]
            issuer_forecasts, issuer_returns = _by_issuer(block_forecasts[scored_day], day)
            scored_events += len(day)

            # Calibration, Brier skill and the base rate are pooled at the
            # issuer-day level, the same unit the information coefficient is
            # credited at. Pooled per filing they counted one outcome once per
            # transaction row — a single Form 4 carried forty-four on a measured
            # day — so the reliability curve's counts read as independent
            # observations while being copies of a few.
            pooled_probabilities.extend(issuer_forecasts)
            pooled_outcomes.extend(_rose(issuer_returns))
            references.extend([fold.fitted_base_rate] * len(issuer_forecasts))
            # A day with a handful of issuers produces a rank correlation that
            # is mostly noise; including it would add variance to the average
            # without adding information.
            if len(issuer_forecasts) >= _MIN_PER_DAY:
                daily[day.day] = information_coefficient(issuer_forecasts, issuer_returns)
                # Counted only for days that contribute a coefficient.
                # Accumulating it for every scored day would report a sample
                # larger than the one the estimate rests on, beside a line
                # saying how many days were excluded — inviting the reader to
                # divide one by the other.
                issuer_days += len(issuer_forecasts)

    return Evaluation(
        ic=summarise_ic(daily, observations=issuer_days),
        daily_ic=daily,
        calibration=calibration_curve(pooled_probabilities, pooled_outcomes),
        brier_skill=brier_skill_score(
            pooled_probabilities, pooled_outcomes, reference_rate=references
        ),
        brier_skill_scored_rate=brier_skill_score(pooled_probabilities, pooled_outcomes),
        fitted_base_rate=sum(references) / len(references),
        base_rate=sum(pooled_outcomes) / len(pooled_outcomes),
        initial_days=initial_days,
        test_days=test_days,
        folds=folds,
        train_events=folds[-1].train_events,
        test_events=scored_events,
        test_events_unlabelled=unlabelled,
        test_issuer_days=issuer_days,
        pooled_issuer_days=len(pooled_outcomes),
        coefficients=blocks[-1][1].coefficients(),
    )


def load_days(root: Path, domain: str, days: Sequence[date]) -> list[DayOfEvents]:
    """Load every processed day among those given, skipping the rest."""
    loaded = (load_day(root, domain, day) for day in days)
    return [day for day in loaded if day is not None]


def stored_days(root: Path, domain: str) -> list[date]:
    """Return every day the store holds labelled events for, in order."""
    labels = root / f"labels-{domain}"
    if not labels.exists():
        return []

    return sorted(date.fromisoformat(path.stem) for path in labels.glob("*.parquet"))
