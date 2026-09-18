"""Measuring an arm against the outcomes it did not see.

The split between what a model is fitted on and what it is judged by is by
date, never at random. A random split would place events from the same day on
both sides, and since events filed on one day share a market, the model would
be scored partly on days it had already been shown — reporting as skill what is
partly memory.

Discrimination is scored one issuer per day rather than one filing per day.
Several insiders at one company on one day produce several events that resolve
to a single outcome, so scoring each would count one observation many times and
shrink the apparent standard error accordingly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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

#: The share of days used to fit. The remainder is held out and scored once.
TRAIN_FRACTION = 0.6


@dataclass(frozen=True)
class DayOfEvents:
    """One day's events, their features, and the outcomes they resolved to."""

    day: date
    features: list[dict[str, float]]
    outcomes: list[int]
    returns: list[float]
    issuers: list[int]

    def __len__(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class Evaluation:
    """What an arm scored on days it was not fitted on."""

    ic: ICSummary
    daily_ic: dict[date, float]
    calibration: list[CalibrationBin]
    brier_skill: float
    base_rate: float
    train_days: list[date]
    test_days: list[date]
    train_events: int
    test_events: int
    test_issuer_days: int
    coefficients: dict[str, float]


def load_day(root: Path, domain: str, day: date) -> DayOfEvents | None:
    """Assemble one day's events with the outcomes they resolved to.

    Events without a label are dropped rather than assigned a neutral outcome:
    they were not measurable, and inventing a zero return for them would put
    fabricated observations into the evaluation.

    Returns `None` for a day that was never processed, so a caller can tell a
    missing day from a day that produced nothing.
    """
    from arbiter.ingestion.store import partition_exists, read_events

    if not partition_exists(root, domain, day) or not partition_exists(
        root, f"labels-{domain}", day
    ):
        return None

    events = read_events(root, domain, day)
    # Several rows of one filing share an accession number and resolve to the
    # same outcome, so the first is representative rather than arbitrary.
    outcomes: dict[str, float] = {}
    for label in read_events(root, f"labels-{domain}", day):
        outcomes.setdefault(str(label["accession_no"]), float(label["abnormal_return"]))

    labelled = [event for event in events if str(event["accession_no"]) in outcomes]
    if not labelled:
        return DayOfEvents(day=day, features=[], outcomes=[], returns=[], issuers=[])

    returns = [outcomes[str(event["accession_no"])] for event in labelled]
    return DayOfEvents(
        day=day,
        features=day_features(labelled),
        outcomes=[int(value > 0) for value in returns],
        returns=returns,
        issuers=[int(event["cik"]) for event in labelled],
    )


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


def split_days(
    days: Sequence[date], train_fraction: float = TRAIN_FRACTION
) -> tuple[list[date], list[date]]:
    """Divide days chronologically into what is fitted on and what is scored.

    Raises:
        InsufficientObservationsError: the split would leave fewer than one
            training day or fewer than two test days, the latter being the
            minimum from which a spread across days can be estimated.
    """
    ordered = sorted(days)
    cut = int(len(ordered) * train_fraction)
    train, test = ordered[:cut], ordered[cut:]

    if not train or len(test) < 2:
        msg = (
            f"{len(ordered)} days split into {len(train)} for fitting and "
            f"{len(test)} for scoring; at least 1 and 2 are needed"
        )
        raise InsufficientObservationsError(msg)
    return train, test


def evaluate_baseline(
    days: Sequence[DayOfEvents], train_fraction: float = TRAIN_FRACTION
) -> Evaluation:
    """Fit the baseline on the earlier days and score it on the later ones.

    Raises:
        InsufficientObservationsError: too few days to split, or no labelled
            events on either side of the split.
    """
    populated = [day for day in days if len(day) > 0]
    train_days, test_days = split_days([day.day for day in populated], train_fraction)

    by_day = {day.day: day for day in populated}
    train = [by_day[day] for day in train_days]
    test = [by_day[day] for day in test_days]

    train_features = [row for day in train for row in day.features]
    train_outcomes = [outcome for day in train for outcome in day.outcomes]
    model: FittedBaseline = fit_baseline(train_features, train_outcomes)

    daily: dict[date, float] = {}
    pooled_probabilities: list[float] = []
    pooled_outcomes: list[int] = []
    issuer_days = 0

    for day in test:
        probabilities = forecast(model, day.features)
        pooled_probabilities.extend(probabilities)
        pooled_outcomes.extend(day.outcomes)

        issuer_forecasts, issuer_returns = _by_issuer(probabilities, day)
        issuer_days += len(issuer_forecasts)
        # A day with a handful of issuers produces a rank correlation that is
        # mostly noise; including it would add variance to the average without
        # adding information.
        if len(issuer_forecasts) >= _MIN_PER_DAY:
            daily[day.day] = information_coefficient(issuer_forecasts, issuer_returns)

    return Evaluation(
        ic=summarise_ic(daily, observations=issuer_days),
        daily_ic=daily,
        calibration=calibration_curve(pooled_probabilities, pooled_outcomes),
        brier_skill=brier_skill_score(pooled_probabilities, pooled_outcomes),
        base_rate=sum(pooled_outcomes) / len(pooled_outcomes),
        train_days=train_days,
        test_days=test_days,
        train_events=len(train_features),
        test_events=len(pooled_outcomes),
        test_issuer_days=issuer_days,
        coefficients=model.coefficients(),
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
