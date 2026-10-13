"""Score the agent arm against the baseline on identical issuer-days.

The comparison reuses the baseline's own machinery — its walk-forward
forecasts, its crediting of a day's filings to one forecast per issuer, its
coefficient and its overlap-robust error — so the two arms differ in what
forecast, never in how they were scored. An issuer-day enters only if the agent
forecast every one of its labelled filings; otherwise it leaves both arms,
because scoring the baseline on filings the agent never saw would compare
different evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date

from arbiter.arms.evaluate import (
    DayOfEvents,
    _scores_by_issuer,  # pyright: ignore[reportPrivateUsage]
    forecast_walk_forward,
)
from arbiter.evaluation.metrics import (
    MIN_EVENTS_PER_DAY,
    ICSummary,
    information_coefficient,
    summarise_ic,
)


class WindowNotScoredError(ValueError):
    """Raised when a window day is not one the baseline's walk-forward scored."""


@dataclass(frozen=True)
class Comparison:
    """The two arms on the same issuer-days, and their paired difference."""

    agent: ICSummary
    baseline: ICSummary
    #: Agent minus baseline, day by day, over days where both are defined.
    difference: ICSummary
    issuer_days: int
    #: Issuer-days the baseline could score that left the comparison because
    #: the agent had not forecast every labelled filing in them.
    dropped_issuer_days: int


def daily_ic(
    days: Sequence[DayOfEvents], forecasts: Mapping[date, Sequence[float]]
) -> tuple[dict[date, float], int]:
    """Each day's coefficient over its issuers, and the issuer-days behind them.

    `forecasts` holds one value per row, in the day's row order. A day with
    fewer than `MIN_EVENTS_PER_DAY` issuers contributes no coefficient, as in
    the baseline's own evaluation.
    """
    daily: dict[date, float] = {}
    issuer_days = 0
    for day in days:
        scores = _scores_by_issuer(forecasts[day.day], day)
        if len(scores) >= MIN_EVENTS_PER_DAY:
            daily[day.day] = information_coefficient(
                [score.forecast for score in scores],
                [score.abnormal_return for score in scores],
            )
            issuer_days += len(scores)
    return daily, issuer_days


def _keep_rows(day: DayOfEvents, rows: Sequence[int]) -> DayOfEvents:
    return replace(
        day,
        features=[day.features[i] for i in rows],
        outcomes=[day.outcomes[i] for i in rows],
        returns=[day.returns[i] for i in rows],
        issuers=[day.issuers[i] for i in rows],
        exit_sessions=[day.exit_sessions[i] for i in rows],
        accessions=[day.accessions[i] for i in rows],
    )


def compare(
    days: Sequence[DayOfEvents],
    agent_by_accession: Mapping[str, float],
    window: tuple[date, date],
    *,
    overlap_lags: int,
) -> Comparison:
    """Pair the agent with the walk-forward baseline over `window`, inclusive.

    `days` is every stored day, so the baseline trains walk-forward as it does
    for its own report; only window days are compared.

    Raises:
        WindowNotScoredError: a window day with events is not a scored day of
            the baseline's walk-forward, so it would have no baseline forecast.
        ValueError: a window day was loaded without accession numbers.
        InsufficientObservationsError: fewer than two days carry a coefficient.
    """
    scored: dict[date, list[float]] = {}
    for _, _, block in forecast_walk_forward(days):
        scored.update(block)

    first, last = window
    kept_days: list[DayOfEvents] = []
    agent_rows: dict[date, list[float]] = {}
    baseline_rows: dict[date, list[float]] = {}
    dropped = 0
    for day in days:
        if not first <= day.day <= last or len(day) == 0:
            continue
        if day.day not in scored:
            msg = f"{day.day} has events but no baseline forecast: it is not a scored day"
            raise WindowNotScoredError(msg)
        if len(day.accessions) != len(day):
            msg = f"{day.day} was loaded without accession numbers"
            raise ValueError(msg)
        covered = {
            issuer
            for issuer in set(day.issuers)
            if all(
                accession in agent_by_accession
                for accession, owner in zip(day.accessions, day.issuers, strict=True)
                if owner == issuer
            )
        }
        dropped += len(set(day.issuers) - covered)
        rows = [i for i, issuer in enumerate(day.issuers) if issuer in covered]
        if not rows:
            continue
        kept = _keep_rows(day, rows)
        kept_days.append(kept)
        agent_rows[day.day] = [agent_by_accession[a] for a in kept.accessions]
        baseline_rows[day.day] = [scored[day.day][i] for i in rows]

    agent_daily, issuer_days = daily_ic(kept_days, agent_rows)
    baseline_daily, _ = daily_ic(kept_days, baseline_rows)
    paired = {
        day: agent_daily[day] - baseline_daily[day]
        for day in agent_daily.keys() & baseline_daily.keys()
    }
    return Comparison(
        agent=summarise_ic(agent_daily, observations=issuer_days, overlap_lags=overlap_lags),
        baseline=summarise_ic(
            baseline_daily, observations=issuer_days, overlap_lags=overlap_lags
        ),
        difference=summarise_ic(paired, observations=issuer_days, overlap_lags=overlap_lags),
        issuer_days=issuer_days,
        dropped_issuer_days=dropped,
    )
