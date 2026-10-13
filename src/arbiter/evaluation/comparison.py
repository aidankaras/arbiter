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

import math
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
    #: Issuer-days both arms were scored on.
    issuer_days: int
    #: Window issuer-days that left both arms because the agent had not
    #: forecast every labelled filing in them.
    dropped_issuer_days: int
    #: Window issuer-days the agent did forecast that neither arm could rank:
    #: on a day left with too few issuers, or a day the agent forecast flat.
    #: Window issuer-days = issuer_days + dropped + unranked, always.
    unranked_issuer_days: int
    #: Days dropped from every line because the agent's coefficient was
    #: undefined (the same forecast for every issuer that day).
    agent_undefined_days: int
    #: Window days with labelled filings, and those on which the agent forecast
    #: none of them: a run still in progress, not a finding.
    window_days: int
    days_not_run: int


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

    The baseline trains walk-forward on every stored day up to the window's end,
    as it does for its own report. Days stored after the window are left out:
    the walk-forward's initial cut and refit blocks are counted in days, so
    they would otherwise move the baseline a frozen agent run is compared with.
    All three lines of the result cover the same days: those where both arms
    have a coefficient.

    Raises:
        ValueError: the window ends before it starts, or a window day was
            loaded without accession numbers.
        WindowNotScoredError: no window day has labelled events, or one that
            does is not a scored day of the baseline's walk-forward.
        InsufficientObservationsError: fewer than two days carry a coefficient.
    """
    first, last = window
    if last < first:
        msg = f"the window ends before it starts: {first} to {last}"
        raise ValueError(msg)
    history = [day for day in days if day.day <= last]
    scored: dict[date, list[float]] = {}
    for _, _, block in forecast_walk_forward(history):
        scored.update(block)

    kept_days: list[DayOfEvents] = []
    agent_rows: dict[date, list[float]] = {}
    baseline_rows: dict[date, list[float]] = {}
    dropped = window_issuer_days = window_days = days_not_run = 0
    for day in history:
        if day.day < first or len(day) == 0:
            continue
        window_days += 1
        window_issuer_days += len(set(day.issuers))
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
        if not any(accession in agent_by_accession for accession in day.accessions):
            days_not_run += 1
        rows = [i for i, issuer in enumerate(day.issuers) if issuer in covered]
        if not rows:
            continue
        kept = _keep_rows(day, rows)
        kept_days.append(kept)
        agent_rows[day.day] = [agent_by_accession[a] for a in kept.accessions]
        baseline_rows[day.day] = [scored[day.day][i] for i in rows]

    if window_days == 0:
        msg = f"no window day from {first} to {last} has labelled events"
        raise WindowNotScoredError(msg)

    agent_daily, _ = daily_ic(kept_days, agent_rows)
    baseline_daily, _ = daily_ic(kept_days, baseline_rows)
    # Only days where both arms rank: an arm's line over other days would not
    # be comparable with the other's, and the difference would not be theirs.
    both = {d for d in agent_daily.keys() & baseline_daily.keys() if _defined(agent_daily[d])}
    both &= {d for d in baseline_daily if _defined(baseline_daily[d])}
    agent_undefined = sum(1 for d in agent_daily if not _defined(agent_daily[d]))
    issuer_days = sum(len(set(day.issuers)) for day in kept_days if day.day in both)
    kept_issuer_days = sum(len(set(day.issuers)) for day in kept_days)

    def summary(daily: Mapping[date, float]) -> ICSummary:
        return summarise_ic(
            {d: daily[d] for d in both}, observations=issuer_days, overlap_lags=overlap_lags
        )

    return Comparison(
        agent=summary(agent_daily),
        baseline=summary(baseline_daily),
        difference=summary({d: agent_daily[d] - baseline_daily[d] for d in both}),
        issuer_days=issuer_days,
        dropped_issuer_days=dropped,
        unranked_issuer_days=kept_issuer_days - issuer_days,
        agent_undefined_days=agent_undefined,
        window_days=window_days,
        days_not_run=days_not_run,
    )


def _defined(value: float) -> bool:
    return not math.isnan(value)
