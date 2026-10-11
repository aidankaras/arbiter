"""The information coefficient within groups of issuers, shared by every breakdown.

A breakdown regroups the headline's issuer-days by some attribute and reports the
rank correlation inside each group. What differs between breakdowns is only how an
issuer-day is assigned to a group; the arithmetic, the minimum issuers per
group-day and the accounting are common and live here so they cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from arbiter.arms.evaluate import IssuerScore
from arbiter.evaluation.metrics import (
    MIN_EVENTS_PER_DAY,
    ICSummary,
    InsufficientObservationsError,
    information_coefficient,
    summarise_ic,
)

#: Maps one day's scored issuers to a group index per issuer CIK. An issuer absent
#: from the mapping is ungrouped.
Assigner = Callable[[date, Sequence[IssuerScore]], Mapping[int, int]]


@dataclass(frozen=True)
class GroupIC:
    """One group's information coefficient across days."""

    #: `None` when fewer than two group-days reached the threshold.
    ic: ICSummary | None
    #: Group-days that contributed a coefficient.
    days: int
    #: Issuer-days on those group-days only.
    issuer_days: int
    #: The issuer-days behind the coefficient, with their day, so a caller can
    #: summarise an attribute over exactly the sample the row rests on.
    members: tuple[tuple[date, IssuerScore], ...]


@dataclass(frozen=True)
class GroupedIC:
    """Per-group coefficients and the sample they leave out.

    Every group's issuer-days, `dropped` and `ungrouped` add up to `scored_issuer_days`.
    """

    groups: tuple[GroupIC, ...]
    #: Scored issuer-days the assigner gave no group.
    ungrouped: int
    #: Grouped issuer-days whose group-day held fewer than `MIN_EVENTS_PER_DAY`.
    dropped: int
    scored_issuer_days: int


def ic_within_groups(
    scores: Mapping[date, Sequence[IssuerScore]],
    assign: Assigner,
    group_count: int,
    overlap_lags: int = 0,
) -> GroupedIC:
    """Compute the information coefficient within each group of each day.

    A group-day contributes only with at least `MIN_EVENTS_PER_DAY` issuers: the
    headline's minimum applied to each group. Per-day coefficients are summarised
    exactly as the headline's are, with `overlap_lags` for the error.
    """
    daily: list[dict[date, float]] = [{} for _ in range(group_count)]
    counted = [0] * group_count
    members: list[list[tuple[date, IssuerScore]]] = [[] for _ in range(group_count)]
    ungrouped = 0
    dropped = 0
    total = 0

    for day, issuers in scores.items():
        total += len(issuers)
        assigned = assign(day, issuers)
        ungrouped += sum(1 for s in issuers if s.cik not in assigned)
        for index in range(group_count):
            rows = [s for s in issuers if assigned.get(s.cik) == index]
            if len(rows) < MIN_EVENTS_PER_DAY:
                dropped += len(rows)
                continue
            daily[index][day] = information_coefficient(
                [m.forecast for m in rows], [m.abnormal_return for m in rows]
            )
            counted[index] += len(rows)
            members[index].extend((day, m) for m in rows)

    groups: list[GroupIC] = []
    for index in range(group_count):
        try:
            summary = summarise_ic(daily[index], counted[index], overlap_lags=overlap_lags)
        except InsufficientObservationsError:
            summary = None
        groups.append(
            GroupIC(
                ic=summary,
                days=summary.days if summary else len(daily[index]),
                issuer_days=counted[index],
                members=tuple(members[index]),
            )
        )
    return GroupedIC(
        tuple(groups), ungrouped=ungrouped, dropped=dropped, scored_issuer_days=total
    )
