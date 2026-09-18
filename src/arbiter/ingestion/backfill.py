"""Building a history to train and measure against.

A single day of filings is enough to prove the pipeline runs; it is not enough
to measure anything. The information coefficient of a forecast has a standard
error near the inverse square root of the number of observations, so a claim
about skill needs thousands of events, drawn from enough distinct days that they
are not all reacting to the same week of market conditions.

Cost is fixed per trading day rather than per event: every Form 4 accepted that
day must be fetched to learn whether it qualifies. Which days to cover is
therefore the only lever on how long a run takes, and sampling every Nth trading
day spreads observations across months at the same cost as a contiguous block.

A run covering months is interrupted sooner or later, so days already stored are
skipped rather than refetched, and a day that fails does not end the run — the
failures are reported together at the end, unless so many fail that continuing
would produce a dataset with unexplained holes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from arbiter.ingestion.edgar import is_trading_day

#: A run losing this share of its days is failing for a reason that will not
#: resolve itself, and continuing would spend hours producing a dataset whose
#: holes have no recorded explanation.
_SYSTEMIC_FAILURE_RATE = 0.25


class SystemicBackfillError(RuntimeError):
    """Raised when so many days fail that the run is not worth continuing."""


def sampled_trading_days(start: date, end: date, every: int = 1) -> list[date]:
    """Return the trading days in an inclusive range, taking every Nth one.

    Stepping over trading days rather than calendar days keeps the spacing even
    and rotates through weekdays, so no weekday effect in filing behaviour is
    confounded with what is later measured.

    Raises:
        ValueError: the range runs backwards, or the step is below one. Both
            would otherwise return an empty list, which is indistinguishable
            from a range that genuinely contains no trading day.
    """
    if every < 1:
        msg = f"every must be at least 1, got {every}"
        raise ValueError(msg)
    if start > end:
        msg = f"range starts after it ends: {start.isoformat()} to {end.isoformat()}"
        raise ValueError(msg)

    trading: list[date] = []
    day = start
    while day <= end:
        if is_trading_day(day):
            trading.append(day)
        day += timedelta(days=1)

    return trading[::every]


@dataclass
class BackfillReport:
    """What a backfill run covered, and what it could not."""

    completed: list[date] = field(default_factory=list[date])
    skipped: list[date] = field(default_factory=list[date])
    failed: dict[date, str] = field(default_factory=dict[date, str])
    events: int = 0
    labels: int = 0

    @property
    def attempted(self) -> int:
        """Days this run actually worked on, excluding those already stored."""
        return len(self.completed) + len(self.failed)


def backfill(
    days: Sequence[date],
    ingest: Callable[[date], dict[str, int]],
    resolve: Callable[[date], int],
    *,
    is_stored: Callable[[date], bool],
    on_progress: Callable[[str], None] | None = None,
) -> BackfillReport:
    """Ingest and label each day, skipping those already stored.

    The work is injected rather than imported so that the sequencing, the
    resume rule, and the failure policy can be tested without reaching the
    network — which is the part worth testing, the rest being a loop.

    Raises:
        SystemicBackfillError: too large a share of the attempted days failed,
            so the run is stopped rather than left to produce a dataset whose
            gaps have no recorded cause.
    """
    report = BackfillReport()

    for day in days:
        if is_stored(day):
            report.skipped.append(day)
            continue

        try:
            counts = ingest(day)
            report.events += counts.get("insider", 0) + counts.get("redflag", 0)
            report.labels += resolve(day)
            report.completed.append(day)
        except Exception as exc:
            # One unreadable day must not cost the hours already spent. The
            # cause is kept per day so a gap in the dataset can always be
            # explained, and the share is judged below.
            report.failed[day] = f"{type(exc).__name__}: {exc}"

        if on_progress is not None:
            # The cause travels with the line rather than waiting for the final
            # summary: a run lasting hours is only steerable if its first
            # failure is legible when it happens.
            cause = report.failed.get(day)
            on_progress(
                f"{day.isoformat()}  events={report.events}  labels={report.labels}  "
                f"done={len(report.completed)}/{len(days)}  failed={len(report.failed)}"
                + (f"  <- {cause}" if cause else "")
            )

        if (
            report.attempted >= 4
            and len(report.failed) > report.attempted * _SYSTEMIC_FAILURE_RATE
        ):
            causes = ", ".join(
                sorted({cause.split(":")[0] for cause in report.failed.values()})
            )
            msg = (
                f"{len(report.failed)} of {report.attempted} attempted days failed "
                f"({causes}); stopping rather than building a dataset with "
                "unexplained gaps"
            )
            raise SystemicBackfillError(msg)

    return report


def day_is_stored(
    root: Path, day: date, domains: Sequence[str] = ("insider", "redflag")
) -> bool:
    """Report whether every domain already holds a partition for this day.

    Resuming on this rather than on a separate progress file means an
    interrupted run leaves nothing to reconcile: the dataset itself records how
    far the run got.
    """
    from arbiter.ingestion.store import partition_exists

    return all(partition_exists(root, domain, day) for domain in domains)


def ingest_and_resolve(
    root: Path, min_value_usd: Decimal
) -> tuple[Callable[[date], dict[str, int]], Callable[[date], int]]:
    """Return the real ingest and resolve steps, bound to a store and a filter."""
    from arbiter.evaluation.resolution import HORIZONS, resolve_stored_day
    from arbiter.ingestion.market import Bar, current_session_date, daily_bars_tolerating_gaps
    from arbiter.ingestion.pipeline import ingest_day
    from arbiter.ingestion.sectors import benchmark_for_issuer, issuer_ticker

    def ingest(day: date) -> dict[str, int]:
        return ingest_day(day, root, min_value_usd)

    def resolve(day: date) -> int:
        today = current_session_date()

        def fetch(symbols: Sequence[str], start: date, end: date) -> dict[str, list[Bar]]:
            return daily_bars_tolerating_gaps(list(symbols), start, end, today)

        return sum(
            resolve_stored_day(
                day=day,
                domain=domain,
                root=root,
                fetch_bars=fetch,
                benchmark_for=benchmark_for_issuer,
                today=today,
                ticker_for=issuer_ticker,
            )
            for domain in HORIZONS
        )

    return ingest, resolve
