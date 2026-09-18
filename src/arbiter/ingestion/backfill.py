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


#: Attempts per day before a transient failure is recorded as a real one. A
#: dropped connection partway through a multi-hour run says nothing about the
#: day it interrupted, and without this it both loses that day permanently and
#: counts toward the share that stops the run.
_ATTEMPTS_PER_DAY = 3

#: Seconds to wait between attempts, doubling. Long enough for a transient
#: network fault to clear, short against a day that costs minutes anyway.
_RETRY_BACKOFF_SECONDS = 5


class SystemicBackfillError(RuntimeError):
    """Raised when so many days fail that the run is not worth continuing."""


def _run_level_faults() -> tuple[type[BaseException], ...]:
    """Return the failures that indict the run rather than one day.

    Imported lazily because they live in modules that reach the network, and
    day selection must stay importable without them.
    """
    from arbiter.evaluation.resolution import MissingBenchmarkSeriesError
    from arbiter.ingestion.market import MissingCredentialsError, SystemicRejectionError

    return (MissingCredentialsError, SystemicRejectionError, MissingBenchmarkSeriesError)


_RUN_LEVEL_FAULTS = _run_level_faults()


def _with_retries[T](step: Callable[[date], T], day: date) -> T:
    """Run one day's work, retrying a transient transport failure.

    Only transport faults are retried. A day that fails to parse will fail
    identically on a second attempt, and retrying it would triple the cost of
    every genuinely broken day for nothing.
    """
    import time

    import httpx

    for attempt in range(1, _ATTEMPTS_PER_DAY + 1):
        try:
            return step(day)
        except httpx.TransportError:
            if attempt == _ATTEMPTS_PER_DAY:
                raise
            time.sleep(_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
    raise AssertionError("unreachable: the loop either returns or raises")


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
            counts = _with_retries(ingest, day)
            report.events += counts.get("insider", 0) + counts.get("redflag", 0)
            report.labels += _with_retries(resolve, day)
            report.completed.append(day)
        except _RUN_LEVEL_FAULTS:
            # Not a property of this day. A missing credential or a refused feed
            # will fail every remaining day identically, and absorbing it into a
            # per-day rate would spend a quarter of the run discovering that.
            raise
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


#: The domains a complete day holds, each with its events and its labels.
_DOMAINS = ("insider", "redflag")


def day_is_stored(root: Path, day: date) -> bool:
    """Report whether this day is complete: events *and* labels, for every domain.

    Resuming on the dataset rather than on a separate progress file means an
    interrupted run leaves nothing to reconcile.

    Labels are part of the test, not just events. Ingestion writes the event
    partitions before labelling runs, so a day whose labelling failed still has
    them — and a resume rule that looked only at events would treat that day as
    finished and skip it forever, leaving a hole exactly where a failure was
    already recorded. The same applies after a labelling bug is fixed: the day
    has to be re-labelled, and only a rule that notices the missing labels will
    let it be.
    """
    from arbiter.ingestion.store import partition_exists

    return all(
        partition_exists(root, domain, day) and partition_exists(root, f"labels-{domain}", day)
        for domain in _DOMAINS
    )


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
