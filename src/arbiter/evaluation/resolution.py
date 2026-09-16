"""Turning price series into a labelled outcome.

A label is only meaningful if it measures what a position could actually have
captured, so the window is defined by what a decision made after the filing
could trade: entry at the first open available afterwards, exit at the close of
the horizon's final session.

Each label records the benchmark it was measured against. Sector classification
is a judgment that may be revised, and a label that did not name its benchmark
would silently change meaning when that judgment changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from arbiter.evaluation.labels import abnormal_return
from arbiter.ingestion.market import Bar

#: Sessions held per domain. Insider information resolves within a week; the
#: drift after an auditor change or restatement runs for about a month, so a
#: five-day window would measure the announcement rather than its consequence.
HORIZONS = {"insider": 5, "redflag": 20}


class UnresolvableEventError(ValueError):
    """Raised when an event cannot be labelled from the series available.

    Callers exclude the event rather than substituting a value. An unresolvable
    event is missing from the sample; a defaulted one is a false observation in
    it, and only the second corrupts the statistics computed afterwards.
    """


@dataclass(frozen=True)
class Label:
    """One event's realised outcome, with the evidence of how it was measured."""

    abnormal_return: Decimal
    benchmark_symbol: str
    horizon_days: int
    entry_session: date
    exit_session: date


def _series_or_raise(bars: list[Bar], horizon: int, name: str) -> list[Bar]:
    """Return a series long enough to span the horizon.

    Raises:
        UnresolvableEventError: the series is absent or the window has not yet
            closed. Both are ordinary states — a benchmark may be missing, and a
            recent event simply has not finished — so neither is an error in the
            pipeline, only a reason to exclude the event.
    """
    if not bars:
        msg = f"no {name} price series is available; the event cannot be labelled"
        raise UnresolvableEventError(msg)

    # The entry session plus the horizon's sessions: a five-day label needs the
    # opening session and five more.
    required = horizon + 1
    if len(bars) < required:
        msg = (
            f"{name} series covers {len(bars)} sessions but the window needs "
            f"{required}; the horizon has not closed yet"
        )
        raise UnresolvableEventError(msg)
    return bars


def resolve_label(
    issuer_bars: list[Bar],
    benchmark_bars: list[Bar],
    benchmark_symbol: str,
    horizon: int,
) -> Label:
    """Measure one event's abnormal return over its horizon.

    Both series begin at the entry session, the first session tradeable after
    the event. Entry is that session's open rather than the previous close,
    which would credit the event with movement that preceded it.

    Raises:
        UnresolvableEventError: either series is missing or too short.
        InsufficientPriceDataError: a price within the window is unusable.
    """
    issuer = _series_or_raise(issuer_bars, horizon, "issuer")
    benchmark = _series_or_raise(benchmark_bars, horizon, "benchmark")

    entry, exit_ = issuer[0], issuer[horizon]
    benchmark_entry, benchmark_exit = benchmark[0], benchmark[horizon]

    return Label(
        abnormal_return=abnormal_return(
            entry_price=entry.open,
            exit_price=exit_.close,
            benchmark_entry=benchmark_entry.open,
            benchmark_exit=benchmark_exit.close,
        ),
        benchmark_symbol=benchmark_symbol,
        horizon_days=horizon,
        entry_session=entry.timestamp.date(),
        exit_session=exit_.timestamp.date(),
    )
