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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from arbiter.evaluation.labels import abnormal_return
from arbiter.ingestion.edgar import is_trading_day
from arbiter.ingestion.market import Bar, is_tradeable_symbol
from arbiter.ingestion.timestamps import SEC_TIMEZONE

#: Sessions held per domain. Insider information resolves within a week; the
#: drift after an auditor change or restatement runs for about a month, so a
#: five-day window would measure the announcement rather than its consequence.
HORIZONS = {"insider": 5, "redflag": 20}


#: Extra sessions fetched past the closing session. The exit session is chosen
#: from the bars themselves, so the window only has to be long enough to contain
#: it; this covers closures the holiday calendar does not list, and a window
#: slightly too long costs nothing but a few unused bars.
_WINDOW_SLACK_SESSIONS = 3

#: Sessions of price history fetched before the first event, so the entry
#: session that follows a filing is always inside the window.
_LEAD_DAYS = 5


@dataclass(frozen=True)
class ResolutionRequest:
    """One stored event, and what measuring it requires."""

    accession_no: str
    as_of: datetime
    ticker: str
    cik: int
    horizon: int


def entry_sessions(bars: list[Bar], as_of: datetime) -> list[Bar]:
    """Return the sessions an event could have traded, entry first.

    A session is tradeable only if it began after the filing became public. The
    bars therefore act as the trading calendar: whichever sessions the exchange
    actually held are the ones that count, so holidays and early closes need no
    separate table.
    """
    return [bar for bar in bars if bar.timestamp > as_of]


def sessions_after(day: date, sessions: int) -> date:
    """Return the calendar date a given number of trading sessions after `day`.

    Counted against the market calendar rather than scaled from calendar days.
    A fixed ratio of days per session is wrong by an amount that depends on
    which weekday the count starts from — five sessions after a Monday is eight
    calendar days, after a Friday it is ten — and that error falls entirely on
    filings late in the week.
    """
    remaining = sessions
    current = day
    while remaining > 0:
        current += timedelta(days=1)
        if is_trading_day(current):
            remaining -= 1
    return current


def window_closes_on(event: ResolutionRequest) -> date:
    """Return the session on which an event's horizon closes.

    Counted against the market calendar rather than scaled from calendar days.
    A fixed ratio of days per session is wrong by an amount that depends on the
    weekday the filing landed on — five sessions after a Monday is eight
    calendar days, after a Friday it is ten — so an approximation cuts the
    window short for filings late in the week and leaves those events with no
    price history covering their exit. That removes whole days from the
    measured population rather than degrading evenly across them.

    The entry session is the first the market holds after the filing became
    public, matching how the exit session is later chosen from the bars.
    """
    entry = sessions_after(event.as_of.astimezone(SEC_TIMEZONE).date(), 1)
    return sessions_after(entry, event.horizon)


def resolvable_on(event: ResolutionRequest, today: date) -> bool:
    """Report whether an event can be measured as of `today`.

    Two conditions, and the second is easy to overlook: the horizon must have
    closed, and the window must end before the current session, because the
    consolidated feed refuses a window covering today and this project does not
    substitute a thinner feed for recent prices.
    """
    return window_closes_on(event) < today


def plan_price_windows(
    events: Sequence[ResolutionRequest],
) -> dict[str, tuple[date, date]]:
    """Group events into one price window per symbol.

    A day's filings concentrate in far fewer issuers than events, and each
    window covers every event for its symbol, so the number of price requests
    follows the number of distinct symbols rather than the number of events.
    """
    windows: dict[str, tuple[date, date]] = {}
    for event in events:
        start = event.as_of.date() - timedelta(days=_LEAD_DAYS)
        # Fetched past the closing session, unlike the ripeness test above:
        # an unlisted closure would otherwise leave the window one session short
        # of the exit and cost the event entirely, while an over-long window
        # costs nothing but a few unused bars.
        end = sessions_after(window_closes_on(event), _WINDOW_SLACK_SESSIONS)
        current = windows.get(event.ticker)
        if current is None:
            windows[event.ticker] = (start, end)
        else:
            windows[event.ticker] = (min(current[0], start), max(current[1], end))
    return windows


class UnresolvableEventError(ValueError):
    """Raised when an event cannot be labelled from the series available.

    Callers exclude the event rather than substituting a value. An unresolvable
    event is missing from the sample; a defaulted one is a false observation in
    it, and only the second corrupts the statistics computed afterwards.
    """


class Label(BaseModel):
    """One event's realised outcome, with the evidence of how it was measured.

    The accession number travels with the label because labels are produced days
    or weeks after the events they measure, in a separate pass, and a measurement
    that cannot be joined back to its event is not evidence of anything.

    A model rather than a plain record because labels are stored, and the store
    derives its column types from the model's fields. Inferring them from a
    day's values instead would let two days disagree on decimal scale, which
    matters here more than anywhere: every label carries a return.
    """

    model_config = ConfigDict(frozen=True)

    accession_no: str
    abnormal_return: Decimal
    benchmark_symbol: str
    horizon_days: int
    entry_session: date
    exit_session: date


def with_tickers(
    rows: Sequence[Mapping[str, Any]], ticker_for: Callable[[int], str | None]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Give each row a ticker, looking one up where the filing does not carry it.

    Form 4 reports a ticker; an 8-K identifies its filer by CIK alone. Rows that
    already carry one are left untouched, since looking it up again would spend
    a request to learn what the filing already said.

    Issuers are looked up once each rather than once per event, because a day's
    filings concentrate in far fewer issuers than filings.

    Returns the rows that can be priced, and records for those that cannot. An
    issuer with no listed security is a fact about the issuer; dropping it
    silently would make it indistinguishable from an event that never existed.

    Raises:
        KeyError: a row carries no CIK, so there is no identity to look up.
    """
    priced: list[dict[str, Any]] = []
    unpriceable: list[dict[str, str]] = []
    resolved: dict[int, str | None] = {}

    for row in rows:
        if "cik" not in row:
            msg = "stored event is missing 'cik'; its issuer cannot be identified"
            raise KeyError(msg)

        existing = str(row.get("ticker") or "").strip()
        if existing:
            priced.append(dict(row))
            continue

        cik = int(row["cik"])
        if cik not in resolved:
            resolved[cik] = ticker_for(cik)

        ticker = resolved[cik]
        if not ticker:
            unpriceable.append(
                {
                    "accession_no": str(row["accession_no"]),
                    "reason": "issuer has no listed ticker",
                }
            )
            continue

        if not is_tradeable_symbol(ticker):
            # A preferred issue or unit, which this study does not cover, and
            # which the price service rejects — taking every other symbol in the
            # same batched request down with it.
            unpriceable.append(
                {
                    "accession_no": str(row["accession_no"]),
                    "reason": f"{ticker} is not a plain equity symbol",
                }
            )
            continue

        priced.append({**row, "ticker": ticker})

    return priced, unpriceable


def requests_from_rows(
    rows: Sequence[Mapping[str, Any]], domain: str
) -> list[ResolutionRequest]:
    """Convert stored event rows into resolution requests.

    The horizon comes from the domain rather than the row, because it is a
    property of the phenomenon being measured rather than of any single event.
    An unknown domain raises: defaulting one would measure every event of that
    kind over the wrong window, consistently and invisibly.

    Raises:
        KeyError: the domain has no horizon, or a row lacks a field the
            measurement needs. A row that cannot be priced is a gap in the
            pipeline, not an event that happens to resolve to nothing.
    """
    if domain not in HORIZONS:
        msg = f"no horizon is defined for domain {domain!r}"
        raise KeyError(msg)
    horizon = HORIZONS[domain]

    requests: list[ResolutionRequest] = []
    for row in rows:
        for field in ("accession_no", "as_of", "ticker", "cik"):
            if field not in row:
                msg = f"stored event is missing {field!r}; it cannot be measured"
                raise KeyError(msg)
        requests.append(
            ResolutionRequest(
                accession_no=str(row["accession_no"]),
                as_of=cast("datetime", row["as_of"]),
                ticker=str(row["ticker"]),
                cik=int(cast("int", row["cik"])),
                horizon=horizon,
            )
        )
    return requests


def resolve_stored_day(
    day: date,
    domain: str,
    root: Path,
    fetch_bars: Callable[[Sequence[str], date, date], dict[str, list[Bar]]],
    benchmark_for: Callable[[int], str],
    today: date,
    ticker_for: Callable[[int], str | None] | None = None,
) -> int:
    """Measure one stored day of a domain and write its labels.

    `ticker_for` supplies a ticker for domains whose filings do not carry one.
    Without it a red-flag day yields nothing, since an 8-K names its filer by
    CIK alone. Rows that already carry a ticker never reach it.

    Returns the number of labels written. A partition is written even when that
    number is zero, so a day that was processed and yielded nothing stays
    distinguishable from a day never processed — the same distinction the event
    store keeps, and for the same reason.

    Issuers that cannot be priced are recorded beside the labels rather than
    discarded, so a thin day can be told apart from a day whose issuers were
    unlistable.
    """
    from arbiter.ingestion.store import read_events, write_events, write_unpriceable

    rows: Sequence[Mapping[str, Any]] = read_events(root, domain, day)
    unpriceable: list[dict[str, str]] = []
    if ticker_for is not None:
        rows, unpriceable = with_tickers(rows, ticker_for)

    labels = resolve_day(requests_from_rows(rows, domain), fetch_bars, benchmark_for, today)
    write_events(labels, root, f"labels-{domain}", day)
    write_unpriceable(unpriceable, root, domain, day, stage="labeling")

    return len(labels)


def resolve_day(
    events: Sequence[ResolutionRequest],
    fetch_bars: Callable[[Sequence[str], date, date], dict[str, list[Bar]]],
    benchmark_for: Callable[[int], str],
    today: date,
) -> list[Label]:
    """Measure every event whose window has closed, and leave the rest.

    Every symbol is fetched in a single request spanning the day, rather than
    one request per symbol: a day's filings run to dozens of issuers against a
    rate limit measured per minute, and requesting them separately exhausted it
    before a single label was produced.

    An event that cannot be measured is omitted, never approximated. Too little
    price history and a missing benchmark are ordinary states — a recent event,
    a thinly covered ticker — and one of them must not cost the day's other
    labels, so each event is measured independently.

    Args:
        events: the stored events considered for measurement.
        fetch_bars: returns a symbol's daily bars across an inclusive window.
        benchmark_for: returns the benchmark symbol for an issuer's CIK.
        today: the current session date, used to decide what has settled.

    Returns:
        A label per measurable event, in the order the events were given.
    """
    due = [event for event in events if resolvable_on(event, today)]
    if not due:
        return []

    benchmarks = {event.accession_no: benchmark_for(event.cik) for event in due}

    # Every symbol is fetched in one request across one window spanning the day's
    # events. Requesting per symbol instead costs a request per issuer, and a
    # day's filings run to hundreds of issuers against a rate limit measured per
    # minute. The window is slightly wider than any single event needs, which is
    # far cheaper than the requests it saves.
    windows = plan_price_windows(due)
    start = min(span[0] for span in windows.values())
    end = max(span[1] for span in windows.values())
    symbols = sorted(set(windows) | set(benchmarks.values()))

    series = fetch_bars(symbols, start, end)

    labels: list[Label] = []
    for event in due:
        benchmark_symbol = benchmarks[event.accession_no]
        issuer_sessions = entry_sessions(series.get(event.ticker, []), event.as_of)
        benchmark_sessions = entry_sessions(series.get(benchmark_symbol, []), event.as_of)
        try:
            labels.append(
                resolve_label(
                    issuer_bars=issuer_sessions,
                    benchmark_bars=benchmark_sessions,
                    benchmark_symbol=benchmark_symbol,
                    horizon=event.horizon,
                    accession_no=event.accession_no,
                )
            )
        except UnresolvableEventError:
            continue
    return labels


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
    accession_no: str = "",
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
        accession_no=accession_no,
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
