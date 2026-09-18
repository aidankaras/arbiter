"""Resolving a day's stored events into labels.

The price fetcher is injected, so these tests describe the orchestration: which
events are measured, how many price requests that costs, and what happens to an
event whose prices do not support a measurement.
"""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from arbiter.evaluation.resolution import (
    MissingBenchmarkSeriesError,
    ResolutionRequest,
    SystemicUnmeasurableError,
    resolve_day,
)
from arbiter.ingestion.market import Bar

SESSION_HOUR = 4


def _series(first_day: int, closes: list[str]) -> list[Bar]:
    return [
        Bar(
            timestamp=datetime(2026, 8, first_day + offset, SESSION_HOUR, tzinfo=UTC),
            open=Decimal(close),
            high=Decimal(close),
            low=Decimal(close),
            close=Decimal(close),
            volume=Decimal("1000"),
        )
        for offset, close in enumerate(closes)
    ]


def _request(accession: str, ticker: str = "MO", day: int = 3) -> ResolutionRequest:
    return ResolutionRequest(
        accession_no=accession,
        as_of=datetime(2026, 8, day, 20, 47, tzinfo=UTC),
        ticker=ticker,
        cik=764180,
        horizon=5,
    )


def _prices(**series: list[Bar]):
    """Return a fetcher over canned series, recording what it was asked for.

    `calls` counts requests, not symbols: the fetcher takes every symbol at once,
    so one entry per call is what the rate limit actually sees.
    """
    calls: list[list[str]] = []

    def fetch(symbols, start: date, end: date) -> dict[str, list[Bar]]:
        calls.append(list(symbols))
        return {symbol: series.get(symbol, []) for symbol in symbols}

    return fetch, calls


def test_a_resolvable_event_becomes_a_label():
    issuer = _series(3, ["100", "100", "101", "102", "103", "104", "110"])
    benchmark = _series(3, ["50", "50", "50", "50", "50", "50", "52"])
    fetch, _ = _prices(MO=issuer, XLP=benchmark)

    labels, _ = resolve_day(
        [_request("a-1")], fetch, benchmark_for=lambda cik: "XLP", today=date(2026, 9, 1)
    )

    assert len(labels) == 1
    assert labels[0].accession_no == "a-1"
    assert labels[0].benchmark_symbol == "XLP"


def test_the_label_measures_from_the_session_after_the_filing():
    """Entry is the 4 August open, not the 3 August one the filing was accepted on."""
    issuer = _series(3, ["999", "100", "100", "100", "100", "100", "110"])
    benchmark = _series(3, ["999", "50", "50", "50", "50", "50", "50"])
    fetch, _ = _prices(MO=issuer, XLP=benchmark)

    labels, _ = resolve_day(
        [_request("a-1")], fetch, benchmark_for=lambda cik: "XLP", today=date(2026, 9, 1)
    )

    assert labels[0].entry_session == date(2026, 8, 4)
    assert abs(labels[0].abnormal_return - Decimal("0.10")) < Decimal("0.0001")


def test_events_whose_window_has_not_closed_are_left_alone():
    """They are not failures; they are simply not measurable yet."""
    fetch, calls = _prices()

    labels, _ = resolve_day(
        [_request("a-1", day=28)],
        fetch,
        benchmark_for=lambda cik: "XLP",
        today=date(2026, 9, 1),
    )

    assert labels == []
    assert calls == [], "an unmeasurable event must not cost a price request"


def test_a_days_requests_do_not_grow_with_the_number_of_events():
    """A request per issuer would exhaust a per-minute rate limit on a real day.

    Two requests, not one: issuer tickers and benchmarks are fetched separately
    because a refused issuer should cost only its own events while a refused
    benchmark indicts the request. Both are constant in the number of events,
    which is the property that matters for the rate limit.
    """
    issuer = _series(3, ["100"] * 12)
    benchmark = _series(3, ["50"] * 12)
    fetch, calls = _prices(MO=issuer, XLP=benchmark)

    resolve_day(
        [_request("a-1"), _request("a-2", day=5), _request("a-3", day=7)],
        fetch,
        benchmark_for=lambda cik: "XLP",
        today=date(2026, 9, 1),
    )

    assert len(calls) == 2
    assert set(calls[0]) == {"MO"}, "issuer tickers travel together"
    assert set(calls[1]) == {"XLP"}, "benchmarks travel apart"


def test_an_event_with_too_little_price_history_is_skipped():
    """A short series means the measurement is unavailable, not that it is zero."""
    issuer = _series(3, ["100", "101"])
    benchmark = _series(3, ["50", "50"])
    fetch, _ = _prices(MO=issuer, XLP=benchmark)

    labels, _ = resolve_day(
        [_request("a-1")], fetch, benchmark_for=lambda cik: "XLP", today=date(2026, 9, 1)
    )

    assert labels == []


def test_a_benchmark_with_no_series_stops_the_day_rather_than_thinning_it():
    """Every event in a sector shares its benchmark, so losing one loses them all.

    A thinly traded issuer the service refuses costs only its own events. A
    sector ETF is one symbol of a dozen that hundreds of events depend on, and
    at that share no proportional guard would ever notice it — so silently
    skipping would remove a whole sector from the measured population.
    """
    issuer = _series(3, ["100"] * 8)
    fetch, _ = _prices(MO=issuer)

    with pytest.raises(MissingBenchmarkSeriesError, match="XLP"):
        resolve_day(
            [_request("a-1")], fetch, benchmark_for=lambda cik: "XLP", today=date(2026, 9, 1)
        )


def test_one_unmeasurable_event_does_not_stop_the_others():
    """A missing series for one issuer must not cost the whole day's labels."""
    good = _series(3, ["100", "100", "101", "102", "103", "104", "110"])
    benchmark = _series(3, ["50"] * 7)
    fetch, _ = _prices(MO=good, XLP=benchmark)

    labels, _ = resolve_day(
        [_request("a-1"), _request("b-1", ticker="NOPRICES")],
        fetch,
        benchmark_for=lambda cik: "XLP",
        today=date(2026, 9, 1),
    )

    assert [label.accession_no for label in labels] == ["a-1"]


def test_resolving_nothing_returns_nothing():
    """No events means no labels, no exclusions, and no request."""
    fetch, calls = _prices()

    labels, unmeasurable = resolve_day(
        [], fetch, benchmark_for=lambda cik: "XLP", today=date(2026, 9, 1)
    )

    assert labels == []
    assert unmeasurable == []
    assert calls == []


def test_an_event_that_cannot_be_measured_is_recorded_not_just_dropped():
    """A ripe event producing no label is a fact about the day.

    Swallowing it is how a price window one session short removed 675 of 675
    events from a day while the run reported success and wrote an empty
    partition.
    """
    issuer = _series(3, ["100", "101"])
    benchmark = _series(3, ["50"] * 12)
    fetch, _ = _prices(MO=issuer, XLP=benchmark)

    labels, unmeasurable = resolve_day(
        [_request("a-1")], fetch, benchmark_for=lambda cik: "XLP", today=date(2026, 9, 1)
    )

    assert labels == []
    assert unmeasurable[0]["accession_no"] == "a-1"
    assert unmeasurable[0]["ticker"] == "MO"
    assert unmeasurable[0]["reason"], "the cause travels with the record"


def test_a_day_where_most_ripe_events_fail_is_refused():
    """They share one cause — a broken window or a refused feed, not thin issuers."""
    benchmark = _series(3, ["50"] * 12)
    fetch, _ = _prices(XLP=benchmark)

    requests = [_request(f"a-{index}") for index in range(12)]

    with pytest.raises(SystemicUnmeasurableError, match="indicts the window"):
        resolve_day(requests, fetch, benchmark_for=lambda cik: "XLP", today=date(2026, 9, 1))


def test_a_thin_day_is_not_judged_against_that_share():
    """One event failing is one event, not evidence that the day is broken."""
    benchmark = _series(3, ["50"] * 12)
    fetch, _ = _prices(XLP=benchmark)

    labels, unmeasurable = resolve_day(
        [_request("a-1")], fetch, benchmark_for=lambda cik: "XLP", today=date(2026, 9, 1)
    )

    assert labels == []
    assert len(unmeasurable) == 1
