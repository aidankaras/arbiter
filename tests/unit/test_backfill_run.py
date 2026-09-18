"""How a backfill behaves over hours: resuming, tolerating, and giving up.

A run covering months takes hours and will be interrupted. What matters is that
restarting it costs nothing for work already done, that one unreadable day does
not discard the hours before it, and that a run failing for a reason which will
not resolve itself stops instead of spending those hours producing a dataset
whose gaps nobody can account for.
"""

from datetime import date

import httpx
import pytest

from arbiter.ingestion import backfill as backfill_module
from arbiter.ingestion.backfill import SystemicBackfillError, backfill
from arbiter.ingestion.market import MissingCredentialsError

DAYS = [date(2026, 8, day) for day in (3, 4, 5, 6, 7)]


def _counts(insider: int = 10, redflag: int = 2) -> dict[str, int]:
    return {"insider": insider, "redflag": redflag, "rejected": 0, "unpriceable": 0}


def test_every_day_is_ingested_then_resolved_in_order():
    calls: list[str] = []

    backfill(
        DAYS,
        ingest=lambda day: (calls.append(f"ingest {day.day}"), _counts())[1],
        resolve=lambda day: (calls.append(f"resolve {day.day}"), 9)[1],
        is_stored=lambda day: False,
    )

    assert calls[:4] == ["ingest 3", "resolve 3", "ingest 4", "resolve 4"]


def test_a_day_already_stored_is_not_fetched_again():
    """Resuming must cost nothing for work already done."""
    fetched: list[date] = []
    stored = {date(2026, 8, 3), date(2026, 8, 4)}

    report = backfill(
        DAYS,
        ingest=lambda day: (fetched.append(day), _counts())[1],
        resolve=lambda day: 9,
        is_stored=lambda day: day in stored,
    )

    assert fetched == [date(2026, 8, 5), date(2026, 8, 6), date(2026, 8, 7)]
    assert report.skipped == [date(2026, 8, 3), date(2026, 8, 4)]
    assert len(report.completed) == 3


def test_one_unreadable_day_does_not_discard_the_others():
    def ingest(day: date) -> dict[str, int]:
        if day == date(2026, 8, 5):
            raise RuntimeError("EDGAR index unavailable")
        return _counts()

    report = backfill(DAYS, ingest=ingest, resolve=lambda day: 9, is_stored=lambda day: False)

    assert len(report.completed) == 4
    assert list(report.failed) == [date(2026, 8, 5)]


def test_a_failed_day_records_why_so_the_gap_can_be_explained():
    """A hole in the dataset with no recorded cause is indistinguishable from a bug."""
    report = backfill(
        [date(2026, 8, 3)],
        ingest=lambda day: (_ for _ in ()).throw(RuntimeError("EDGAR index unavailable")),
        resolve=lambda day: 9,
        is_stored=lambda day: False,
    )

    assert "RuntimeError" in report.failed[date(2026, 8, 3)]
    assert "EDGAR index unavailable" in report.failed[date(2026, 8, 3)]


def test_a_run_that_mostly_fails_stops_rather_than_grinding_on():
    """Hours spent producing unexplained gaps is the outcome worth preventing."""
    with pytest.raises(SystemicBackfillError, match="unexplained gaps"):
        backfill(
            DAYS,
            ingest=lambda day: (_ for _ in ()).throw(ConnectionError("network down")),
            resolve=lambda day: 9,
            is_stored=lambda day: False,
        )


def test_the_systemic_check_waits_for_enough_days_to_judge():
    """One failure out of one is not evidence that a long run is doomed."""
    report = backfill(
        DAYS,
        ingest=lambda day: (
            (_ for _ in ()).throw(RuntimeError("x")) if day == DAYS[0] else _counts()
        ),
        resolve=lambda day: 9,
        is_stored=lambda day: False,
    )

    assert len(report.failed) == 1
    assert len(report.completed) == 4


def test_events_and_labels_accumulate_across_days():
    report = backfill(
        DAYS,
        ingest=lambda day: _counts(insider=10, redflag=2),
        resolve=lambda day: 9,
        is_stored=lambda day: False,
    )

    assert report.events == 5 * 12
    assert report.labels == 5 * 9


def test_progress_is_reported_per_day_so_a_long_run_is_observable():
    lines: list[str] = []

    backfill(
        DAYS,
        ingest=lambda day: _counts(),
        resolve=lambda day: 9,
        is_stored=lambda day: False,
        on_progress=lines.append,
    )

    assert len(lines) == 5
    assert "2026-08-03" in lines[0]
    assert "done=1/5" in lines[0]


def test_no_days_is_a_run_that_does_nothing_rather_than_an_error():
    report = backfill(
        [], ingest=lambda day: _counts(), resolve=lambda day: 9, is_stored=lambda d: False
    )

    assert report.attempted == 0
    assert report.events == 0


def test_a_dropped_connection_is_retried_rather_than_losing_the_day(
    monkeypatch: pytest.MonkeyPatch,
):
    """A transient fault says nothing about the day it interrupted.

    Without a retry it both loses that day permanently — the resume rule sees
    stored events and skips it — and counts toward the share that stops the run.
    A real run aborted at 3 of 10 days on read timeouts this way.
    """
    monkeypatch.setattr(backfill_module, "_RETRY_BACKOFF_SECONDS", 0)
    attempts: list[date] = []

    def flaky(day: date) -> dict[str, int]:
        attempts.append(day)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("the read operation timed out")
        return _counts()

    report = backfill(
        [DAYS[0]], ingest=flaky, resolve=lambda day: 9, is_stored=lambda day: False
    )

    assert report.completed == [DAYS[0]]
    assert len(attempts) == 3


def test_a_day_that_keeps_timing_out_is_eventually_recorded_as_failed(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(backfill_module, "_RETRY_BACKOFF_SECONDS", 0)

    def always_timing_out(day: date) -> dict[str, int]:
        raise httpx.ReadTimeout("the read operation timed out")

    report = backfill(
        [DAYS[0]],
        ingest=always_timing_out,
        resolve=lambda day: 9,
        is_stored=lambda day: False,
    )

    assert "ReadTimeout" in report.failed[DAYS[0]]


def test_a_parse_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch):
    """It would fail identically, tripling the cost of every broken day."""
    monkeypatch.setattr(backfill_module, "_RETRY_BACKOFF_SECONDS", 0)
    attempts: list[date] = []

    def unparseable(day: date) -> dict[str, int]:
        attempts.append(day)
        raise ValueError("the filing format changed")

    backfill([DAYS[0]], ingest=unparseable, resolve=lambda day: 9, is_stored=lambda day: False)

    assert len(attempts) == 1


def test_a_missing_credential_stops_the_run_immediately(monkeypatch: pytest.MonkeyPatch):
    """It will fail every remaining day identically.

    Absorbing it into a per-day rate spends a quarter of a multi-hour run
    discovering something the first day already proved.
    """
    monkeypatch.setattr(backfill_module, "_RETRY_BACKOFF_SECONDS", 0)
    attempted: list[date] = []

    def no_credentials(day: date) -> dict[str, int]:
        attempted.append(day)
        raise MissingCredentialsError("ALPACA_API_KEY is required")

    with pytest.raises(MissingCredentialsError):
        backfill(
            DAYS, ingest=no_credentials, resolve=lambda day: 9, is_stored=lambda day: False
        )

    assert attempted == [DAYS[0]], "the run stops on the first day, not the fourth"
