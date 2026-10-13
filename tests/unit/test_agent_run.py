"""The runner's decisions: what it writes, skips, excludes, and when it stops."""

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from arbiter.arms.agent.ledger import Exclusion
from arbiter.arms.agent.prompt import group_issuer_days, prompt_version
from arbiter.arms.agent.run import Limits, PartialForecastError, run_days, start_of_day
from arbiter.arms.agent.runtime import Status
from arbiter.packets.schema import EvidencePacket
from tests.agent_fakes import FakeLedger
from tests.agent_samples import RECORDED_RESULT, ScriptedRuntime, failed, make_packet, ok

DAY = date(2026, 7, 14)
NOW = datetime(2026, 7, 20, 16, 0, tzinfo=UTC)
GENEROUS = Limits(
    daily_spend_usd=Decimal("100"),
    run_tokens=10_000_000,
    max_calls=1_000,
    max_weekly_utilization=0.95,
)
# Issuer 1 files twice (two filings), issuer 2 once.
PACKETS = [
    make_packet("0000000001-26-000001", cik=1),
    make_packet("0000000001-26-000002", cik=1, shares="2000"),
    make_packet("0000000002-26-000001", cik=2),
]


def _run(
    runtime: ScriptedRuntime,
    ledger: FakeLedger,
    limits: Limits = GENEROUS,
    packets: list[EvidencePacket] = PACKETS,
):
    return run_days(
        [DAY], lambda _: packets, runtime, ledger, limits, run_id="run-1", now=lambda: NOW
    )


def test_each_group_gets_the_mean_of_three_samples_on_every_filing() -> None:
    ledger = FakeLedger()
    report = _run(
        ScriptedRuntime([ok(0.2), ok(0.5), ok(0.8), ok(0.6), ok(0.6), ok(0.6)]), ledger
    )
    assert (report.forecasts, report.calls, report.exclusions) == (2, 6, 0)
    version = prompt_version()
    issuer_1 = [
        ledger.forecasts_by_key[(version, e)][1]
        for e in ("0000000001-26-000001", "0000000001-26-000002")
    ]
    assert [row.p_up for row in issuer_1] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert issuer_1[0].samples == (0.2, 0.5, 0.8)
    assert issuer_1[0].issuer_day_hash == issuer_1[1].issuer_day_hash
    # Every call is metered against the group it served.
    group_hashes = {g.issuer_day_hash for g in group_issuer_days(DAY, PACKETS)}
    assert {packet_hash for _, _, packet_hash, _ in ledger.calls} == group_hashes
    assert len(ledger.calls) == 6


def test_a_second_run_skips_what_the_first_wrote() -> None:
    ledger = FakeLedger()
    _run(ScriptedRuntime([ok(0.5)] * 6), ledger)
    report = _run(ScriptedRuntime([]), ledger)
    assert (report.skipped, report.calls) == (2, 0)


def test_a_partly_written_group_stops_the_run() -> None:
    ledger = FakeLedger()
    group = group_issuer_days(DAY, PACKETS)[0]
    from arbiter.arms.agent.ledger import ForecastRow

    ledger.write_forecast(
        "claude-sonnet-5",
        prompt_version(),
        [ForecastRow(group.event_ids[0], "h", group.issuer_day_hash, 0.5, (0.5,), ("r",))],
    )
    with pytest.raises(PartialForecastError):
        _run(ScriptedRuntime([]), ledger)


def test_an_invalid_sample_is_retried_once() -> None:
    ledger = FakeLedger()
    script = [failed(Status.INVALID), ok(0.4), ok(0.4), ok(0.4)] + [ok(0.5)] * 3
    report = _run(ScriptedRuntime(script), ledger)
    assert (report.forecasts, report.exclusions, report.calls) == (2, 0, 7)
    # The failed attempt was metered too: it cost as much as a good one.
    assert len(ledger.calls) == 7


@pytest.mark.parametrize("status", [Status.INVALID, Status.REFUSED])
def test_a_sample_that_fails_twice_excludes_the_group(status: Status) -> None:
    ledger = FakeLedger()
    script = [ok(0.4), failed(status), failed(status)] + [ok(0.5)] * 3
    report = _run(ScriptedRuntime(script), ledger)
    assert (report.forecasts, report.exclusions) == (1, 1)
    (exclusion,) = ledger.exclusions.values()
    assert isinstance(exclusion, Exclusion)
    assert (exclusion.cik, exclusion.reason, exclusion.samples_valid) == (1, status.value, 1)
    assert not any(e.startswith("0000000001") for _, e in ledger.forecasts_by_key)
    # Recorded, so the next run does not spend on it again.
    assert _run(ScriptedRuntime([]), ledger).skipped == 2


@pytest.mark.parametrize("status", [Status.ERROR, Status.RATE_LIMITED])
def test_a_failure_unrelated_to_the_evidence_stops_without_recording(status: Status) -> None:
    ledger = FakeLedger()
    report = _run(ScriptedRuntime([ok(0.4), failed(status, metered=False)]), ledger)
    assert report.halted is not None and status.value in report.halted
    assert (report.forecasts, report.exclusions) == (0, 0)
    assert ledger.exclusions == {}
    # The group is still open: the next run forecasts it.
    later = _run(ScriptedRuntime([ok(0.5)] * 6), ledger)
    assert later.forecasts == 2


def test_the_daily_ceiling_stops_the_run_before_a_call() -> None:
    ledger = FakeLedger()
    ledger.now = NOW
    for _ in range(3):
        ledger.record_call(RECORDED_RESULT, "earlier-run", "x")
    spent = ledger.spent_since(start_of_day(NOW))
    # The premise: something was spent today, and the ceiling leaves no room for a group.
    assert spent > 0
    limits = Limits(spent + Decimal("0.01"), 10_000_000, 1_000, 0.95)
    report = _run(ScriptedRuntime([]), ledger, limits)
    assert report.calls == 0
    assert report.halted is not None and "daily ceiling" in report.halted


def test_the_daily_ceiling_resets_at_midnight_eastern_not_utc() -> None:
    ledger = FakeLedger()
    # 22:00 Eastern on the 20th is already the 21st in UTC: the calendars disagree.
    late_evening = datetime(2026, 7, 21, 2, 0, tzinfo=UTC)
    assert late_evening.date() != start_of_day(late_evening).date()
    ledger.now = datetime(2026, 7, 20, 3, 0, tzinfo=UTC)  # 23:00 Eastern on the 19th
    ledger.record_call(RECORDED_RESULT, "yesterday", "x")
    ledger.now = datetime(2026, 7, 20, 16, 0, tzinfo=UTC)  # noon Eastern on the 20th
    ledger.record_call(RECORDED_RESULT, "today", "x")
    one_call = Decimal(str(RECORDED_RESULT["total_cost_usd"]))
    assert abs(ledger.spent_since(start_of_day(late_evening)) - one_call) < Decimal("0.000001")


def test_the_call_cap_counts_retries_before_starting_a_group() -> None:
    limits = Limits(Decimal("100"), 10_000_000, 5, 0.95)
    report = _run(ScriptedRuntime([]), FakeLedger(), limits)
    assert report.calls == 0
    assert report.halted is not None and "call cap" in report.halted


def test_the_weekly_usage_ceiling_stops_the_next_group() -> None:
    limits = Limits(Decimal("100"), 10_000_000, 1_000, 0.85)
    report = _run(ScriptedRuntime([ok(0.5, weekly=0.86)] * 3), FakeLedger(), limits)
    assert (report.forecasts, report.calls) == (1, 3)
    assert report.halted is not None and "weekly usage" in report.halted
    assert report.utilization_end["seven_day"] == 0.86
