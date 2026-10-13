"""Forecast issuer-days with the analyst, writing every outcome to the ledger.

Each forecast group gets `SAMPLES` independent calls, averaged. The decisions
that matter are about what happens when something goes wrong, and they are
fixed in advance rather than chosen once results are visible:

- A sample that is invalid or refused is retried once. A group still short of
  `SAMPLES` valid answers is excluded and recorded, never averaged over fewer:
  that would mix two estimators under one prompt version.
- A failure unrelated to the evidence — the process failed, a usage window is
  exhausted — stops the run without recording anything against the group, so
  the next run picks it up. Recording it would turn an outage into a permanent
  hole in the sample.
- Budget pressure stops the run before dispatch. It never reduces the work done
  per group.
- A group that is already partly written stops the run: that cannot happen in
  normal operation, and papering over it would hide whatever caused it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from zoneinfo import ZoneInfo

from arbiter.arms.agent.ledger import Exclusion, ForecastRow, Ledger
from arbiter.arms.agent.prompt import (
    MODEL_ID,
    SAMPLES,
    IssuerDay,
    group_issuer_days,
    prompt_version,
    render_user_message,
    system_prompt,
)
from arbiter.arms.agent.runtime import Runtime, Sample, Status
from arbiter.llm.budget import (
    BudgetExceededError,
    assert_within_spend_ceiling,
    assert_within_token_ceiling,
)
from arbiter.llm.usage import parse_model_usage
from arbiter.packets.schema import EvidencePacket

_EASTERN = ZoneInfo("America/New_York")
#: Assumed cost of one call before this run has measured any. The largest
#: measured issuer-day (64 packets) cost $0.064; a median one about $0.023.
DEFAULT_CALL_USD = Decimal("0.07")
#: Assumed tokens of one call before this run has measured any; the same
#: 64-packet issuer-day used 14,882.
DEFAULT_CALL_TOKENS = 15_000


#: The first scored day: the day after claude-sonnet-5's release on 2026-06-30,
#: so no scored filing can be in what it was trained on. Earlier days are the
#: development set, where the pilot runs and the prompt may still change.
SCORED_FROM = date(2026, 7, 1)


@dataclass(frozen=True)
class Limits:
    """When a run must stop before dispatching more work."""

    daily_spend_usd: Decimal
    run_tokens: int
    #: Calls this invocation may make, retries included.
    max_calls: int
    #: Stop once the seven-day usage window reports at least this share used.
    max_weekly_utilization: float


@dataclass
class RunReport:
    """What a run did, and why it stopped if it stopped early."""

    forecasts: int = 0
    exclusions: int = 0
    skipped: int = 0
    #: Groups whose filings changed after some of them were forecast or excluded,
    #: as when a re-ingest adds a late filing. Neither forecast nor excluded
    #: again, so the comparison drops them from both arms.
    changed: int = 0
    calls: int = 0
    halted: str | None = None
    utilization_start: dict[str, float] = field(default_factory=dict[str, float])
    utilization_end: dict[str, float] = field(default_factory=dict[str, float])


class _Halt(Exception):  # noqa: N818 - control flow inside this module only
    """Stop the run, recording nothing against the group in progress."""


@dataclass
class _Meter:
    """Per-call cost and tokens seen so far, for estimating the next group."""

    max_call_usd: Decimal = DEFAULT_CALL_USD
    max_call_tokens: int = DEFAULT_CALL_TOKENS

    def observe(self, sample: Sample) -> None:
        if sample.payload is None:
            return
        usage = parse_model_usage(sample.payload)
        self.max_call_usd = max(self.max_call_usd, sum((u.cost_usd for u in usage), Decimal(0)))
        self.max_call_tokens = max(self.max_call_tokens, sum(u.total_tokens for u in usage))


def start_of_day(instant: datetime) -> datetime:
    """Midnight Eastern on `instant`'s Eastern date: when a daily ceiling resets."""
    local = instant.astimezone(_EASTERN)
    return datetime.combine(local.date(), time(0), tzinfo=_EASTERN)


def run_days(
    days: Sequence[date],
    load: Callable[[date], Sequence[EvidencePacket]],
    runtime: Runtime,
    ledger: Ledger,
    limits: Limits,
    *,
    run_id: str,
    now: Callable[[], datetime],
) -> RunReport:
    """Forecast every unforecast group on `days`, in day then CIK order.

    Raises:
        EnvironmentViolationError: a call ran without the required isolation.
    """
    report = RunReport()
    meter = _Meter()
    version = prompt_version()
    system = system_prompt()
    try:
        for day in days:
            for group in group_issuer_days(day, load(day)):
                _forecast_group(
                    group, runtime, ledger, limits, report, meter,
                    run_id=run_id, version=version, system=system, now=now,
                )  # fmt: skip
    except (_Halt, BudgetExceededError) as stop:
        report.halted = str(stop)
    return report


def _forecast_group(
    group: IssuerDay,
    runtime: Runtime,
    ledger: Ledger,
    limits: Limits,
    report: RunReport,
    meter: _Meter,
    *,
    run_id: str,
    version: str,
    system: str,
    now: Callable[[], datetime],
) -> None:
    events = set(group.event_ids)
    decided = ledger.predicted(version, group.event_ids) | ledger.excluded_events(
        version, group.event_ids
    )
    if decided == events or ledger.is_excluded(version, group.issuer_day_hash):
        report.skipped += 1
        return
    if decided:
        # Drawing again would forecast a filing twice, or give a group that
        # failed a second chance only because its evidence changed.
        report.changed += 1
        return

    _check_limits(ledger, limits, report, meter, run_id=run_id, now=now)
    user = render_user_message(group)
    valid: list[Sample] = []
    failure: Sample | None = None
    for _ in range(SAMPLES):
        sample = _draw(runtime, ledger, report, meter, system, user, run_id, group)
        if sample.status is not Status.OK:
            sample = _draw(runtime, ledger, report, meter, system, user, run_id, group)
        if sample.status is Status.OK:
            valid.append(sample)
        else:
            failure = sample
            break

    if failure is not None:
        ledger.write_exclusion(
            Exclusion(
                event_day=group.day,
                cik=group.cik,
                issuer_day_hash=group.issuer_day_hash,
                prompt_version=version,
                reason=failure.status.value,
                samples_valid=len(valid),
                event_ids=group.event_ids,
                detail=failure.detail,
            )
        )
        report.exclusions += 1
        return

    probabilities = tuple(float(s.p_up) for s in valid if s.p_up is not None)
    rationales = tuple(s.rationale or "" for s in valid)
    p_up = sum(probabilities) / len(probabilities)
    ledger.write_forecast(
        MODEL_ID,
        version,
        [
            ForecastRow(
                event_id=event_id,
                packet_hash=group.packet_hash_for(event_id),
                issuer_day_hash=group.issuer_day_hash,
                p_up=p_up,
                samples=probabilities,
                rationales=rationales,
            )
            for event_id in group.event_ids
        ],
    )
    report.forecasts += 1


def _draw(
    runtime: Runtime,
    ledger: Ledger,
    report: RunReport,
    meter: _Meter,
    system: str,
    user: str,
    run_id: str,
    group: IssuerDay,
) -> Sample:
    """Draw and meter one sample; stop the run on a failure unrelated to the evidence."""
    sample = runtime.sample(system, user)
    report.calls += 1
    outage = sample.status in (Status.ERROR, Status.RATE_LIMITED)
    unmetered = ""
    if sample.payload is not None:
        try:
            ledger.record_call(sample.payload, run_id, group.issuer_day_hash)
            meter.observe(sample)
        except ValueError as refused:
            # A rejected request can report no usage at all. The halt it calls
            # for still happens; only an answer that cannot be metered is fatal.
            if not outage:
                raise
            unmetered = f"; its usage could not be metered ({refused})"
    report.utilization_start = report.utilization_start or dict(sample.utilization)
    # The latest call's report, even when empty: a stale reading must not pass
    # for a current one.
    report.utilization_end = dict(sample.utilization)
    if outage:
        msg = f"{sample.status.value} on issuer {group.cik} ({group.day}): {sample.detail}"
        raise _Halt(msg + unmetered)
    return sample


def _check_limits(
    ledger: Ledger,
    limits: Limits,
    report: RunReport,
    meter: _Meter,
    *,
    run_id: str,
    now: Callable[[], datetime],
) -> None:
    """Refuse to start a group the limits cannot cover, retries included.

    Raises:
        _Halt: the call cap or the weekly usage ceiling would be crossed.
        BudgetExceededError: the daily spend or the run's tokens would be.
    """
    worst_case_calls = 2 * SAMPLES
    if report.calls + worst_case_calls > limits.max_calls:
        msg = f"call cap: {report.calls} made, a group may need {worst_case_calls} more"
        raise _Halt(msg)
    weekly = report.utilization_end.get("seven_day")
    if report.calls and weekly is None:
        # Read from each call; a call that did not report it leaves the
        # ceiling unenforceable, and unreported is not unused.
        msg = "weekly usage not reported by the last call; the ceiling cannot be checked"
        raise _Halt(msg)
    if weekly is not None and weekly >= limits.max_weekly_utilization:
        msg = f"weekly usage at {weekly:.0%}, ceiling {limits.max_weekly_utilization:.0%}"
        raise _Halt(msg)
    assert_within_spend_ceiling(
        ledger.spent_since(start_of_day(now())),
        meter.max_call_usd * worst_case_calls,
        limits.daily_spend_usd,
    )
    assert_within_token_ceiling(
        ledger.tokens_for_run(run_id),
        meter.max_call_tokens * worst_case_calls,
        limits.run_tokens,
    )
