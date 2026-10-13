"""Behaviour both agent ledgers must share, run against each of them."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from arbiter.arms.agent.ledger import Exclusion, ForecastRow, Ledger

FIXTURE = Path(__file__).parent / "fixtures" / "claude_run.json"


def row(event_id: str, p_up: float = 0.6) -> ForecastRow:
    return ForecastRow(
        event_id=event_id,
        # Derived from the event so no other test in the shared database
        # can hold the same hash.
        packet_hash=hashlib.sha256(event_id.encode()).hexdigest(),
        issuer_day_hash=hashlib.sha256(b"day:" + event_id.encode()).hexdigest(),
        p_up=p_up,
        samples=(p_up, p_up, p_up),
        rationales=("r1", "r2", "r3"),
    )


def payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


def check_written_events_read_back_as_predicted(ledger: Ledger, version: str) -> None:
    ledger.write_forecast("claude-sonnet-5", version, [row("e-1"), row("e-2")])
    assert ledger.predicted(version, ["e-1", "e-2", "e-3"]) == {"e-1", "e-2"}
    assert ledger.predicted("another-version", ["e-1"]) == set()


def check_a_forecast_with_a_duplicate_writes_nothing(ledger: Ledger, version: str) -> None:
    ledger.write_forecast("claude-sonnet-5", version, [row("d-1")])
    with pytest.raises(Exception):  # noqa: B017 - each ledger raises its own type
        ledger.write_forecast("claude-sonnet-5", version, [row("d-2"), row("d-1")])
    # The new row in the failed write must not have landed on its own.
    assert ledger.predicted(version, ["d-2"]) == set()


def check_exclusions_are_per_forecast_group(ledger: Ledger, version: str) -> None:
    day = date(2026, 7, 1)
    morning, afternoon = "1" * 64, "2" * 64
    # One issuer, one day, two groups either side of the open: both can be excluded.
    ledger.write_exclusion(Exclusion(day, 42, morning, version, "invalid_output", 1, ("x-1",)))
    ledger.write_exclusion(Exclusion(day, 42, afternoon, version, "refused", 0, ("x-2",)))
    assert ledger.is_excluded(version, morning)
    assert ledger.is_excluded(version, afternoon)
    assert not ledger.is_excluded(version, "3" * 64)
    assert not ledger.is_excluded("another-version", morning)


def check_usage_sums_every_token_and_cost(ledger: Ledger, run_id: str) -> None:
    data = payload()
    before = ledger.spent_since(datetime(2000, 1, 1).astimezone())
    ledger.record_call(data, run_id, "c" * 64)
    ledger.record_call(data, run_id, "c" * 64)
    # Another run's call must not count toward this run's token ceiling.
    ledger.record_call(data, run_id + "-other", "c" * 64)
    expected_tokens = 2 * sum(
        usage["inputTokens"]
        + usage["outputTokens"]
        + usage["cacheReadInputTokens"]
        + usage["cacheCreationInputTokens"]
        for usage in data["modelUsage"].values()
    )
    assert ledger.tokens_for_run(run_id) == expected_tokens
    spent = ledger.spent_since(datetime(2000, 1, 1).astimezone()) - before
    # Nothing was spent after an instant that has not happened yet.
    assert ledger.spent_since(datetime.now(UTC) + timedelta(days=1)) == 0
    expected_cost = 3 * sum(
        Decimal(str(usage["costUSD"])) for usage in data["modelUsage"].values()
    )
    # Postgres stores each row's cost to six decimals, so each row may be off by
    # half a millionth; the in-memory ledger keeps full precision.
    rows = 3 * len(data["modelUsage"])
    assert abs(spent - expected_cost) <= rows * Decimal("0.0000005")


def check_forecasts_and_exclusions_read_back_by_version(ledger: Ledger, version: str) -> None:
    ledger.write_forecast("claude-sonnet-5", version, [row("f-1", 0.25), row("f-2", 0.75)])
    ledger.write_forecast("claude-sonnet-5", "other-" + version, [row("f-3", 0.5)])
    day = date(2026, 7, 2)
    ledger.write_exclusion(Exclusion(day, 7, "7" * 64, version, "refused", 0, ("f-4",)))
    ledger.write_exclusion(Exclusion(day, 8, "8" * 64, version, "refused", 1, ("f-5",)))
    assert ledger.forecasts(version) == {"f-1": 0.25, "f-2": 0.75}
    assert ledger.exclusion_counts(version) == {"refused": 2}


def check_an_exclusion_is_written_once_per_version(ledger: Ledger, version: str) -> None:
    day, group = date(2026, 7, 3), "9" * 64
    ledger.write_exclusion(Exclusion(day, 9, group, version, "refused", 0, ("g-1",)))
    with pytest.raises(Exception):  # noqa: B017 - each ledger raises its own type
        ledger.write_exclusion(Exclusion(day, 9, group, version, "invalid_output", 1, ("g-1",)))
    assert ledger.exclusion_counts(version) == {"refused": 1}
    # A new prompt version is a new arm configuration: it may exclude the same group.
    ledger.write_exclusion(Exclusion(day, 9, group, "other-" + version, "refused", 0, ("g-1",)))
    assert ledger.is_excluded(version, group)
    assert ledger.is_excluded("other-" + version, group)
