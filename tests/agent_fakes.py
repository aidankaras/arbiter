"""In-memory stand-ins for the agent arm's ledger and runtime.

The fake ledger honours the same contract as `PgLedger` — pinned by
`tests/agent_ledger_contract.py`, run against both — so a runner test passing
against it says something about the real database.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from arbiter.arms.agent.ledger import Exclusion, ForecastRow
from arbiter.llm.usage import parse_model_usage


class DuplicateWriteError(Exception):
    """Raised where Postgres would raise a unique-constraint violation."""


class FakeLedger:
    def __init__(self) -> None:
        self.forecasts_by_key: dict[tuple[str, str], tuple[str, ForecastRow]] = {}
        self.exclusions: dict[tuple[str, str], Exclusion] = {}
        self.calls: list[tuple[datetime, str, str, Mapping[str, Any]]] = []
        self.now = datetime(2026, 10, 12, 16, 0, tzinfo=UTC)

    def predicted(self, prompt_version: str, event_ids: Sequence[str]) -> set[str]:
        return {
            event_id
            for event_id in event_ids
            if (prompt_version, event_id) in self.forecasts_by_key
        }

    def is_excluded(self, prompt_version: str, issuer_day_hash: str) -> bool:
        return (prompt_version, issuer_day_hash) in self.exclusions

    def excluded_events(self, prompt_version: str, event_ids: Sequence[str]) -> set[str]:
        excluded = {
            event_id
            for (version, _), exclusion in self.exclusions.items()
            if version == prompt_version
            for event_id in exclusion.event_ids
        }
        return excluded & set(event_ids)

    def write_forecast(
        self, model_id: str, prompt_version: str, rows: Sequence[ForecastRow]
    ) -> None:
        keys = [(prompt_version, row.event_id) for row in rows]
        # All or nothing, as one database transaction would be.
        if len(set(keys)) != len(keys) or any(key in self.forecasts_by_key for key in keys):
            raise DuplicateWriteError(keys)
        for key, row in zip(keys, rows, strict=True):
            self.forecasts_by_key[key] = (model_id, row)

    def write_exclusion(self, exclusion: Exclusion) -> None:
        key = (exclusion.prompt_version, exclusion.issuer_day_hash)
        if key in self.exclusions:
            raise DuplicateWriteError(key)
        self.exclusions[key] = exclusion

    def record_call(self, payload: Mapping[str, Any], run_id: str, packet_hash: str) -> None:
        parse_model_usage(payload)  # refuse a payload the real ledger would refuse
        self.calls.append((self.now, run_id, packet_hash, payload))

    def spent_since(self, instant: datetime) -> Decimal:
        return sum(
            (
                record.cost_usd
                for at, _, _, payload in self.calls
                if at >= instant
                for record in parse_model_usage(payload)
            ),
            Decimal(0),
        )

    def forecasts(self, prompt_version: str) -> dict[str, float]:
        return {
            event_id: row.p_up
            for (version, event_id), (_, row) in self.forecasts_by_key.items()
            if version == prompt_version
        }

    def exclusion_counts(self, prompt_version: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for (version, _), exclusion in self.exclusions.items():
            if version == prompt_version:
                counts[exclusion.reason] = counts.get(exclusion.reason, 0) + 1
        return counts

    def tokens_for_run(self, run_id: str) -> int:
        return sum(
            record.total_tokens
            for _, run, _, payload in self.calls
            if run == run_id
            for record in parse_model_usage(payload)
        )
