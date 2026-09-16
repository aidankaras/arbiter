"""Usage accounting for Claude Code runs.

Every run's tokens and cost are persisted so that cost per decision is a
measured quantity rather than an estimate. Missing fields raise: a usage record
that silently reported zero cost would disable the spend ceiling that reads it.

A single run routinely spans more than one model, so usage is recorded per
model rather than per run. A flat per-run row would attribute an expensive
model's cost to whichever name the caller happened to pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from arbiter.db.models import LlmCall
from arbiter.db.session import session_scope


@dataclass(frozen=True)
class ClaudeRunUsage:
    """Token and cost totals for one `claude -p` invocation."""

    session_id: str
    total_cost_usd: Decimal
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int


@dataclass(frozen=True)
class ModelUsage:
    """Tokens and cost attributed to a single model within one run."""

    model_id: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: Decimal


def _require(payload: Mapping[str, Any], key: str) -> Any:
    """Return `payload[key]`, naming the field when it is absent.

    Raises:
        ValueError: the key is missing from the payload.
    """
    if key not in payload:
        msg = f"Claude run payload is missing the {key!r} field"
        raise ValueError(msg)
    return payload[key]


def _money(value: object) -> Decimal:
    """Convert a reported cost to `Decimal` without passing through binary float.

    The payload encodes costs as JSON numbers. Converting via `str` preserves the
    digits as reported, where `Decimal(float)` would introduce representation
    error that accumulates across thousands of calls.
    """
    return Decimal(str(value))


def parse_claude_run(payload: Mapping[str, Any]) -> ClaudeRunUsage:
    """Summarize one run's totals from a `--output-format json` payload.

    Raises:
        ValueError: a required field is absent.
    """
    usage = _require(payload, "usage")
    return ClaudeRunUsage(
        session_id=str(_require(payload, "session_id")),
        total_cost_usd=_money(_require(payload, "total_cost_usd")),
        input_tokens=int(_require(usage, "input_tokens")),
        output_tokens=int(_require(usage, "output_tokens")),
        cache_read_tokens=int(usage.get("cache_read_input_tokens", 0)),
    )


def parse_model_usage(payload: Mapping[str, Any]) -> list[ModelUsage]:
    """Break one run's usage down by model.

    Returns one record per model the run invoked, keyed by the exact model
    identifier the payload reports, so that cost attribution survives a change
    of model without editing call sites.

    Raises:
        ValueError: the per-model usage block is absent.
    """
    reported: Mapping[str, Mapping[str, Any]] = _require(payload, "modelUsage")
    return [
        ModelUsage(
            model_id=model_id,
            input_tokens=int(entry["inputTokens"]),
            output_tokens=int(entry["outputTokens"]),
            cache_read_tokens=int(entry["cacheReadInputTokens"]),
            cost_usd=_money(entry["costUSD"]),
        )
        for model_id, entry in reported.items()
    ]


def record_run(payload: Mapping[str, Any], run_id: str, agent_name: str | None = None) -> int:
    """Persist one run's usage to the ledger, one row per model.

    `run_id` identifies the pipeline run that dispatched the work, so that a
    night's total cost is a single query. Returns the number of rows written.

    Raises:
        ValueError: the payload is missing a required field.
    """
    session_id = str(_require(payload, "session_id"))
    records = parse_model_usage(payload)

    with session_scope() as session:
        for record in records:
            session.add(
                LlmCall(
                    run_id=run_id,
                    session_id=session_id,
                    model_id=record.model_id,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cache_read_tokens=record.cache_read_tokens,
                    cost_usd=record.cost_usd,
                    agent_name=agent_name,
                )
            )

    return len(records)
