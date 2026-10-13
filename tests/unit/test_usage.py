import json
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.llm.usage import (
    ClaudeRunUsage,
    ModelUsage,
    parse_claude_run,
    parse_model_usage,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "claude_run.json"


def _payload() -> dict:
    return json.loads(FIXTURE.read_text())


def test_parses_a_real_captured_payload():
    """The parser must work against output a real run produced, not a mock."""
    usage = parse_claude_run(_payload())
    assert isinstance(usage, ClaudeRunUsage)
    assert usage.session_id
    assert usage.total_cost_usd > Decimal("0")
    assert usage.input_tokens > 0
    assert usage.cache_read_tokens > 0


def test_cost_is_decimal_not_float():
    """Money is never a float: rounding error would accumulate across calls."""
    usage = parse_claude_run(_payload())
    assert isinstance(usage.total_cost_usd, Decimal)


def test_cost_keeps_full_precision_from_the_payload():
    payload = _payload()
    assert parse_claude_run(payload).total_cost_usd == Decimal(str(payload["total_cost_usd"]))


def test_missing_cost_field_raises_rather_than_defaulting_to_zero():
    payload = _payload()
    del payload["total_cost_usd"]
    with pytest.raises(ValueError, match="total_cost_usd"):
        parse_claude_run(payload)


def test_missing_usage_block_raises():
    payload = _payload()
    del payload["usage"]
    with pytest.raises(ValueError, match="usage"):
        parse_claude_run(payload)


def test_missing_token_counts_raise():
    payload = _payload()
    del payload["usage"]["input_tokens"]
    with pytest.raises(ValueError, match="input_tokens"):
        parse_claude_run(payload)


def test_absent_cache_reads_are_zero_not_an_error():
    """A run that read nothing from cache legitimately omits the counter."""
    payload = _payload()
    del payload["usage"]["cache_read_input_tokens"]
    assert parse_claude_run(payload).cache_read_tokens == 0


def test_one_usage_record_per_model_in_the_run():
    """A single run can span several models, each billed separately."""
    records = parse_model_usage(_payload())
    assert len(records) == len(_payload()["modelUsage"])
    assert all(isinstance(record, ModelUsage) for record in records)
    assert {record.model_id for record in records} == set(_payload()["modelUsage"])


def test_model_records_carry_their_own_tokens_and_cost():
    records = {record.model_id: record for record in parse_model_usage(_payload())}
    for model_id, reported in _payload()["modelUsage"].items():
        record = records[model_id]
        assert record.input_tokens == reported["inputTokens"]
        assert record.output_tokens == reported["outputTokens"]
        assert record.cache_read_tokens == reported["cacheReadInputTokens"]
        assert record.cost_usd == Decimal(str(reported["costUSD"]))


def test_per_model_costs_account_for_the_run_total():
    """Per-model costs must reconcile with the run total, or attribution is wrong."""
    payload = _payload()
    per_model = sum(record.cost_usd for record in parse_model_usage(payload))
    assert abs(per_model - Decimal(str(payload["total_cost_usd"]))) < Decimal("0.000001")


def test_missing_model_usage_block_raises():
    payload = _payload()
    del payload["modelUsage"]
    with pytest.raises(ValueError, match="modelUsage"):
        parse_model_usage(payload)


def test_cache_writes_are_parsed_per_model() -> None:
    """Most of a headless call's input arrives as cache writes; dropping them
    would meter a fraction of a percent of what the call read."""
    payload = json.loads(FIXTURE.read_text())
    records = {record.model_id: record for record in parse_model_usage(payload)}
    for model_id, reported in payload["modelUsage"].items():
        assert records[model_id].cache_creation_tokens == reported["cacheCreationInputTokens"]
    # The premise: the fixture holds a model whose cache writes dominate its input.
    assert any(
        reported["cacheCreationInputTokens"] > 100 * reported["inputTokens"]
        for reported in payload["modelUsage"].values()
    )


def test_total_tokens_counts_every_kind_of_token() -> None:
    payload = json.loads(FIXTURE.read_text())
    for record in parse_model_usage(payload):
        reported = payload["modelUsage"][record.model_id]
        assert record.total_tokens == (
            reported["inputTokens"]
            + reported["outputTokens"]
            + reported["cacheReadInputTokens"]
            + reported["cacheCreationInputTokens"]
        )


@pytest.mark.parametrize(
    "field",
    [
        "inputTokens",
        "outputTokens",
        "cacheReadInputTokens",
        "cacheCreationInputTokens",
        "costUSD",
    ],
)
def test_a_missing_per_model_field_raises_naming_it(field: str):
    payload = _payload()
    del next(iter(payload["modelUsage"].values()))[field]
    with pytest.raises(ValueError, match=field):
        parse_model_usage(payload)


def test_a_run_reporting_no_per_model_usage_raises():
    # A call that ran but reported no usage would otherwise be metered as nothing.
    payload = _payload()
    payload["modelUsage"] = {}
    with pytest.raises(ValueError, match="modelUsage"):
        parse_model_usage(payload)
