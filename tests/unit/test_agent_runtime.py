"""Classifying one headless call, from a real recorded stream and edits of it.

`stream_ok.jsonl` is a recorded `claude -p --output-format stream-json` call to
claude-sonnet-5 (Claude Code 2.1.296), with machine paths replaced. Every
failure case below is that recording with one field changed, and each test
asserts its edit took before asserting the outcome. The refusal and the
rate-limit shapes are inferred from the recorded fields, not recorded
themselves: a call was never refused or limited while this was written.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from arbiter.arms.agent.runtime import (
    ClaudeCodeRuntime,
    EnvironmentViolationError,
    Status,
    claude_command,
    parse_stream,
)

OK = (Path(__file__).parents[1] / "fixtures" / "agent" / "stream_ok.jsonl").read_text()


def _edit(kind: str, change: Callable[[dict[str, Any]], None]) -> str:
    """Return the recording with `change` applied to every event of `kind`."""
    lines = []
    for line in OK.splitlines():
        event = json.loads(line)
        if event.get("type") == kind or (kind == "init" and event.get("subtype") == "init"):
            change(event)
        lines.append(json.dumps(event))
    edited = "\n".join(lines)
    assert edited != "\n".join(json.dumps(json.loads(line)) for line in OK.splitlines())
    return edited


def _answer(answer: object) -> str:
    def change(event: dict[str, Any]) -> None:
        event["structured_output"] = answer

    return _edit("result", change)


def test_the_recorded_call_is_a_valid_sample() -> None:
    sample = parse_stream(OK, 0)
    assert sample.status is Status.OK
    assert sample.p_up == 0.5
    assert sample.payload is not None and "modelUsage" in sample.payload
    assert sample.utilization == {"five_hour": 0.32, "seven_day": 0.24}


@pytest.mark.parametrize(
    "answer",
    [
        {"p_up": 1.4, "rationale": "too sure"},
        {"p_up": -0.1, "rationale": "below zero"},
        {"p_up": True, "rationale": "a boolean is not a probability"},
        {"p_up": "0.6", "rationale": "a string is not a number"},
        {"p_up": 0.6, "rationale": "x" * 601},
        {"rationale": "no probability"},
        None,
    ],
)
def test_an_answer_outside_the_schema_is_invalid(answer: object) -> None:
    sample = parse_stream(_answer(answer), 0)
    assert sample.status is Status.INVALID
    assert sample.payload is not None  # still metered


@pytest.mark.parametrize("p_up", [0, 1, 0.0, 1.0])
def test_the_bounds_of_a_probability_are_valid(p_up: float) -> None:
    assert parse_stream(_answer({"p_up": p_up, "rationale": "edge"}), 0).status is Status.OK


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tools", ["StructuredOutput", "WebFetch"]),
        ("tools", []),
        ("mcp_servers", [{"name": "packet", "status": "connected"}]),
        ("model", "claude-sonnet-5-5"),
        # Each of these can put text in the context without adding a tool.
        ("skills", ["market-insight"]),
        ("slash_commands", ["review"]),
        ("output_style", "custom"),
        (
            "plugins",
            [{"name": "notes", "path": "/home/u/.claude/plugins/notes", "source": "notes@x"}],
        ),
        ("permissionMode", "bypassPermissions"),
        # An API key bills per token; the study runs on the subscription only.
        ("apiKeySource", "ANTHROPIC_API_KEY"),
    ],
)
def test_a_call_that_was_not_isolated_stops_the_run(field: str, value: object) -> None:
    def change(event: dict[str, Any]) -> None:
        event[field] = value

    with pytest.raises(EnvironmentViolationError):
        parse_stream(_edit("init", change), 0)


def test_usage_from_a_second_model_stops_the_run() -> None:
    def change(event: dict[str, Any]) -> None:
        event["modelUsage"]["claude-haiku-4-5"] = event["modelUsage"]["claude-sonnet-5"]

    with pytest.raises(EnvironmentViolationError):
        parse_stream(_edit("result", change), 0)


def test_a_refusal_is_classified_as_one() -> None:
    def change(event: dict[str, Any]) -> None:
        event["safety_stops"] = 1
        event["structured_output"] = None

    assert parse_stream(_edit("result", change), 0).status is Status.REFUSED


def test_a_warning_about_a_nearly_used_window_does_not_void_a_good_answer() -> None:
    # `allowed_warning` was observed on a live, successful call.
    def change(event: dict[str, Any]) -> None:
        event["rate_limit_info"]["status"] = "allowed_warning"

    assert parse_stream(_edit("rate_limit_event", change), 0).status is Status.OK


def test_a_call_failed_by_an_exhausted_window_is_rate_limited() -> None:
    edited = _edit(
        "rate_limit_event",
        lambda event: event["rate_limit_info"].update(status="rejected"),
    )
    lines = []
    for line in edited.splitlines():
        event = json.loads(line)
        if event.get("type") == "result":
            event.update(
                is_error=True, subtype="error_during_execution", structured_output=None
            )
        lines.append(json.dumps(event))
    assert parse_stream("\n".join(lines), 1).status is Status.RATE_LIMITED


def test_a_failure_while_merely_warned_is_an_error_not_a_limit() -> None:
    """The halt message must name the real cause: a warning is not an exhausted window."""
    edited = _edit(
        "rate_limit_event",
        lambda event: event["rate_limit_info"].update(status="allowed_warning"),
    )
    lines = []
    for line in edited.splitlines():
        event = json.loads(line)
        if event.get("type") == "result":
            event.update(
                is_error=True, subtype="error_during_execution", structured_output=None
            )
        lines.append(json.dumps(event))
    assert parse_stream("\n".join(lines), 1).status is Status.ERROR


def test_a_429_from_the_api_is_rate_limited() -> None:
    def change(event: dict[str, Any]) -> None:
        event.update(is_error=True, api_error_status=429, structured_output=None)

    assert parse_stream(_edit("result", change), 1).status is Status.RATE_LIMITED


def test_an_errored_result_is_an_error() -> None:
    def change(event: dict[str, Any]) -> None:
        event["is_error"] = True
        event["subtype"] = "error_during_execution"

    assert parse_stream(_edit("result", change), 1).status is Status.ERROR


def test_a_process_that_printed_nothing_is_an_error() -> None:
    sample = parse_stream("", 1, "claude: command failed")
    assert sample.status is Status.ERROR
    assert sample.payload is None
    assert "command failed" in sample.detail


def test_the_command_grants_no_tools_and_never_uses_bare_mode() -> None:
    argv = claude_command("SYSTEM")
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--bare" not in argv  # bare mode ignores subscription sign-in
    assert argv[-2:] == ["--system-prompt", "SYSTEM"]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5"


def test_a_memory_directory_with_content_stops_the_run(tmp_path: Path) -> None:
    (tmp_path / "MEMORY.md").write_text("baseline IC was +0.029")

    def change(event: dict[str, Any]) -> None:
        event["memory_paths"] = {"auto": str(tmp_path)}

    with pytest.raises(EnvironmentViolationError, match="memory"):
        parse_stream(_edit("init", change), 0)


def test_a_truncated_stream_is_an_error_not_a_crash() -> None:
    # A process killed mid-write leaves half a line; one call must not abort a run.
    truncated = OK[:-40]
    with pytest.raises(json.JSONDecodeError):
        json.loads(truncated.splitlines()[-1])
    assert parse_stream(truncated, -9).status is Status.ERROR


def test_a_stray_line_on_stdout_does_not_void_a_good_call() -> None:
    sample = parse_stream(OK + '\nnot json at all\n"a bare string"\n', 0)
    assert sample.status is Status.OK
    assert sample.payload is not None


def test_a_result_without_a_startup_event_is_an_error_that_is_still_metered() -> None:
    lines = [line for line in OK.splitlines() if json.loads(line).get("subtype") != "init"]
    assert len(lines) == len(OK.splitlines()) - 1
    sample = parse_stream("\n".join(lines), 0)
    assert sample.status is Status.ERROR
    assert sample.payload is not None and "modelUsage" in sample.payload


def test_a_failed_exit_with_a_valid_answer_is_an_error() -> None:
    sample = parse_stream(OK, 1)
    assert sample.status is Status.ERROR
    assert sample.payload is not None


@pytest.mark.parametrize(
    "info", [None, {"status": "allowed", "unifiedWindows": {"five_hour": {}}}]
)
def test_a_malformed_usage_window_does_not_void_a_good_call(info: object) -> None:
    def change(event: dict[str, Any]) -> None:
        event["rate_limit_info"] = info

    sample = parse_stream(_edit("rate_limit_event", change), 0)
    assert sample.status is Status.OK
    assert sample.utilization == {}


def _fake_claude(tmp_path: Path, body: str) -> str:
    script = tmp_path / "claude"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(0o755)
    return str(script)


def test_a_call_that_times_out_keeps_what_it_printed(tmp_path: Path) -> None:
    fixture = tmp_path / "stream.jsonl"
    fixture.write_text(OK)
    # Prints a complete, paid call and then hangs: its usage must still be metered.
    executable = _fake_claude(tmp_path, f"cat {fixture}\nexec sleep 5\n")
    sample = ClaudeCodeRuntime(executable, timeout_seconds=1).sample("system", "user")
    assert sample.status is Status.ERROR
    assert "timed out" in sample.detail
    assert sample.payload is not None and "modelUsage" in sample.payload


def test_the_call_does_not_inherit_the_callers_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_SESSION_ID",
    ):
        monkeypatch.setenv(name, "set-by-the-caller")
    fixture, seen = tmp_path / "stream.jsonl", tmp_path / "env.txt"
    fixture.write_text(OK)
    executable = _fake_claude(tmp_path, f"env > {seen}\ncat {fixture}\n")
    ClaudeCodeRuntime(executable).sample("system", "user")
    environment = seen.read_text()
    assert "HOME=" in environment
    assert "set-by-the-caller" not in environment
