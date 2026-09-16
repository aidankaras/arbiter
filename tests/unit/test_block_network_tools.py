import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).parents[2] / ".claude" / "hooks" / "block_network_tools.py"


def run_hook(payload: object, *, forecast_run: bool = True, raw: str | None = None) -> dict:
    env = {"PATH": "/usr/bin:/bin"}
    if forecast_run:
        env["ARBITER_FORECAST_RUN"] = "1"
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=raw if raw is not None else json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, f"hook must exit 0 to be honoured: {result.stderr}"
    return json.loads(result.stdout) if result.stdout.strip() else {}


def denied(out: dict) -> bool:
    return out.get("hookSpecificOutput", {}).get("permissionDecision") == "deny"


def call(tool_name: str, tool_input: object = None) -> dict:
    return run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input if tool_input is not None else {},
        }
    )


# --- The boundary holds against deliberate evasion -------------------------


@pytest.mark.parametrize(
    "command",
    [
        "python3 -c \"import urllib.request; urllib.request.urlopen('https://sec.gov')\"",
        'cur""l https://www.sec.gov',
        "exec 3<>/dev/tcp/1.2.3.4/80",
        "git fetch origin",
        "pip install requests",
        "echo aGVsbG8= | base64 -d | sh",
    ],
)
def test_shell_execution_is_denied_however_it_is_written(command: str):
    """A shell can reach the network by means no name-matching list anticipates."""
    assert denied(call("Bash", {"command": command}))


@pytest.mark.parametrize(
    "path", [".claude/hooks/block_network_tools.py", ".claude/settings.json"]
)
def test_writing_to_the_guard_itself_is_denied(path: str):
    """A write that disarms the hook would unguard every later call in the run."""
    assert denied(call("Write", {"file_path": path, "content": "x"}))


def test_editing_files_is_denied():
    assert denied(call("Edit", {"file_path": "/tmp/x", "old_string": "a", "new_string": "b"}))


@pytest.mark.parametrize("tool", ["WebFetch", "WebSearch", "Task", "Skill", "ToolSearch"])
def test_retrieval_and_delegation_tools_are_denied(tool: str):
    assert denied(call(tool, {"url": "https://example.com"}))


def test_an_unrecognized_mcp_server_is_denied():
    assert denied(call("mcp__firecrawl__firecrawl_scrape", {"url": "https://example.com"}))


def test_an_unrecognized_tool_is_denied_by_default():
    assert denied(call("SomeFutureTool"))


# --- Failure paths deny rather than letting the call through ----------------


def test_malformed_json_denies():
    """A hook that raises exits non-zero, which does not block; it must deny instead."""
    assert denied(run_hook(None, raw="{not json"))


def test_a_null_tool_input_denies():
    assert denied(
        run_hook({"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": None})
    )


def test_a_payload_that_is_not_an_object_denies():
    assert denied(run_hook(["unexpected"]))


def test_a_missing_tool_name_denies():
    assert denied(run_hook({"hook_event_name": "PreToolUse", "tool_input": {}}))


def test_empty_input_denies():
    assert denied(run_hook(None, raw=""))


# --- What an arm may legitimately do ---------------------------------------


@pytest.mark.parametrize("tool", ["Read", "Glob", "Grep"])
def test_local_read_tools_are_permitted(tool: str):
    assert call(tool, {"file_path": "/tmp/packet.json"}) == {}


def test_the_packet_server_is_permitted():
    """Arms read evidence through the packet server, which serves frozen data."""
    assert call("mcp__arbiter_packet__get_filing_section", {"event_id": "x"}) == {}


# --- Scope -----------------------------------------------------------------


def test_outside_a_forecast_run_nothing_is_denied():
    """Development sessions keep their tools; the ban scopes to forecasting."""
    out = run_hook(
        {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": "ls"}},
        forecast_run=False,
    )
    assert out == {}
