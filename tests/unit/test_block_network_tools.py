import json
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).parents[2] / ".claude" / "hooks" / "block_network_tools.py"


def run_hook(payload: dict, *, forecast_run: bool) -> dict:
    env = {"PATH": "/usr/bin:/bin"}
    if forecast_run:
        env["ARBITER_FORECAST_RUN"] = "1"
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout) if result.stdout.strip() else {}


def test_webfetch_is_denied_during_a_forecast_run():
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "WebFetch",
            "tool_input": {"url": "https://example.com"},
        },
        forecast_run=True,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_websearch_is_denied_during_a_forecast_run():
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "WebSearch",
            "tool_input": {"query": "earnings surprise"},
        },
        forecast_run=True,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_networked_bash_command_is_denied():
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "curl https://www.sec.gov/cgi-bin/browse-edgar"},
        },
        forecast_run=True,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "curl" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_ordinary_bash_command_is_not_denied():
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "python scripts/build_packets.py"},
        },
        forecast_run=True,
    )
    assert out == {}


def test_a_word_containing_a_tool_name_is_not_denied():
    """`curliness` is not `curl`; matching must respect word boundaries."""
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "echo curliness"},
        },
        forecast_run=True,
    )
    assert out == {}


def test_reading_a_file_is_never_denied():
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "/tmp/packet.json"},
        },
        forecast_run=True,
    )
    assert out == {}


def test_outside_a_forecast_run_nothing_is_denied():
    """Development sessions keep their tools; the ban scopes to forecasting."""
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "WebFetch",
            "tool_input": {"url": "https://example.com"},
        },
        forecast_run=False,
    )
    assert out == {}
