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


def test_an_mcp_tool_that_can_fetch_is_denied():
    """A retrieval tool reached through MCP is still retrieval."""
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "mcp__firecrawl__firecrawl_scrape",
            "tool_input": {"url": "https://example.com"},
        },
        forecast_run=True,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_the_packet_server_is_allowed():
    """Arms read their evidence through the packet server, which serves frozen data."""
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "mcp__arbiter_packet__get_filing_section",
            "tool_input": {"event_id": "0001234567-25-000001"},
        },
        forecast_run=True,
    )
    assert out == {}


def test_delegating_to_a_subagent_is_denied():
    """A subagent would run outside this hook's view of the tool call."""
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Task",
            "tool_input": {"prompt": "look up the filing"},
        },
        forecast_run=True,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_invoking_a_skill_is_denied():
    """A skill can wrap a retrieval tool, so the name alone proves nothing."""
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Skill",
            "tool_input": {"skill": "firecrawl:firecrawl-scrape"},
        },
        forecast_run=True,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_an_unrecognized_tool_is_denied_by_default():
    """New tools must be reviewed before an arm may use them."""
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "SomeFutureTool",
            "tool_input": {},
        },
        forecast_run=True,
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_writing_results_is_allowed():
    out = run_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Write",
            "tool_input": {"file_path": "/tmp/prediction.json"},
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
