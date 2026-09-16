#!/usr/bin/env python3
"""Restrict forecasting runs to their evidence packet.

Forecasting arms read a frozen evidence packet and nothing else. Enforcing that
here makes the boundary structural: an arm cannot reach past its packet even if
its prompt is rewritten, truncated, or ignored. That boundary is what makes a
comparison between arms meaningful, so it does not rest on instructions.

The policy is deny-by-default. A denylist of retrieval tools was tried first and
failed in practice: a run reached the network through an unrelated MCP server
whose name the list did not anticipate. Only tools on the allowlist below may
run, so a newly installed server cannot silently widen what an arm can see.

The guard applies only when ARBITER_FORECAST_RUN is set, leaving ordinary
development sessions unaffected.
"""

from __future__ import annotations

import json
import os
import re
import sys

# Local, non-retrieving tools an arm legitimately needs: reading its packet from
# disk, searching within it, and writing its own prediction out.
ALLOWED_TOOLS = frozenset(
    {"Read", "Glob", "Grep", "Write", "Edit", "NotebookRead", "TodoWrite", "Bash"}
)

# The packet server serves frozen, point-in-time evidence and cannot fetch.
ALLOWED_MCP_PREFIX = "mcp__arbiter_packet__"

NETWORK_COMMANDS = ("curl", "wget", "nc", "ncat", "telnet", "ssh", "scp", "sftp")
_COMMAND_PATTERN = re.compile(rf"\b({'|'.join(NETWORK_COMMANDS)})\b")


def deny(reason: str) -> None:
    """Emit a deny decision and exit successfully.

    Exiting zero is deliberate: the decision travels in the payload, and a
    non-zero exit would report a hook malfunction rather than a policy decision.
    """
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.exit(0)


def main() -> None:
    """Inspect one PreToolUse payload and deny anything outside the allowlist."""
    if os.environ.get("ARBITER_FORECAST_RUN") != "1":
        sys.exit(0)

    payload = json.load(sys.stdin)
    tool_name = str(payload.get("tool_name", ""))

    if tool_name.startswith(ALLOWED_MCP_PREFIX):
        sys.exit(0)

    if tool_name not in ALLOWED_TOOLS:
        deny(
            f"{tool_name!r} is unavailable during a forecasting run, which may use "
            "only its frozen evidence packet. Every arm must see identical inputs "
            "for their comparison to mean anything."
        )

    if tool_name == "Bash":
        command = str(payload.get("tool_input", {}).get("command", ""))
        match = _COMMAND_PATTERN.search(command)
        if match:
            deny(
                f"The command uses {match.group(1)!r}, which can reach the network. "
                "Forecasting runs may not retrieve anything beyond their packet."
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
