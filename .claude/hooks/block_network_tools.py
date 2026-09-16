#!/usr/bin/env python3
"""Deny network-capable tools while a forecasting run is in progress.

Forecasting arms read a frozen evidence packet and nothing else. Enforcing that
here makes the boundary structural: an arm cannot reach past its packet even if
its prompt is rewritten, truncated, or ignored. The boundary is what makes a
comparison between arms meaningful, so it does not rest on instructions.

The guard applies only when ARBITER_FORECAST_RUN is set, so ordinary
development sessions keep their tools.
"""

from __future__ import annotations

import json
import os
import re
import sys

BLOCKED_TOOLS = frozenset({"WebFetch", "WebSearch"})
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
    """Inspect one PreToolUse payload and deny it if it can reach the network."""
    if os.environ.get("ARBITER_FORECAST_RUN") != "1":
        sys.exit(0)

    payload = json.load(sys.stdin)
    tool_name = payload.get("tool_name", "")

    if tool_name in BLOCKED_TOOLS:
        deny(
            f"{tool_name} is unavailable during a forecasting run. Arms read only "
            "the frozen evidence packet, so that every arm sees identical inputs."
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
