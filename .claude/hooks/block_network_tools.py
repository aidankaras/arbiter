#!/usr/bin/env python3
"""Restrict forecasting runs to their evidence packet.

Forecasting arms read a frozen evidence packet and nothing else. That boundary
is what makes a comparison between arms meaningful, so it is enforced here
rather than by instructions an arm may ignore or have rewritten.

Two properties matter more than the tool list itself.

**Deny by default.** Only tools that cannot reach outside the packet are
permitted. An earlier version allowed shell and file-writing tools and
denylisted network command names inside them; that is not a boundary. A shell
can fetch by other means, string-splitting defeats any name match, and a
file-writing tool can overwrite this hook and disarm the guard for the rest of
the run. Nothing that executes arbitrary code or writes files is permitted.

**Fail closed.** Any malformed payload, unreadable input, or unexpected error
denies the call. A hook that exits non-zero without a decision is treated as a
non-blocking error and the tool proceeds, so every failure path here must end in
an explicit denial.

The guard applies only when ARBITER_FORECAST_RUN is set, leaving ordinary
development sessions unaffected. The forecast entrypoint is responsible for
setting it and for asserting that a canary retrieval is denied before dispatching
any arm; an unarmed guard enforces nothing.
"""

from __future__ import annotations

import json
import os
import sys

# Tools that read local state and cannot reach beyond it. Shell execution and
# file writing are deliberately absent: either can defeat the boundary, and a
# file write can disarm this hook.
ALLOWED_TOOLS = frozenset({"Read", "Glob", "Grep", "NotebookRead", "TodoWrite"})

# The packet server serves frozen, point-in-time evidence and cannot fetch.
ALLOWED_MCP_PREFIX = "mcp__arbiter_packet__"

_ARMED = "ARBITER_FORECAST_RUN"


def deny(reason: str) -> None:
    """Emit a deny decision and exit successfully.

    Exiting zero is deliberate: the decision travels in the payload, and a
    non-zero exit would be read as a malfunctioning hook, which does not block.
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


def decide(payload: object) -> None:
    """Deny anything outside the allowlist. Never returns for a denied call."""
    if not isinstance(payload, dict):
        deny("hook received a malformed payload; denying by default")

    tool_name = payload.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        deny("hook received no tool name; denying by default")

    if tool_name.startswith(ALLOWED_MCP_PREFIX):
        sys.exit(0)

    if tool_name not in ALLOWED_TOOLS:
        deny(
            f"{tool_name!r} is unavailable during a forecasting run, which may use "
            "only its frozen evidence packet. Every arm must see identical inputs "
            "for their comparison to mean anything."
        )

    sys.exit(0)


def main() -> None:
    """Read one PreToolUse payload and decide, denying on any failure."""
    if os.environ.get(_ARMED) != "1":
        sys.exit(0)

    try:
        payload = json.load(sys.stdin)
    except Exception:
        deny("hook could not read its input; denying by default")

    try:
        decide(payload)
    except SystemExit:
        raise
    except Exception:
        deny("hook failed while evaluating the call; denying by default")


if __name__ == "__main__":
    main()
