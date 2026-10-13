"""One analyst sample: a headless Claude Code call, and what it returned.

The model is run with no tools but the one that returns its structured answer,
no MCP servers, no settings or plugins from any project, and no session saved.
Those conditions are checked on every call, from the call's own startup event,
rather than assumed from the flags: a forecast made by a model that could have
fetched something is not a forecast from the packet.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, cast

from arbiter.arms.agent.prompt import EFFORT, MODEL_ID, OUTPUT_SCHEMA

#: The only tool the model may hold: the one it answers the schema with.
PERMITTED_TOOLS = ["StructuredOutput"]
RATIONALE_LIMIT = int(OUTPUT_SCHEMA["properties"]["rationale"]["maxLength"])


class Status(StrEnum):
    """How a sample ended. Only `OK` carries a forecast."""

    OK = "ok"
    #: The answer was missing or outside the schema.
    INVALID = "invalid_output"
    #: The model declined.
    REFUSED = "refused"
    #: The call failed for a reason unrelated to the issuer-day.
    ERROR = "error"
    #: A usage window is exhausted; nothing should be retried until it resets.
    RATE_LIMITED = "rate_limited"


class EnvironmentViolationError(RuntimeError):
    """Raised when a call ran with tools, servers, or a model it must not have.

    Not a failed sample: it means the isolation the study rests on did not
    hold, so the run stops rather than recording anything.
    """


@dataclass(frozen=True)
class Sample:
    """One call's outcome."""

    status: Status
    p_up: float | None = None
    rationale: str | None = None
    #: The final `result` event, which carries the usage to meter. Absent only
    #: when the process produced none.
    payload: Mapping[str, Any] | None = None
    #: Share of each usage window consumed, as the call last reported it.
    utilization: Mapping[str, float] = field(default_factory=dict[str, float])
    detail: str = ""


class Runtime(Protocol):
    """Anything that can draw one sample for a prompt."""

    def sample(self, system: str, user: str) -> Sample:
        """Run one independent call and report how it ended."""
        ...


def claude_command(system: str, executable: str = "claude") -> list[str]:
    """The exact command line for one isolated, tool-less, schema-bound call."""
    return [
        executable,
        "-p",
        "--model",
        MODEL_ID,
        "--effort",
        EFFORT,
        "--output-format",
        "stream-json",
        "--verbose",
        "--json-schema",
        json.dumps(OUTPUT_SCHEMA, sort_keys=True),
        "--tools",
        "",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--system-prompt",
        system,
    ]


def _require_isolation(init: Mapping[str, Any]) -> None:
    problems: list[str] = []
    if init.get("tools") != PERMITTED_TOOLS:
        problems.append(f"tools {init.get('tools')!r}")
    if init.get("mcp_servers"):
        problems.append(f"MCP servers {init.get('mcp_servers')!r}")
    if init.get("model") != MODEL_ID:
        problems.append(f"model {init.get('model')!r}")
    if problems:
        msg = "the call did not run isolated: " + ", ".join(problems)
        raise EnvironmentViolationError(msg)


def _utilization(events: list[dict[str, Any]]) -> dict[str, float]:
    windows: dict[str, float] = {}
    for event in events:
        if event.get("type") == "rate_limit_event":
            info = event.get("rate_limit_info", {})
            for name, window in info.get("unifiedWindows", {}).items():
                windows[name] = float(window["utilization"])
    return windows


def _rate_limited(events: list[dict[str, Any]], result: Mapping[str, Any]) -> bool:
    """Whether a failed call failed because a usage window is exhausted.

    Only consulted for a call that failed. A successful call can carry a
    `rate_limit_event` whose status is `allowed_warning` (observed live, nearing
    a window's limit); that is a warning, not a failure, and its answer stands.
    """
    rejected = any(
        event.get("type") == "rate_limit_event"
        and event.get("rate_limit_info", {}).get("status") == "rejected"
        for event in events
    )
    return rejected or result.get("api_error_status") == 429


def parse_stream(stdout: str, returncode: int, stderr: str = "") -> Sample:
    """Classify one call from its `stream-json` output.

    Raises:
        EnvironmentViolationError: the call's startup event shows tools, MCP
            servers or a model other than the permitted ones, or its usage
            names a model other than the pinned one.
    """
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    init = next(
        (e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None
    )
    if init is not None:
        _require_isolation(init)
    result = next((e for e in events if e.get("type") == "result"), None)
    windows = _utilization(events)
    if result is None or init is None:
        detail = (
            f"exit {returncode}, no {'result' if init else 'startup'} event: {stderr[-500:]}"
        )
        return Sample(Status.ERROR, utilization=windows, detail=detail)

    models = set(result.get("modelUsage", {}))
    if models and models != {MODEL_ID}:
        msg = f"the call's usage names {sorted(models)}, not only {MODEL_ID}"
        raise EnvironmentViolationError(msg)

    if result.get("is_error") or returncode != 0:
        if _rate_limited(events, result):
            return Sample(Status.RATE_LIMITED, payload=result, utilization=windows)
        detail = (
            f"exit {returncode}, {result.get('subtype')}: {str(result.get('result'))[:500]}"
        )
        return Sample(Status.ERROR, payload=result, utilization=windows, detail=detail)
    if result.get("safety_stops") or result.get("stop_reason") == "refusal":
        return Sample(Status.REFUSED, payload=result, utilization=windows)

    answer: object = result.get("structured_output")
    shown = repr(answer)[:500]
    fields = cast("dict[str, object]", answer) if isinstance(answer, dict) else {}
    p_up = fields.get("p_up")
    rationale = fields.get("rationale")
    if (
        isinstance(p_up, int | float)
        and not isinstance(p_up, bool)
        and 0.0 <= float(p_up) <= 1.0
        and isinstance(rationale, str)
        and len(rationale) <= RATIONALE_LIMIT
    ):
        return Sample(
            Status.OK,
            p_up=float(p_up),
            rationale=rationale,
            payload=result,
            utilization=windows,
        )
    return Sample(Status.INVALID, payload=result, utilization=windows, detail=shown)


class ClaudeCodeRuntime:
    """Draws samples from headless Claude Code on the machine's subscription."""

    def __init__(self, executable: str = "claude", timeout_seconds: float = 300.0) -> None:
        """Configure the executable and how long one call may run."""
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def sample(self, system: str, user: str) -> Sample:
        """Run one call in a fresh empty directory, so no project memory loads.

        Raises:
            EnvironmentViolationError: as `parse_stream`.
        """
        with tempfile.TemporaryDirectory(prefix="arbiter-agent-") as cwd:
            try:
                completed = subprocess.run(
                    claude_command(system, self.executable),
                    input=user,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    cwd=cwd,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return Sample(Status.ERROR, detail=f"timed out after {self.timeout_seconds}s")
        return parse_stream(completed.stdout, completed.returncode, completed.stderr)
