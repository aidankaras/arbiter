"""One analyst sample: a headless Claude Code call, and what it returned.

The model is run with no tools but the one that returns its structured answer,
no MCP servers, no settings or plugins from any project, and no session saved.
Those conditions are checked on every call, from the call's own startup event,
rather than assumed from the flags: a forecast made by a model that could have
fetched something is not a forecast from the packet.

Claude Code 2.1.296 still adds a short block of its own to every call: the
account's email, the working directory and platform, the model's name, and
today's date. No flag that keeps the subscription login removes it
(`--exclude-dynamic-system-prompt-sections` is ignored with `--system-prompt`,
and `--bare` ignores the login), so the template tells the model that the date
is when the archive is processed, not when the filings were made. None of it
carries a price or an outcome.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, cast

from arbiter.arms.agent.prompt import EFFORT, MODEL_ID, OUTPUT_SCHEMA

#: The only tool the model may hold: the one it answers the schema with.
PERMITTED_TOOLS = ["StructuredOutput"]
#: Plugins Claude Code 2.1.296 loads by itself, observed in a recorded call. Any
#: other plugin, or a change to these after an upgrade, stops the run until the
#: live test is rerun and this list is reviewed.
PERMITTED_PLUGINS = frozenset(
    {
        "cc-plugin-agents-md@builtin",
        "cc-plugin-telemetry@builtin",
        "cc-plugin-plugin-authoring@builtin",
    }
)
#: What the call inherits from the caller's environment. Anything else, such as
#: an API key, a different endpoint or configuration directory, or the variables
#: of a Claude Code session the run was started from, could change what the
#: model is, sees or bills without appearing in the startup event.
INHERITED_ENVIRONMENT = ("PATH", "HOME", "LANG", "USER", "TMPDIR")
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


def _mapping(value: object) -> dict[str, Any]:
    """`value` as a JSON object, or empty when the stream put something else there."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _require_isolation(init: Mapping[str, Any]) -> None:
    problems: list[str] = []
    if init.get("tools") != PERMITTED_TOOLS:
        problems.append(f"tools {init.get('tools')!r}")
    if init.get("mcp_servers"):
        problems.append(f"MCP servers {init.get('mcp_servers')!r}")
    if init.get("model") != MODEL_ID:
        problems.append(f"model {init.get('model')!r}")
    for name in ("skills", "slash_commands"):
        if init.get(name):
            problems.append(f"{name} {init.get(name)!r}")
    if init.get("output_style") != "default":
        problems.append(f"output style {init.get('output_style')!r}")
    if init.get("permissionMode") != "default":
        problems.append(f"permission mode {init.get('permissionMode')!r}")
    # "none" is the subscription login; anything else is a key billed per token.
    if init.get("apiKeySource") != "none":
        problems.append(f"API key source {init.get('apiKeySource')!r}")
    listed: object = init.get("plugins")
    entries = cast("list[object]", listed) if isinstance(listed, list) else []
    plugins = frozenset(str(_mapping(entry).get("source")) for entry in entries)
    if plugins != PERMITTED_PLUGINS:
        problems.append(f"plugins {sorted(plugins)!r}")
    memory = _mapping(init.get("memory_paths")).get("auto")
    if memory and Path(memory).is_dir() and any(Path(memory).iterdir()):
        problems.append(f"memory loaded from {memory}")
    if problems:
        msg = "the call did not run isolated: " + ", ".join(problems)
        raise EnvironmentViolationError(msg)


def _utilization(events: list[dict[str, Any]]) -> dict[str, float]:
    """Each usage window's reported utilization, skipping any it could not read.

    Informational only: the shape is undocumented, and a change to it must not
    void a call whose answer and usage are intact.
    """
    windows: dict[str, float] = {}
    for event in events:
        if event.get("type") != "rate_limit_event":
            continue
        info = _mapping(event.get("rate_limit_info"))
        for name, window in _mapping(info.get("unifiedWindows")).items():
            value = _mapping(window).get("utilization")
            if isinstance(value, int | float):
                windows[name] = float(value)
    return windows


def _rate_limited(events: list[dict[str, Any]], result: Mapping[str, Any]) -> bool:
    """Whether a failed call failed because a usage window is exhausted.

    Only consulted for a call that failed. A successful call can carry a
    `rate_limit_event` whose status is `allowed_warning` (observed live, nearing
    a window's limit); that is a warning, not a failure, and its answer stands.
    """
    rejected = any(
        event.get("type") == "rate_limit_event"
        and _mapping(event.get("rate_limit_info")).get("status") == "rejected"
        for event in events
    )
    return rejected or result.get("api_error_status") == 429


def _decode(stdout: str) -> list[dict[str, Any]]:
    """Every event on stdout, skipping lines that are not a JSON object.

    Per line, so a stray warning or a line cut off by a killed process costs
    that line rather than the call's result and its usage.
    """
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(cast("dict[str, Any]", event))
    return events


def parse_stream(stdout: str, returncode: int, stderr: str = "") -> Sample:
    """Classify one call from its `stream-json` output.

    Raises:
        EnvironmentViolationError: the call's startup event shows tools, MCP
            servers or a model other than the permitted ones, or its usage
            names a model other than the pinned one.
    """
    events = _decode(stdout)
    init = next(
        (e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), None
    )
    if init is not None:
        _require_isolation(init)
    result = next((e for e in events if e.get("type") == "result"), None)
    windows = _utilization(events)
    if result is None or init is None:
        # A result without a startup event cannot be shown to have run isolated,
        # so its answer is not used; its usage is still returned for metering.
        detail = (
            f"exit {returncode}, no {'result' if init else 'startup'} event: {stderr[-500:]}"
        )
        return Sample(Status.ERROR, payload=result, utilization=windows, detail=detail)

    models = set(_mapping(result.get("modelUsage")))
    if models and models != {MODEL_ID}:
        msg = f"the call's usage names {sorted(models)}, not only {MODEL_ID}"
        raise EnvironmentViolationError(msg)

    if result.get("is_error") or returncode != 0:
        if _rate_limited(events, result):
            return Sample(Status.RATE_LIMITED, payload=result, utilization=windows)
        detail = (
            f"exit {returncode}, {result.get('subtype')}: {str(result.get('result'))[:500]}"
            + (f"; {stderr[-300:]}" if stderr else "")
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
    if _rate_limited(events, result):
        # No usable answer and an exhausted window: an outage, not the evidence.
        return Sample(Status.RATE_LIMITED, payload=result, utilization=windows, detail=shown)
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
                    env={k: os.environ[k] for k in INHERITED_ENVIRONMENT if k in os.environ},
                    check=False,
                )
            except subprocess.TimeoutExpired as timeout:
                # The call may have finished and been billed before it hung:
                # classify what it printed so its usage is still metered.
                # Captured output on a timeout is bytes even with `text=True`.
                printed = (timeout.stdout or b"").decode()
                return parse_stream(printed, -1, f"timed out after {self.timeout_seconds}s")
        return parse_stream(completed.stdout, completed.returncode, completed.stderr)
