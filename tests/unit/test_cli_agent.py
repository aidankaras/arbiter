from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from arbiter.arms.agent import run
from arbiter.arms.agent.run import RunReport
from arbiter.cli import _agent_days, app


def test_the_pilot_refuses_a_scored_day() -> None:
    result = CliRunner().invoke(app, ["agent-pilot", "2026-07-01"])
    assert result.exit_code == 2
    assert "scored window" in result.output


def test_the_pilot_accepts_the_last_development_day(tmp_path: Path) -> None:
    # Gets past the window check and stops at the missing packets, before the
    # ledger is started: a unit test must never launch a database server.
    ledger_dir = tmp_path / "ledger"
    result = CliRunner().invoke(
        app,
        ["agent-pilot", "2026-06-30", "--packet-root", str(tmp_path),
         "--ledger-dir", str(ledger_dir)],
    )  # fmt: skip
    assert result.exit_code == 2
    assert "2026-06-30" in result.output and "never built" in result.output
    assert not ledger_dir.exists()


def _built(root: Path, *days: str) -> None:
    insider = root / "insider"
    insider.mkdir(exist_ok=True)
    for name in days:
        (insider / f"{name}.jsonl.gz").write_bytes(b"")


def test_agent_days_are_every_trading_day_in_range(tmp_path: Path) -> None:
    _built(tmp_path, "2026-06-30", "2026-07-01", "2026-07-02", "2026-09-10")
    assert _agent_days(tmp_path, date(2026, 7, 1), date(2026, 7, 2)) == [
        date(2026, 7, 1),
        date(2026, 7, 2),
    ]


def test_a_trading_day_whose_packets_were_never_built_is_named_not_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Monday 07-13 to Friday 07-17, with Wednesday's build missing.
    _built(tmp_path, "2026-07-13", "2026-07-14", "2026-07-16", "2026-07-17")
    with pytest.raises(typer.Exit):
        _agent_days(tmp_path, date(2026, 7, 13), date(2026, 7, 17))
    assert "2026-07-15" in capsys.readouterr().err


def _stub_run(monkeypatch: pytest.MonkeyPatch, report: RunReport) -> None:
    settings = SimpleNamespace(daily_spend_ceiling_usd=1, per_run_token_ceiling=1)
    monkeypatch.setattr("arbiter.cli._local_ledger", lambda _: settings)
    monkeypatch.setattr(run, "run_days", lambda *_, **__: report)


@pytest.mark.parametrize(("halted", "code"), [(None, 0), ("rate_limited on issuer 1", 3)])
def test_a_halted_run_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, halted: str | None, code: int
) -> None:
    # A script that repeats the run until nothing is left must not read a halt
    # as a finished window.
    _built(tmp_path, "2026-06-30")
    _stub_run(monkeypatch, RunReport(halted=halted))
    result = CliRunner().invoke(
        app, ["agent-pilot", "2026-06-30", "--packet-root", str(tmp_path)]
    )
    assert result.exit_code == code
    if halted:
        assert halted in result.output


def test_agent_run_takes_max_calls_as_the_documented_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _built(tmp_path, "2026-07-01")
    _stub_run(monkeypatch, RunReport())
    result = CliRunner().invoke(
        app,
        [
            "agent-run",
            "--max-calls",
            "6",
            "--end",
            "2026-07-01",
            "--packet-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
