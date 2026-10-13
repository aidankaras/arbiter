from datetime import date
from pathlib import Path

from typer.testing import CliRunner

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
    assert "no packet days" in result.output
    assert not ledger_dir.exists()


def test_agent_days_are_the_built_days_in_range(tmp_path: Path) -> None:
    insider = tmp_path / "insider"
    insider.mkdir()
    for name in ("2026-06-30", "2026-07-01", "2026-07-02", "2026-09-10"):
        (insider / f"{name}.jsonl.gz").write_bytes(b"")
    assert _agent_days(tmp_path, date(2026, 7, 1), date(2026, 9, 9)) == [
        date(2026, 7, 1),
        date(2026, 7, 2),
    ]
