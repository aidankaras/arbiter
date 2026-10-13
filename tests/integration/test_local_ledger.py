import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from typer.testing import CliRunner

from arbiter.cli import app
from arbiter.config import get_settings
from arbiter.db.local import ensure_local_ledger
from arbiter.db.session import get_engine

pytestmark = pytest.mark.integration


@pytest.fixture
def restore_database_url(migrated_database: str):
    """`ledger-up` repoints the process at a new server; put the suite's back."""
    yield
    os.environ["DATABASE_URL"] = migrated_database
    get_settings.cache_clear()
    get_engine.cache_clear()


def test_ledger_up_starts_a_migrated_server_with_a_stable_url(
    tmp_path: Path, restore_database_url: None
) -> None:
    import pgserver

    ledger_dir = tmp_path / "ledger"
    try:
        result = CliRunner().invoke(app, ["ledger-up", "--ledger-dir", str(ledger_dir)])
        assert result.exit_code == 0, result.output
        # Migration logging precedes it; the URL is always the last line.
        url = (
            result.output.strip()
            .splitlines()[-1]
            .removeprefix('DATABASE_URL="')
            .removesuffix('"')
        )
        tables = set(inspect(create_engine(url)).get_table_names())
        assert {"predictions", "llm_calls", "agent_exclusions"} <= tables
        # A second start attaches to the same server rather than a new one.
        assert ensure_local_ledger(ledger_dir) == url
    finally:
        server = pgserver.get_server(ledger_dir)
        # `ledger-up` created this directory's handle with cleanup_mode=None, and
        # pgserver returns that cached handle here, so a plain cleanup() would
        # leave the server running after the test.
        server.cleanup_mode = "stop"
        server.cleanup()
    assert not (ledger_dir / "postmaster.pid").exists(), "the test leaked a server"
