"""A persistent embedded Postgres for running the ledger on one machine.

The ledger's guarantees — rows that cannot be edited, duplicate forecasts that
cannot be written — are enforced by Postgres triggers and constraints, so a
local run needs real Postgres, not a substitute engine. `pgserver` ships the
server as a Python package, which keeps the requirement to `uv sync --extra
ledger` on a machine without Docker.
"""

from __future__ import annotations

from pathlib import Path

#: Where `arbiter ledger-up` keeps the database files unless told otherwise.
DEFAULT_LEDGER_DIR = Path("data/ledger")


def ensure_local_ledger(data_dir: Path) -> str:
    """Start the ledger server for `data_dir`, or attach to it, and return its URL.

    The server outlives this process, and the URL is stable for a given
    directory. It does not survive a reboot: after one, run `ledger-up` again
    before any command that reads `DATABASE_URL`, since nothing else starts it.

    Raises:
        ModuleNotFoundError: the `ledger` extra is not installed.
    """
    from pgserver.postgres_server import get_server

    data_dir.mkdir(parents=True, exist_ok=True)
    server = get_server(data_dir, cleanup_mode=None)
    return server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)


def upgrade_to_head() -> None:
    """Apply every migration to the database the settings point at."""
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parents[3]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    command.upgrade(config, "head")
