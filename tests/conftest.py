"""Shared test fixtures.

Integration tests run against a real PostgreSQL instance rather than SQLite,
because the properties under test — trigger-enforced immutability, JSONB
columns, timezone-aware timestamps — do not exist in a substitute engine. The
server is an ephemeral embedded instance, so the suite needs no running service
and no credentials.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def _sec_identity() -> None:
    """Supply the contact string EDGAR requires, for every test in the suite.

    Configuration is validated at startup and fails fast when this is absent, so
    a test touching the ingestion layer would otherwise fail on configuration
    rather than on the behaviour it covers.
    """
    os.environ.setdefault("SEC_USER_AGENT", "arbiter-tests tests@example.com")


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """Start an ephemeral PostgreSQL server and yield its SQLAlchemy URL.

    The URL is also exported as DATABASE_URL so that application code reading
    configuration from the environment reaches this server rather than a
    developer's own database.
    """
    pgserver = pytest.importorskip(
        "pgserver", reason="embedded PostgreSQL is required for integration tests"
    )

    data_dir = Path(tempfile.mkdtemp(prefix="arbiter-pg-"))
    server = pgserver.get_server(data_dir)
    url = server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)

    os.environ["DATABASE_URL"] = url
    os.environ.setdefault("SEC_USER_AGENT", "arbiter-tests tests@example.com")

    try:
        yield url
    finally:
        server.cleanup()
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def migrated_database(database_url: str) -> str:
    """Apply every migration to the ephemeral server, then yield its URL.

    Cached settings and engines are reset first: both are memoized for the life
    of a process, and a value cached before this fixture ran would point at
    whatever DATABASE_URL held at import time.
    """
    from alembic import command
    from alembic.config import Config

    from arbiter.config import get_settings
    from arbiter.db.session import get_engine

    get_settings.cache_clear()
    get_engine.cache_clear()

    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    command.upgrade(config, "head")

    return database_url
