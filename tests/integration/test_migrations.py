import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import inspect

from arbiter.db.session import get_engine

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.integration


def test_migrations_create_the_ledger_tables(migrated_database: str) -> None:
    """`alembic upgrade head` must produce every table the ledger needs."""
    tables = set(inspect(get_engine()).get_table_names())
    assert {"predictions", "llm_calls"} <= tables


def test_autogenerate_finds_the_migrated_schema_complete(migrated_database: str) -> None:
    """`alembic check` against a migrated database must propose no changes.

    It runs `migrations/env.py` exactly as `revision --autogenerate` does, so it
    fails if the models drift from the migrations, and also if `env.py` hands
    Alembic an incomplete target schema — in which case autogenerate would
    propose dropping every ledger table. It runs in a fresh process because
    table registration is process-wide: any earlier test that imported the
    models would complete the schema here and hide a missing import in `env.py`.
    """
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(REPO_ROOT / "alembic.ini"), "check"],
        cwd=REPO_ROOT,
        env={**os.environ, "DATABASE_URL": migrated_database},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
