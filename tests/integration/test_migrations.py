import pytest
from sqlalchemy import inspect

from arbiter.db.session import get_engine

pytestmark = pytest.mark.integration


def test_migrations_create_the_ledger_tables(migrated_database: str) -> None:
    """`alembic upgrade head` must produce every table the ledger needs."""
    tables = set(inspect(get_engine()).get_table_names())
    assert {"predictions", "llm_calls"} <= tables
