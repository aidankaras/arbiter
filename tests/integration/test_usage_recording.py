import json
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import func, select

from arbiter.db.models import LlmCall
from arbiter.db.session import session_scope
from arbiter.llm.usage import record_run

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).parents[1] / "fixtures" / "claude_run.json"


def test_a_run_writes_one_ledger_row_per_model(migrated_database: str) -> None:
    payload = json.loads(FIXTURE.read_text())

    written = record_run(payload, run_id="2026-09-15-nightly", agent_name="research-analyst")

    assert written == len(payload["modelUsage"])
    with session_scope() as session:
        rows = (
            session.execute(select(LlmCall).where(LlmCall.run_id == "2026-09-15-nightly"))
            .scalars()
            .all()
        )
    assert {row.model_id for row in rows} == set(payload["modelUsage"])
    assert all(row.agent_name == "research-analyst" for row in rows)


def test_recorded_cost_reconciles_with_the_run_total(migrated_database: str) -> None:
    """The ledger must account for the whole bill, not a subset of it."""
    payload = json.loads(FIXTURE.read_text())

    record_run(payload, run_id="2026-09-15-reconcile")

    with session_scope() as session:
        total = session.execute(
            select(func.sum(LlmCall.cost_usd)).where(LlmCall.run_id == "2026-09-15-reconcile")
        ).scalar_one()
    assert abs(total - Decimal(str(payload["total_cost_usd"]))) < Decimal("0.000001")
