from datetime import date

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DatabaseError, IntegrityError

from arbiter.db.models import AgentExclusion, LlmCall
from arbiter.db.session import session_scope

pytestmark = pytest.mark.integration


def _exclusion(cik: int, reason: str = "invalid_output") -> AgentExclusion:
    return AgentExclusion(
        event_day=date(2026, 7, 1),
        cik=cik,
        issuer_day_hash=f"{cik:064d}",
        arm="agent",
        prompt_version="v-test-exclusions",
        reason=reason,
        samples_valid=1,
        event_ids=["0001234567-26-000001"],
        detail="two samples failed schema validation after a retry",
    )


def test_an_exclusion_is_written_once_and_reads_back(migrated_database: str) -> None:
    with session_scope() as session:
        session.add(_exclusion(cik=101))
    with session_scope() as session:
        stored = session.execute(
            select(AgentExclusion).where(AgentExclusion.cik == 101)
        ).scalar_one()
    assert stored.reason == "invalid_output"
    assert stored.event_ids == ["0001234567-26-000001"]


@pytest.mark.parametrize(
    ("cik", "statement"),
    [
        (202, "UPDATE agent_exclusions SET reason = 'edited'"),
        (203, "DELETE FROM agent_exclusions"),
    ],
)
def test_exclusions_refuse_edits(migrated_database: str, cik: int, statement: str) -> None:
    # Each case writes its own issuer-day: a shared one would hit the unique
    # constraint on the second case and never reach the statement under test.
    with session_scope() as session:
        session.add(_exclusion(cik=cik))
    with pytest.raises(DatabaseError, match="append-only"), session_scope() as session:
        session.execute(text(statement))


def test_a_forecast_group_is_excluded_at_most_once_per_arm_and_version(
    migrated_database: str,
) -> None:
    with session_scope() as session:
        session.add(_exclusion(cik=303))
    with pytest.raises(IntegrityError), session_scope() as session:
        session.add(_exclusion(cik=303, reason="refused"))


def test_a_call_row_defaults_its_cache_writes_to_zero(migrated_database: str) -> None:
    with session_scope() as session:
        session.execute(
            text(
                "INSERT INTO llm_calls (run_id, session_id, model_id, input_tokens, "
                "output_tokens, cache_read_tokens, cost_usd) "
                "VALUES ('r-default', 's', 'm', 1, 1, 0, 0)"
            )
        )
    with session_scope() as session:
        row = session.execute(select(LlmCall).where(LlmCall.run_id == "r-default")).scalar_one()
    assert row.cache_creation_tokens == 0
