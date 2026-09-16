from decimal import Decimal

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import DatabaseError

from arbiter.db.models import Prediction
from arbiter.db.session import session_scope

pytestmark = pytest.mark.integration


def _prediction(packet_hash: str) -> Prediction:
    return Prediction(
        event_id="0001234567-25-000001",
        arm="baseline",
        model_id="logistic-v1",
        prompt_version="n/a",
        packet_hash=packet_hash,
        brief_hash=None,
        horizon_days=5,
        class_probs={"down": 0.25, "flat": 0.35, "up": 0.40},
        point_estimate=Decimal("0.0123"),
    )


def test_a_prediction_can_be_written_once(migrated_database: str) -> None:
    packet_hash = "a" * 64
    with session_scope() as session:
        session.add(_prediction(packet_hash))

    with session_scope() as session:
        stored = session.execute(
            select(Prediction).where(Prediction.packet_hash == packet_hash)
        ).scalar_one()
        assert stored.class_probs["up"] == 0.40
        assert stored.point_estimate == Decimal("0.0123")
        assert stored.created_at.tzinfo is not None


def test_updating_a_prediction_is_rejected_by_the_database(migrated_database: str) -> None:
    """A forecast must not be editable once its outcome is known."""
    with session_scope() as session:
        session.add(_prediction("b" * 64))

    with pytest.raises(DatabaseError, match="append-only"), session_scope() as session:
        session.execute(text("UPDATE predictions SET arm = 'tampered'"))


def test_deleting_a_prediction_is_rejected_by_the_database(migrated_database: str) -> None:
    with session_scope() as session:
        session.add(_prediction("c" * 64))

    with pytest.raises(DatabaseError, match="append-only"), session_scope() as session:
        session.execute(text("DELETE FROM predictions"))


def test_the_guard_also_covers_the_usage_ledger(migrated_database: str) -> None:
    """Cost records are evidence too, so they carry the same protection."""
    with pytest.raises(DatabaseError, match="append-only"), session_scope() as session:
        session.execute(text("UPDATE llm_calls SET cost_usd = 0"))
