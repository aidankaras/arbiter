"""A duplicate prediction is permanent, so the write must fail instead.

Predictions cannot be updated or deleted, which is what makes them evidence. The
same property means a duplicate written by a retry can never be cleaned up, and
every aggregate computed afterwards would count it twice.
"""

from decimal import Decimal

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from arbiter.db.models import Prediction
from arbiter.db.session import session_scope

pytestmark = pytest.mark.integration


def _prediction(packet_hash: str, arm: str = "baseline") -> Prediction:
    return Prediction(
        event_id="0001234567-26-000099",
        arm=arm,
        model_id="logistic-v1",
        prompt_version="v1",
        packet_hash=packet_hash,
        brief_hash=None,
        horizon_days=5,
        class_probs={"down": 0.25, "flat": 0.35, "up": 0.40},
        point_estimate=Decimal("0.0123"),
    )


def test_a_retried_write_is_refused(migrated_database: str) -> None:
    with session_scope() as session:
        session.add(_prediction("d" * 64))

    with pytest.raises(IntegrityError), session_scope() as session:
        session.add(_prediction("e" * 64))


def test_the_refusal_leaves_exactly_one_row(migrated_database: str) -> None:
    """The point is the count, not the error: aggregates must not double-count."""
    with session_scope() as session:
        session.add(_prediction("f" * 64, arm="dl"))

    with pytest.raises(IntegrityError), session_scope() as session:
        session.add(_prediction("g" * 64, arm="dl"))

    with session_scope() as session:
        count = session.execute(
            select(func.count())
            .select_from(Prediction)
            .where(Prediction.event_id == "0001234567-26-000099", Prediction.arm == "dl")
        ).scalar_one()
    assert count == 1


def test_different_arms_may_forecast_the_same_event(migrated_database: str) -> None:
    """The constraint must not stop the comparison the project exists to make."""
    with session_scope() as session:
        session.add(_prediction("h" * 64, arm="agent"))
        session.add(_prediction("i" * 64, arm="arbiter"))

    with session_scope() as session:
        arms = session.execute(
            select(Prediction.arm).where(Prediction.event_id == "0001234567-26-000099")
        ).scalars()
    assert {"agent", "arbiter"} <= set(arms)
