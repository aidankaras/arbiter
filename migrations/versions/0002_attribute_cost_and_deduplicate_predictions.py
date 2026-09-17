"""Attribute model cost to its decision, and refuse duplicate predictions.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-17
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add cost attribution columns and a uniqueness constraint on predictions."""
    # Cost per decision is a headline metric, and it cannot be computed from a
    # run identifier alone: one run scores many events. These columns tie a
    # model call to the evidence it read and the forecast it produced. They are
    # nullable because calls that belong to no single decision — a backfill, an
    # exploratory run — legitimately have neither.
    op.add_column("llm_calls", sa.Column("packet_hash", sa.String(64), nullable=True))
    op.add_column("llm_calls", sa.Column("prediction_id", sa.Integer(), nullable=True))
    op.create_index("ix_llm_calls_packet_hash", "llm_calls", ["packet_hash"])
    op.create_foreign_key(
        "fk_llm_calls_prediction",
        "llm_calls",
        "predictions",
        ["prediction_id"],
        ["id"],
    )

    # Predictions are append-only, so a duplicate written by a retry can never be
    # deleted, and every aggregate computed afterwards would double-count it.
    # The constraint makes the retry fail instead.
    op.create_unique_constraint(
        "uq_predictions_event_arm_version",
        "predictions",
        ["event_id", "arm", "prompt_version"],
    )


def downgrade() -> None:
    """Remove the constraint and the attribution columns."""
    op.drop_constraint("uq_predictions_event_arm_version", "predictions", type_="unique")
    op.drop_constraint("fk_llm_calls_prediction", "llm_calls", type_="foreignkey")
    op.drop_index("ix_llm_calls_packet_hash", table_name="llm_calls")
    op.drop_column("llm_calls", "prediction_id")
    op.drop_column("llm_calls", "packet_hash")
