"""Create the append-only ledger tables and their mutation guard.

Revision ID: 0001
Revises:
Create Date: 2026-09-15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_GUARD_FUNCTION = """
CREATE OR REPLACE FUNCTION arbiter_forbid_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME;
END;
$$;
"""

_GUARDED_TABLES = ("predictions", "llm_calls")


def upgrade() -> None:
    """Create the tables, then attach the append-only trigger to each."""
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.String(64), nullable=False),
        sa.Column("arm", sa.String(32), nullable=False),
        sa.Column("model_id", sa.String(64), nullable=False),
        sa.Column("prompt_version", sa.String(32), nullable=False),
        sa.Column("packet_hash", sa.String(64), nullable=False),
        sa.Column("brief_hash", sa.String(64), nullable=True),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("class_probs", postgresql.JSONB(), nullable=False),
        sa.Column("point_estimate", sa.Numeric(12, 6), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint("horizon_days > 0", name="predictions_horizon_positive"),
    )
    op.create_index("ix_predictions_event_id", "predictions", ["event_id"])
    op.create_index("ix_predictions_arm", "predictions", ["arm"])
    op.create_index("ix_predictions_packet_hash", "predictions", ["packet_hash"])

    op.create_table(
        "llm_calls",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("model_id", sa.String(64), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(12, 6), nullable=False),
        sa.Column("agent_name", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_llm_calls_run_id", "llm_calls", ["run_id"])
    op.create_index("ix_llm_calls_model_id", "llm_calls", ["model_id"])

    op.execute(_GUARD_FUNCTION)
    for table in _GUARDED_TABLES:
        # A statement-level trigger is deliberate: a row-level trigger would let
        # a statement matching zero rows succeed silently, which reads to the
        # caller as a successful edit.
        op.execute(
            f"CREATE TRIGGER {table}_append_only "
            f"BEFORE UPDATE OR DELETE ON {table} "
            "FOR EACH STATEMENT EXECUTE FUNCTION arbiter_forbid_mutation();"
        )


def downgrade() -> None:
    """Drop the tables and the shared guard function."""
    for table in _GUARDED_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS {table}_append_only ON {table};")
    op.drop_table("llm_calls")
    op.drop_table("predictions")
    op.execute("DROP FUNCTION IF EXISTS arbiter_forbid_mutation();")
