"""Meter cache-write tokens, and record issuer-days an agent arm could not score.

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add the cache-write column and the append-only exclusions table."""
    # A headless Claude Code call reports almost all of its input as tokens
    # written to the prompt cache: a measured call carried 2 uncached input
    # tokens and 14,154 cache-write tokens. Without this column the ledger
    # would record a fraction of a percent of what each call read, and a token
    # ceiling summed from it could never trip.
    op.add_column(
        "llm_calls",
        sa.Column("cache_creation_tokens", sa.Integer(), nullable=False, server_default="0"),
    )

    # An issuer-day an arm could not score is evidence about the arm, not noise
    # to drop: the comparison removes it from every arm, and the report counts
    # it by reason. Append-only for the same reason predictions are.
    op.create_table(
        "agent_exclusions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_day", sa.Date(), nullable=False),
        sa.Column("cik", sa.BigInteger(), nullable=False),
        # The forecast group's identity. Usually one per issuer-day, but filings
        # either side of a market open form two groups for one issuer and day.
        sa.Column("issuer_day_hash", sa.String(64), nullable=False),
        sa.Column("arm", sa.String(32), nullable=False),
        sa.Column("prompt_version", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(32), nullable=False),
        sa.Column("samples_valid", sa.Integer(), nullable=False),
        sa.Column("event_ids", JSONB(), nullable=False),
        sa.Column("detail", sa.String(2000), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "issuer_day_hash", "arm", "prompt_version", name="uq_agent_exclusions_group"
        ),
    )
    op.execute(
        "CREATE TRIGGER agent_exclusions_append_only "
        "BEFORE UPDATE OR DELETE ON agent_exclusions "
        "FOR EACH STATEMENT EXECUTE FUNCTION arbiter_forbid_mutation();"
    )


def downgrade() -> None:
    """Drop the exclusions table and the cache-write column."""
    op.execute("DROP TRIGGER IF EXISTS agent_exclusions_append_only ON agent_exclusions;")
    op.drop_table("agent_exclusions")
    op.drop_column("llm_calls", "cache_creation_tokens")
