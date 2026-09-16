"""Append-only ledger tables.

Rows here are written exactly once. Updates and deletes are refused by a
database trigger rather than by application convention, because a forecast that
can be edited after its outcome is known is not evidence of anything.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from arbiter.db.base import Base


class Prediction(Base):
    """One arm's forecast for one event, written before any outcome exists.

    `packet_hash` ties the forecast to the exact evidence it was made from, so a
    published result can be reproduced and any change to packet construction is
    detectable rather than silent. `brief_hash` is null for arms that consume no
    research brief, such as the feature baseline.
    """

    __tablename__ = "predictions"
    __table_args__ = (CheckConstraint("horizon_days > 0", name="predictions_horizon_positive"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    arm: Mapped[str] = mapped_column(String(32), index=True)
    model_id: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(32))
    packet_hash: Mapped[str] = mapped_column(String(64), index=True)
    brief_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    horizon_days: Mapped[int]
    class_probs: Mapped[dict[str, Any]] = mapped_column(JSONB)
    point_estimate: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class LlmCall(Base):
    """Token and cost accounting for one model invocation.

    Recorded for every call so that cost per decision is a measured quantity and
    the spend ceiling has real data to enforce against.
    """

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(64), index=True)
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    cache_read_tokens: Mapped[int]
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    agent_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
