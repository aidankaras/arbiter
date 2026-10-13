"""What the agent arm reads from and writes to the append-only ledger.

A protocol rather than direct database calls, so the runner's decisions — skip,
raise, exclude, halt — are tested in the default suite against an in-memory
ledger, and the Postgres implementation is tested once, against a real server,
for the properties only a database has.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy import func, select

from arbiter.db.models import AgentExclusion, LlmCall, Prediction
from arbiter.db.session import session_scope
from arbiter.llm.usage import record_run

#: The arm name every agent prediction and exclusion is stored under.
ARM = "agent"
#: The agent forecasts the insider domain's five-session abnormal return.
HORIZON_DAYS = 5


@dataclass(frozen=True)
class ForecastRow:
    """One event's share of an issuer-day forecast, ready to be written."""

    event_id: str
    packet_hash: str
    #: The digest of every packet the issuer-day's calls read; the same value
    #: is stored on those calls' usage rows, which is how cost joins to forecast.
    issuer_day_hash: str
    p_up: float
    samples: tuple[float, ...]
    rationales: tuple[str, ...]


@dataclass(frozen=True)
class Exclusion:
    """An issuer-day the arm attempted and could not score."""

    event_day: date
    cik: int
    issuer_day_hash: str
    prompt_version: str
    reason: str
    samples_valid: int
    event_ids: tuple[str, ...]
    detail: str = ""


class Ledger(Protocol):
    """The ledger operations the agent runner depends on."""

    def predicted(self, prompt_version: str, event_ids: Sequence[str]) -> set[str]:
        """Return which of `event_ids` already hold an agent prediction."""
        ...

    def is_excluded(self, prompt_version: str, issuer_day_hash: str) -> bool:
        """Whether this forecast group was already recorded as excluded."""
        ...

    def excluded_events(self, prompt_version: str, event_ids: Sequence[str]) -> set[str]:
        """Return which of `event_ids` belong to any recorded exclusion."""
        ...

    def write_forecast(
        self, model_id: str, prompt_version: str, rows: Sequence[ForecastRow]
    ) -> None:
        """Write every row of one issuer-day in a single transaction."""
        ...

    def write_exclusion(self, exclusion: Exclusion) -> None:
        """Record an issuer-day the arm could not score."""
        ...

    def record_call(self, payload: Mapping[str, Any], run_id: str, packet_hash: str) -> None:
        """Meter one model call, whatever its outcome."""
        ...

    def spent_since(self, instant: datetime) -> Decimal:
        """Total recorded cost of calls made at or after `instant`."""
        ...

    def forecasts(self, prompt_version: str) -> dict[str, float]:
        """Every agent forecast under `prompt_version`, by filing."""
        ...

    def exclusion_counts(self, prompt_version: str) -> dict[str, int]:
        """Excluded forecast groups under `prompt_version`, by reason."""
        ...

    def tokens_for_run(self, run_id: str) -> int:
        """Every token recorded against `run_id`."""
        ...


class PgLedger:
    """The ledger held in Postgres, behind the append-only triggers."""

    def predicted(self, prompt_version: str, event_ids: Sequence[str]) -> set[str]:
        """Return which of `event_ids` already hold an agent prediction."""
        with session_scope() as session:
            found = session.execute(
                select(Prediction.event_id).where(
                    Prediction.arm == ARM,
                    Prediction.prompt_version == prompt_version,
                    Prediction.event_id.in_(list(event_ids)),
                )
            ).scalars()
            return set(found)

    def is_excluded(self, prompt_version: str, issuer_day_hash: str) -> bool:
        """Whether this forecast group was already recorded as excluded."""
        with session_scope() as session:
            count = session.execute(
                select(func.count())
                .select_from(AgentExclusion)
                .where(
                    AgentExclusion.arm == ARM,
                    AgentExclusion.prompt_version == prompt_version,
                    AgentExclusion.issuer_day_hash == issuer_day_hash,
                )
            ).scalar_one()
            return count > 0

    def excluded_events(self, prompt_version: str, event_ids: Sequence[str]) -> set[str]:
        """Return which of `event_ids` belong to any recorded exclusion."""
        # ponytail: reads every exclusion of the version; a few hundred at most.
        with session_scope() as session:
            recorded = session.execute(
                select(AgentExclusion.event_ids).where(
                    AgentExclusion.arm == ARM, AgentExclusion.prompt_version == prompt_version
                )
            ).scalars()
            excluded = {event_id for ids in recorded for event_id in ids}
        return excluded & set(event_ids)

    def write_forecast(
        self, model_id: str, prompt_version: str, rows: Sequence[ForecastRow]
    ) -> None:
        """Write every row of one issuer-day in a single transaction."""
        with session_scope() as session:
            for row in rows:
                session.add(
                    Prediction(
                        event_id=row.event_id,
                        arm=ARM,
                        model_id=model_id,
                        prompt_version=prompt_version,
                        packet_hash=row.packet_hash,
                        brief_hash=None,
                        horizon_days=HORIZON_DAYS,
                        class_probs={
                            "up": row.p_up,
                            "samples": list(row.samples),
                            "rationales": list(row.rationales),
                            "issuer_day_hash": row.issuer_day_hash,
                        },
                        point_estimate=Decimal(str(row.p_up)),
                    )
                )

    def write_exclusion(self, exclusion: Exclusion) -> None:
        """Record an issuer-day the arm could not score."""
        with session_scope() as session:
            session.add(
                AgentExclusion(
                    event_day=exclusion.event_day,
                    cik=exclusion.cik,
                    issuer_day_hash=exclusion.issuer_day_hash,
                    arm=ARM,
                    prompt_version=exclusion.prompt_version,
                    reason=exclusion.reason,
                    samples_valid=exclusion.samples_valid,
                    event_ids=list(exclusion.event_ids),
                    detail=exclusion.detail[:2000],
                )
            )

    def record_call(self, payload: Mapping[str, Any], run_id: str, packet_hash: str) -> None:
        """Meter one model call, whatever its outcome."""
        record_run(payload, run_id=run_id, agent_name=ARM, packet_hash=packet_hash)

    def spent_since(self, instant: datetime) -> Decimal:
        """Total recorded cost of calls made at or after `instant`."""
        with session_scope() as session:
            total = session.execute(
                select(func.coalesce(func.sum(LlmCall.cost_usd), 0)).where(
                    LlmCall.created_at >= instant
                )
            ).scalar_one()
            return Decimal(total)

    def forecasts(self, prompt_version: str) -> dict[str, float]:
        """Every agent forecast under `prompt_version`, by filing."""
        with session_scope() as session:
            rows = session.execute(
                select(Prediction.event_id, Prediction.point_estimate).where(
                    Prediction.arm == ARM, Prediction.prompt_version == prompt_version
                )
            ).all()
            return {event_id: float(estimate) for event_id, estimate in rows}

    def exclusion_counts(self, prompt_version: str) -> dict[str, int]:
        """Excluded forecast groups under `prompt_version`, by reason."""
        with session_scope() as session:
            rows = session.execute(
                select(AgentExclusion.reason, func.count())
                .where(
                    AgentExclusion.arm == ARM, AgentExclusion.prompt_version == prompt_version
                )
                .group_by(AgentExclusion.reason)
            ).all()
            return {reason: int(count) for reason, count in rows}

    def tokens_for_run(self, run_id: str) -> int:
        """Every token recorded against `run_id`, cache writes included."""
        with session_scope() as session:
            total = session.execute(
                select(
                    func.coalesce(
                        func.sum(
                            LlmCall.input_tokens
                            + LlmCall.output_tokens
                            + LlmCall.cache_read_tokens
                            + LlmCall.cache_creation_tokens
                        ),
                        0,
                    )
                ).where(LlmCall.run_id == run_id)
            ).scalar_one()
            return int(total)
