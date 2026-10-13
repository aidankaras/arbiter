"""Packets, recorded samples and a scripted runtime for agent-arm tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from arbiter.arms.agent.runtime import Sample, Status
from arbiter.packets.schema import (
    EvidencePacket,
    InsiderExtract,
    IssuerIdentity,
    SessionBar,
)

_STREAM = Path(__file__).parent / "fixtures" / "agent" / "stream_ok.jsonl"
#: The `result` event of a real recorded call: real usage for metering.
RECORDED_RESULT: dict[str, Any] = next(
    event
    for event in (json.loads(line) for line in _STREAM.read_text().splitlines())
    if event.get("type") == "result"
)
# 15:00 Eastern on Tuesday 2026-07-14.
AFTERNOON = datetime(2026, 7, 14, 19, 0, tzinfo=UTC)


def make_packet(
    event_id: str, cik: int, *, as_of: datetime = AFTERNOON, shares: str = "1000"
) -> EvidencePacket:
    return EvidencePacket(
        event_id=event_id,
        domain="insider",
        as_of=as_of,
        issuer=IssuerIdentity(cik=cik, ticker=f"T{cik}", company=f"Co {cik}", sector_etf="XLK"),
        extracted=InsiderExtract(
            insider_name="Jane Roe",
            position="Officer",
            transaction_code="P",
            shares=Decimal(shares),
            price=Decimal("50.00"),
            value_usd=Decimal("50000.00"),
        ),
        bars=(
            SessionBar(
                session=(as_of - timedelta(days=1)).date(),
                open=Decimal("50"),
                high=Decimal("51"),
                low=Decimal("49"),
                close=Decimal("50.5"),
                volume=Decimal("1000000"),
            ),
        ),
    )


def ok(p_up: float, weekly: float = 0.30) -> Sample:
    return Sample(
        Status.OK,
        p_up=p_up,
        rationale=f"reason {p_up}",
        payload=RECORDED_RESULT,
        utilization={"five_hour": 0.1, "seven_day": weekly},
    )


def failed(status: Status, *, metered: bool = True) -> Sample:
    # A completed call reports its usage windows whatever its answer was.
    return Sample(
        status,
        payload=RECORDED_RESULT if metered else None,
        utilization={"five_hour": 0.1, "seven_day": 0.30},
        detail=status.value,
    )


class ScriptedRuntime:
    """Returns prepared samples in order, and records every prompt it was sent."""

    def __init__(self, samples: list[Sample]) -> None:
        self.samples = list(samples)
        self.prompts: list[str] = []

    def sample(self, system: str, user: str) -> Sample:
        self.prompts.append(user)
        if not self.samples:
            raise AssertionError("the runner drew more samples than the test prepared")
        return self.samples.pop(0)
