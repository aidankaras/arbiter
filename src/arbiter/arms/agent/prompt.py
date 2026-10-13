"""Turn one issuer-day's evidence packets into what the analyst is shown.

Everything the model sees comes from here, and everything here comes from the
packets: this module imports nothing that can read a label, an outcome, or a
later filing, and a test holds it to that. Packet text is filed by the issuer
and is therefore attacker-controllable, so it is fenced between delimiters the
packet itself cannot close.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, time
from decimal import Decimal
from functools import cached_property
from importlib import resources
from typing import Any
from zoneinfo import ZoneInfo

from arbiter.packets.schema import EvidencePacket

#: Pinned for the whole study: released 2026-06-30, so nothing it was trained on
#: postdates the scored window, and not retired before 2027-06-30.
MODEL_ID = "claude-sonnet-5"
#: Independent samples per issuer-day, averaged. Fixed by pre-registration.
SAMPLES = 3
EFFORT = "medium"
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "p_up": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": 600},
    },
    "required": ["p_up", "rationale"],
    "additionalProperties": False,
}
TEMPLATE_NAME = "insider_v1.md"

_OPEN = "<untrusted-data>"
_CLOSE = "</untrusted-data>"
_EASTERN = ZoneInfo("America/New_York")
_MARKET_OPEN = time(9, 30)


class UnsupportedPacketError(ValueError):
    """Raised for a packet holding content this prompt does not render.

    Rendering a subset silently would show the model less than the packet
    holds, while the prediction still cites the whole packet's hash.
    """


def system_prompt() -> str:
    """Return the analyst's instructions, read from the versioned template."""
    return resources.files("arbiter.arms.agent.prompts").joinpath(TEMPLATE_NAME).read_text()


def prompt_version() -> str:
    """Identify everything that decides what the model is asked.

    Any change to the template, the answer schema, the model, the effort or the
    number of samples yields a new version, so predictions made under different
    conditions can never be pooled by accident.
    """
    identity = json.dumps(
        {
            "template": system_prompt(),
            "schema": OUTPUT_SCHEMA,
            "model": MODEL_ID,
            "effort": EFFORT,
            "samples": SAMPLES,
        },
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class IssuerDay:
    """Every packet one issuer filed on one stored day: one forecast's evidence."""

    day: date
    cik: int
    packets: tuple[EvidencePacket, ...]

    @cached_property
    def ordered(self) -> tuple[EvidencePacket, ...]:
        """Packets in acceptance order, ties broken by content so order is stable."""
        return tuple(sorted(self.packets, key=lambda p: (p.as_of, p.event_id, p.content_hash)))

    @property
    def event_ids(self) -> tuple[str, ...]:
        """Distinct filings, in acceptance order. One filing can span several packets."""
        return tuple(dict.fromkeys(packet.event_id for packet in self.ordered))

    @property
    def issuer_day_hash(self) -> str:
        """A digest of every packet the forecast read, independent of their order."""
        hashes = sorted({packet.content_hash for packet in self.packets})
        return hashlib.sha256("\n".join(hashes).encode()).hexdigest()

    def packet_hash_for(self, event_id: str) -> str:
        """The packet hash a filing's prediction cites: its first packet's.

        A filing with several transactions has several packets; the prediction
        row cites the first, and `issuer_day_hash` covers all of them.
        """
        return next(p.content_hash for p in self.ordered if p.event_id == event_id)


def group_issuer_days(day: date, packets: Sequence[EvidencePacket]) -> list[IssuerDay]:
    """Group a stored day's packets into forecasts: by issuer, then by entry open.

    Filings by one issuer that straddle a market open have outcome windows
    starting at different opens. Showing every filing to the earlier window's
    forecast would show it a filing made after that window began, so such an
    issuer-day becomes one forecast per side of the open. Scoring still averages
    by issuer, as it does for the baseline. Groups are ordered by CIK, then by
    acceptance.

    Raises:
        UnsupportedPacketError: a packet is not an insider packet, or holds
            filing sections or comparables, which this prompt does not render.
    """
    groups: dict[tuple[int, date, bool], list[EvidencePacket]] = defaultdict(list)
    for packet in packets:
        _require_renderable(packet)
        local_date, before_open = _entry_key(packet)
        groups[(packet.issuer.cik, local_date, not before_open)].append(packet)
    return [IssuerDay(day, cik, tuple(group)) for (cik, _, _), group in sorted(groups.items())]


def _require_renderable(packet: EvidencePacket) -> None:
    if packet.domain != "insider":
        msg = f"packet {packet.event_id} is a {packet.domain} packet; only insider is supported"
        raise UnsupportedPacketError(msg)
    if packet.sections or packet.comparables:
        msg = (
            f"packet {packet.event_id} holds filing sections or comparables, which this "
            "prompt does not render; showing less than the hashed packet is not allowed"
        )
        raise UnsupportedPacketError(msg)


def _entry_key(packet: EvidencePacket) -> tuple[date, bool]:
    """Which open a filing's window starts at: Eastern date, and whether before 09:30.

    Stricter than the label's own entry session: a filing late one evening and
    one early the next morning share an open but get different keys. Splitting
    them costs an extra call and a little shared context, never a leak.
    """
    local = packet.as_of.astimezone(_EASTERN)
    return local.date(), local.time() < _MARKET_OPEN


def _text(value: object) -> str:
    """Render filed text so it cannot close or reopen the data fence."""
    rendered = "" if value is None else str(value)
    return rendered.replace(_OPEN, "[removed]").replace(_CLOSE, "[removed]")


def _number(value: Decimal | None) -> str:
    return "n/a" if value is None else format(value.normalize(), "f")


def render_user_message(group: IssuerDay) -> str:
    """Render an issuer-day's evidence as fenced, plain data.

    Bars and market statistics come from the earliest filing's packet: they are
    the ones knowable before every filing in the group, so no filing's packet
    contributes a bar another filing could not have seen.
    """
    first = group.ordered[0]
    lines = [
        _OPEN,
        f"company: {_text(first.issuer.company)}",
        f"ticker: {_text(first.issuer.ticker)}",
        f"sector_etf: {_text(first.issuer.sector_etf)}",
        "",
        "filings:",
    ]
    for packet in group.ordered:
        extract = packet.extracted
        lines.append(
            f"- accepted {packet.as_of.isoformat()}"
            f" | insider {_text(extract.insider_name)}"
            f" | position {_text(extract.position)}"
            f" | code {_text(extract.transaction_code)}"
            f" | shares {_number(extract.shares)}"
            f" | price {_number(extract.price)}"
            f" | value_usd {_number(extract.value_usd)}"
            f" | remaining_shares {_number(extract.remaining_shares)}"
            f" | 10b5-1 {extract.is_10b5_1}"
        )
    market = first.market
    lines += [
        "",
        "market (as of the earliest filing):",
        f"trailing_return_21d: {_number(market.trailing_return_21d)}",
        f"realised_volatility_21d: {_number(market.realised_volatility_21d)}",
        f"volume_percentile_63d: {_number(market.volume_percentile_63d)}",
        "",
        "bars (session,open,high,low,close,volume):",
    ]
    lines += [
        f"{bar.session.isoformat()},{_number(bar.open)},{_number(bar.high)},"
        f"{_number(bar.low)},{_number(bar.close)},{_number(bar.volume)}"
        for bar in first.bars
    ]
    lines.append(_CLOSE)
    return "\n".join(lines)
