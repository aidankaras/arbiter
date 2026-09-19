"""The evidence packet: one event, as it appeared at one instant.

A packet is the only thing a forecasting arm ever sees. That single fact decides
what belongs in it, and the decision is a research one rather than a technical
one, so it is stated here rather than left to be inferred from the fields.

The packet carries the filing's text, the identity of the issuer, the fields a
domain extractor pulled from the document, the raw daily bars that had closed by
the event instant, and a short list of descriptive statistics over those bars.
It does not carry the features the conventional arm is fitted on. Handing every
arm that feature vector would silently change the question from "given identical
evidence, who forecasts better?" to "given identical features, who maps them to a
forecast better?" — and it would foreclose the only interesting way a language
model could win, which is by noticing something the feature set leaves out. The
summary statistics are included because the alternative is worse: a model asked
to compute a trailing return from sixty rows of prices is being measured on
arithmetic rather than on judgement. They are deliberately descriptive and
deliberately few, so that neither arm inherits the other's hypothesis.

Two properties are enforced here rather than trusted to the code that builds
packets, because both fail silently and neither is visible in a result:

**Nothing in a packet postdates its instant.** Every bar must have closed by
`as_of`, and every comparable must have resolved by then. A packet holding one
future bar produces a forecast that looks ordinary and is worthless.

**A packet has one identity, derived from its contents.** `content_hash` is
computed, never stored and never passed in, so it cannot disagree with the
packet it names.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from arbiter.ingestion.market import session_close_instant
from arbiter.packets.hashing import content_hash


class LookaheadError(Exception):
    """Raised when a packet would contain information postdating its instant.

    Deliberately not a `ValueError`. Pydantic catches `ValueError` and
    `AssertionError` raised inside a validator and re-raises them as its own
    `ValidationError`, which would bury this behind the same exception type as a
    misspelled field — leaving a caller unable to distinguish the project's
    central correctness failure from a typo. Inheriting from `Exception` lets it
    propagate under its own name, so `except LookaheadError` catches exactly one
    thing and catches all of it.
    """


class _Frozen(BaseModel):
    """Base for every packet component.

    `frozen` prevents a field being reassigned after construction. It does not
    deep-freeze a nested container, so it is not the whole guarantee: the hash
    is. A packet whose contents were altered after the fact no longer hashes to
    the value stored with the prediction that cited it, which is what makes the
    alteration detectable rather than invisible.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class IssuerIdentity(_Frozen):
    """Who filed, and what it is compared against."""

    cik: int
    ticker: str
    company: str
    #: The sector ETF the issuer's abnormal return is measured against. Stored
    #: in the packet because a label computed against a different benchmark is
    #: not comparable, and the benchmark must be recoverable from the evidence.
    sector_etf: str


class SessionBar(_Frozen):
    """One daily bar, addressed by session rather than by instant.

    Distinct from `ingestion.market.Bar`, which is shaped by the price vendor
    and timestamped at the session's start. This type is part of a published,
    hashed artefact: if the vendor changes its field names or its timestamp
    convention, packet hashes must not move with it.
    """

    session: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class MarketSummary(_Frozen):
    """Descriptive statistics over the bars, in units a reader can reason about.

    Each is `None` when the window holds too little history to compute it. An
    explicit absence rather than a neutral fill: a zero trailing return states
    that the stock did not move, which is a different claim from stating that
    the history is too short to say.
    """

    #: Cumulative return over the 21 sessions ending at the last closed bar.
    trailing_return_21d: Decimal | None = None
    #: Standard deviation of daily returns over the same window, annualised.
    realised_volatility_21d: Decimal | None = None
    #: Where the most recent session's volume falls in the trailing 63, as a
    #: fraction in [0, 1]. Unusual volume before a filing is the kind of context
    #: a reader would want and a fitted feature set does not currently hold.
    volume_percentile_63d: Decimal | None = None


class InsiderExtract(_Frozen):
    """The fields a Form 4 yields, typed.

    Typed rather than carried in a `dict[str, str | int | Decimal | ...]`,
    because a loose bag does not survive storage. The canonical form renders a
    `Decimal` to a string so the digest is stable, and on reload that string is
    indistinguishable from a value that was always a string — so `value_usd`
    returned as `"50000"` rather than `Decimal("50000")`, and the packet hash
    matched anyway, because both render identically. An arm doing arithmetic on
    it would have concatenated. Declaring the field's type is what lets the
    value be parsed back into the type it was written from.

    The red-flag domain will need its own extract model, and `extracted` becomes
    a union discriminated on `domain` when that domain's packets are built. It is
    typed to this one until then rather than left loose, because a field that
    accepts anything accepts the drift above.
    """

    insider_name: str
    position: str
    transaction_code: str
    shares: Decimal
    price: Decimal
    value_usd: Decimal
    #: Absent when the filing reports no post-transaction holding. That is a
    #: statement that the filing does not say, which differs from a holding of
    #: zero — an insider who has sold out entirely.
    remaining_shares: Decimal | None = None
    is_10b5_1: bool = False


class FilingSection(_Frozen):
    """One titled section of the filing's text.

    Kept sectioned rather than flattened so that a prompt can delimit each part
    explicitly. Filing text is attacker-controllable: an issuer chooses what its
    own document says, and a section boundary is where an instruction-shaped
    sentence stops being able to pass itself off as surrounding context.
    """

    heading: str
    text: str


class Comparable(_Frozen):
    """A historical case, and how it turned out.

    The sharpest lookahead risk in the packet. A comparable is useful precisely
    because its outcome is known, so the constraint is that the outcome must
    have been known *then*: `resolved_at` is when the case's own measurement
    window closed, and a case that resolved after this event is a case whose
    answer had not been published yet.
    """

    event_id: str
    as_of: datetime
    resolved_at: datetime
    abnormal_return: Decimal
    summary: str


class EvidencePacket(_Frozen):
    """Everything one arm is given about one event, and nothing else."""

    event_id: str
    domain: str
    #: The instant the filing became public. Timezone-aware without exception:
    #: a naive value would be compared against aware ones below and raise, and
    #: would hash as no particular moment.
    as_of: datetime
    issuer: IssuerIdentity
    sections: tuple[FilingSection, ...] = ()
    #: The fields a domain extractor pulled from the document, typed so that a
    #: stored packet reloads with the types it was written from.
    extracted: InsiderExtract
    bars: tuple[SessionBar, ...] = ()
    market: MarketSummary = MarketSummary()
    comparables: tuple[Comparable, ...] = ()

    @model_validator(mode="after")
    def _reject_information_from_the_future(self) -> Self:
        """Refuse a packet holding anything that postdated its own instant."""
        if self.as_of.tzinfo is None:
            msg = f"packet {self.event_id} has a naive as_of, which names no instant"
            raise LookaheadError(msg)

        for bar in self.bars:
            closed_at = session_close_instant(bar.session)
            if closed_at > self.as_of:
                msg = (
                    f"packet {self.event_id} holds the {bar.session.isoformat()} bar, "
                    f"which closed at {closed_at.isoformat()}, after the event at "
                    f"{self.as_of.isoformat()}. Truncate bars with "
                    "`ingestion.market.closed_bars` before building a packet."
                )
                raise LookaheadError(msg)

        for case in self.comparables:
            if case.resolved_at > self.as_of:
                msg = (
                    f"packet {self.event_id} cites comparable {case.event_id}, which "
                    f"resolved at {case.resolved_at.isoformat()}, after the event at "
                    f"{self.as_of.isoformat()}. Its outcome was not knowable here."
                )
                raise LookaheadError(msg)

        return self

    @property
    def content_hash(self) -> str:
        """The packet's identity, derived from its contents on every read.

        Computed rather than stored so that it cannot be set to a value the
        packet does not have, and so that a packet mutated through a nested
        container reports a hash that no longer matches the one recorded with
        the prediction that cited it.
        """
        return content_hash(self.hashable_fields())

    def hashable_fields(self) -> dict[str, Any]:
        """Render the packet in the plain form the digest is taken over.

        Public because the packet store writes exactly these bytes: what is on
        disk is then the same thing the identity was computed from, rather than
        a second serialisation that could drift from it.

        `mode="python"` deliberately: it leaves `Decimal` and `datetime` as
        themselves, so the canonical form still sees the types it normalises.

        `mode="json"` would not lose precision — it renders a `Decimal` as a
        string, not a float — but it renders it *before* the canonical form can
        normalise it, handing over `"1.50"` where the digest needs `1.50` and
        `1.5` to be one value. The same applies to instants, which arrive
        pre-formatted in one zone rather than as a moment to be expressed in
        UTC. Both guarantees are pinned by tests over `hashing`, and dumping to
        JSON here would defeat them without failing any of those tests.
        """
        return self.model_dump(mode="python")
