"""Assembling an evidence packet from a stored event and its price history.

The schema refuses a packet that contains information postdating its event. This
module's job is to never hand it one, which means the truncation happens here and
the validator is the second line rather than the first. A builder that relies on
the validator to catch its mistakes produces a stream of exceptions in a backfill
and no packets; a builder that truncates correctly produces packets the validator
then confirms.

Three things the packet does not yet carry, stated because an empty field and an
unbuilt feature look identical from the outside:

`sections` is empty for the insider domain. A Form 4 is a structured filing whose
content is its table of transactions, not prose, and that table arrives in
`extracted`. The field exists for the red-flag domain, whose 8-K items are
narrative and whose text must reach an arm delimited rather than concatenated.

`comparables` is always empty, because comparable retrieval is not written. When
it exists it will populate this field; until then no caller can ask for them, so
an empty tuple here means "not built" and never "searched and found none". Those
must not become indistinguishable once retrieval lands.

The market summary is computed from the closed bars alone. It is deliberately
descriptive rather than predictive: it reports what the price series did, in
units a reader can check, and it does not encode the conventional arm's view of
which of those facts matters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from typing import Any

from arbiter.ingestion.market import Bar, closed_bars
from arbiter.ingestion.timestamps import SEC_TIMEZONE
from arbiter.packets.schema import (
    EvidencePacket,
    InsiderExtract,
    IssuerIdentity,
    MarketSummary,
    SessionBar,
)

#: Sessions of return the trailing statistics describe. Twenty-one sessions is
#: about one calendar month, long enough to carry a drift and short enough that
#: it has not been overtaken by a different regime.
_TRAILING_SESSIONS = 21

#: Sessions the volume percentile ranks against — about one quarter, which is
#: long enough to contain an earnings cycle's worth of normal volume.
_VOLUME_SESSIONS = 63

#: Sessions in a trading year, for annualising a daily standard deviation.
_SESSIONS_PER_YEAR = 252


def to_session_bar(bar: Bar) -> SessionBar:
    """Convert a vendor bar into the packet's own bar type.

    The session is the bar's date in the market's timezone, not in UTC.

    The vendor stamps a daily bar at its session's start, which is 05:00 UTC —
    00:00 EST or 01:00 EDT, the same Eastern date all year, so that particular
    stamp is not itself ambiguous. The conversion is here for the convention
    that would be: a bar stamped 00:00 UTC falls on the previous Eastern day
    every day of the year, and dating such a series by its UTC date would move
    every bar one session earlier. Nothing in the vendor's contract pins which
    convention it uses, so the packet does not depend on it.
    """
    return SessionBar(
        session=bar.timestamp.astimezone(SEC_TIMEZONE).date(),
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
    )


def _trailing_return(bars: Sequence[SessionBar]) -> Decimal | None:
    """Cumulative return over the trailing window, or `None` if too short."""
    if len(bars) < _TRAILING_SESSIONS + 1:
        return None
    start = bars[-(_TRAILING_SESSIONS + 1)].close
    if start <= 0:
        # A non-positive close is not a price. Returning None reports that the
        # statistic could not be computed, rather than dividing and reporting a
        # number whose sign is meaningless.
        return None
    return (bars[-1].close - start) / start


def _realised_volatility(bars: Sequence[SessionBar]) -> Decimal | None:
    """Annualised standard deviation of daily log returns, or `None`.

    Log returns rather than simple ones: they add across sessions, so scaling by
    the square root of the session count is the right arithmetic rather than an
    approximation that drifts with the size of the moves.
    """
    if len(bars) < _TRAILING_SESSIONS + 1:
        return None

    window = bars[-(_TRAILING_SESSIONS + 1) :]
    returns: list[Decimal] = []
    for earlier, later in pairwise(window):
        if earlier.close <= 0 or later.close <= 0:
            return None
        try:
            returns.append((later.close / earlier.close).ln())
        except InvalidOperation:
            return None

    mean = sum(returns) / Decimal(len(returns))
    # Sample variance: the window is a sample of the return process, not the
    # whole of it, so the denominator is n-1.
    variance = sum((value - mean) ** 2 for value in returns) / Decimal(len(returns) - 1)
    return (variance * Decimal(_SESSIONS_PER_YEAR)).sqrt()


def _volume_percentile(bars: Sequence[SessionBar]) -> Decimal | None:
    """Where the latest session's volume ranks in the trailing window.

    Reported as the fraction of the window at or below the latest volume, so 1
    means the heaviest session in the window and 0.5 the median. `None` when the
    window is too short to rank against.
    """
    if len(bars) < _VOLUME_SESSIONS:
        return None
    window = [bar.volume for bar in bars[-_VOLUME_SESSIONS:]]
    latest = window[-1]
    at_or_below = sum(1 for volume in window if volume <= latest)
    return Decimal(at_or_below) / Decimal(len(window))


def market_summary(bars: Sequence[SessionBar]) -> MarketSummary:
    """Describe a price series in a few figures a reader can verify.

    Every statistic is `None` when its window is too short. That is the whole
    reason they are optional: a zero trailing return asserts the stock did not
    move, and filling a missing one with zero would put that assertion into the
    evidence of every event near the start of a price history.
    """
    return MarketSummary(
        trailing_return_21d=_trailing_return(bars),
        realised_volatility_21d=_realised_volatility(bars),
        volume_percentile_63d=_volume_percentile(bars),
    )


def _insider_extract(row: Mapping[str, Any]) -> InsiderExtract:
    """Read the Form 4 fields out of a stored row.

    Each field is named rather than the row being passed through whole. A row
    carries storage bookkeeping an arm has no business seeing, and a column
    added to the event store later would otherwise change every packet hash
    without anyone deciding that it should.
    """
    return InsiderExtract(
        insider_name=str(row["insider_name"]),
        position=str(row["position"]),
        transaction_code=str(row["transaction_code"]),
        shares=Decimal(str(row["shares"])),
        price=Decimal(str(row["price"])),
        value_usd=Decimal(str(row["value_usd"])),
        remaining_shares=(
            None
            if row.get("remaining_shares") is None
            else Decimal(str(row["remaining_shares"]))
        ),
        is_10b5_1=bool(row["is_10b5_1"]),
    )


def build_packet(
    row: Mapping[str, Any],
    bars: Sequence[Bar],
    *,
    domain: str,
    sector_etf: str,
) -> EvidencePacket:
    """Assemble one packet from a stored event row and a price series.

    Args:
        row: one stored event, as the event store holds it.
        bars: daily bars for the issuer, which may extend past the event. They
            are truncated here; passing a longer series than needed is safe and
            passing a shorter one produces absent statistics rather than wrong
            ones.
        domain: the event domain, which decides how the row is read.
        sector_etf: the benchmark this issuer's abnormal return is measured
            against, carried in the packet so a label stays interpretable.

    Returns:
        A packet containing only what was knowable at the event instant.

    Raises:
        LookaheadError: the truncation here disagreed with the schema's check,
            which means one of them is wrong and no packet should be produced.
    """
    as_of = row["as_of"]
    truncated = [to_session_bar(bar) for bar in closed_bars(bars, as_of)]

    return EvidencePacket(
        event_id=str(row["accession_no"]),
        domain=domain,
        as_of=as_of,
        issuer=IssuerIdentity(
            cik=int(row["cik"]),
            ticker=str(row["ticker"]),
            company=str(row["issuer"]),
            sector_etf=sector_etf,
        ),
        sections=(),
        extracted=_insider_extract(row),
        bars=tuple(truncated),
        market=market_summary(truncated),
        comparables=(),
    )
