"""Building one day's packets from its stored events.

A packet is built for every stored event, not only for events whose outcome can
be measured yet. The distinction matters more than it looks: the packet records
what was knowable at the filing, and whether the label has since resolved is a
fact about the calendar rather than about the evidence. Building only resolvable
events would make the set of packets depend on when the pass was run, and would
repeat the defect that once fitted a day's features over only its labelled rows.

That is also why this pass plans its own price window rather than reusing the
one labelling plans. Labelling's window is planned over the events resolvable
today and runs forwards from each event; this one runs backwards and stops at
the event. Neither is a subset of the other, and sharing them would couple a
packet's existence to a label's availability.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from decimal import InvalidOperation
from pathlib import Path
from typing import Any

from arbiter.ingestion.market import Bar, closed_bars, tradeable_symbol
from arbiter.packets.build import build_packet
from arbiter.packets.schema import EvidencePacket
from arbiter.packets.store import write_packets

#: Calendar days of history requested before the earliest event in a day.
#:
#: Sized by the deepest backward statistic a packet reports, a volume percentile
#: ranked over 63 sessions. Counted in calendar days rather than against the
#: market calendar because counting sessions backwards from early January would
#: reach into a year the holiday table does not cover and raise — and because it
#: is unnecessary: the bars returned are themselves the trading calendar, and a
#: packet reports an absent statistic rather than a wrong one when it receives
#: fewer sessions than a window needs. 63 sessions span 88 calendar days of
#: weekends alone, so this leaves room for holidays on top.
HISTORY_DAYS = 120


class UnsupportedPacketDomainError(ValueError):
    """Raised for a domain that has no extract model yet.

    Red-flag packets need their own typed extract before they can be built, and
    an 8-K names its filer by CIK alone. Raising is the honest answer: building
    them with insider fields would produce packets whose evidence is empty and
    whose hashes look ordinary.
    """


def build_day(
    day: date,
    root: Path,
    packet_root: Path,
    *,
    fetch_bars: Callable[[Sequence[str], date, date], dict[str, list[Bar]]],
    benchmark_for: Callable[[int], str],
    domain: str = "insider",
) -> tuple[int, list[dict[str, str]]]:
    """Build and store every packet for one stored day.

    Args:
        day: the stored day to build.
        root: the event store's root.
        packet_root: the packet store's root.
        fetch_bars: returns each symbol's daily bars across an inclusive window.
        benchmark_for: returns the sector benchmark for an issuer's CIK.
        domain: the event domain.

    Returns:
        The number of packets written, and one record per event excluded, each
        naming the event and the reason. A partition is written even when the
        count is zero, so a day built and found empty stays distinguishable from
        a day never built.

    Raises:
        UnsupportedPacketDomainError: the domain has no extract model.
    """
    from arbiter.ingestion.store import read_events, write_unpriceable

    if domain != "insider":
        msg = (
            f"packets are not built for the {domain!r} domain: it has no typed "
            "extract model, and building it with another domain's fields would "
            "produce packets with empty evidence and ordinary-looking hashes"
        )
        raise UnsupportedPacketDomainError(msg)

    rows: Sequence[Mapping[str, Any]] = read_events(root, domain, day)
    excluded: list[dict[str, str]] = []

    usable: list[tuple[Mapping[str, Any], str]] = []
    for row in rows:
        symbol = tradeable_symbol(row.get("ticker"))
        if symbol is None:
            # Recorded rather than dropped: an event with no priceable symbol is
            # missing from the packet set for a stated reason, which is what
            # keeps a thin day distinguishable from a broken one.
            excluded.append(
                {
                    "accession_no": str(row.get("accession_no", "")),
                    "reason": f"no tradeable symbol in {row.get('ticker')!r}",
                }
            )
            continue
        usable.append((row, symbol))

    packets: list[EvidencePacket] = []
    if usable:
        # One request across every symbol, over one window spanning the day's
        # events. A request per issuer would cost hundreds against a rate limit
        # measured per minute, and the window being wider than any single event
        # needs is far cheaper than the requests it saves.
        earliest = min(row["as_of"].date() for row, _ in usable)
        latest = max(row["as_of"].date() for row, _ in usable)
        symbols = sorted({symbol for _, symbol in usable})
        series = fetch_bars(symbols, earliest - timedelta(days=HISTORY_DAYS), latest)

        for row, symbol in usable:
            # Truncated here rather than left to `build_packet`, because the
            # emptiness test below has to run on what the packet will actually
            # hold. A series that is non-empty but lies entirely after the event
            # passes a test on the vendor's rows and truncates to nothing, which
            # writes exactly the packet this guard exists to prevent: hashed,
            # citable, and carrying no market evidence at all.
            bars = closed_bars(series.get(symbol, []), row["as_of"])
            if not bars:
                excluded.append(
                    {
                        "accession_no": str(row["accession_no"]),
                        "reason": f"no price history for {symbol} before the event",
                    }
                )
                continue
            try:
                packets.append(
                    build_packet(
                        row,
                        bars,
                        domain=domain,
                        sector_etf=benchmark_for(int(row["cik"])),
                    )
                )
            except (InvalidOperation, TypeError, ValueError) as error:
                # One malformed row must cost one event, not the day. Reaching
                # here without a handler aborted `build_day` before any packet
                # was written, so a single unparseable price discarded every
                # other packet on the day along with it.
                excluded.append(
                    {
                        "accession_no": str(row["accession_no"]),
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )

    write_packets(packets, packet_root, domain, day)
    # Written for every day, including days that excluded nothing, for the same
    # reason an empty partition is still written: a day thinned by unpriceable
    # issuers must not look identical to a quiet one. Echoing the list to stdout
    # left the distinction in a terminal scrollback.
    write_unpriceable(excluded, packet_root, domain, day, stage="packets")
    return len(packets), excluded
