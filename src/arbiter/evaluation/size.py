"""The information coefficient by issuer size.

The insider-trading literature locates the signal in small companies, so a
headline averaged over all issuers can hide a result that exists in one tercile
and is absent or reversed in another.

Size is the median daily dollar volume (close times volume) over the issuer's
last `SIZE_SESSIONS` sessions in its evidence packet. Packets are point-in-time:
`build_packet` keeps only bars whose session had closed by the filing's
acceptance, so the proxy uses nothing the forecast could not have seen. Dollar
volume rather than market capitalisation because the packet carries bars and no
share count, and traded value is what a position would have to be sized against.

Terciles are drawn within each scored day, not over the pooled sample. A
market-wide change in volume then moves every issuer together and leaves each
one's bucket unchanged; pooled terciles would let a volume boom relabel the
whole market "large" and make the buckets a calendar split in disguise.

Tie rule: issuers are ranked by (size, CIK) ascending and the rank `r` of `n`
sized issuers goes to bucket `3r // n`. Equal sizes may therefore fall on
opposite sides of a boundary, broken by CIK, which is arbitrary but deterministic.
A value-based rule (percentile thresholds) would instead put all tied issuers in
one bucket and make bucket populations uneven exactly where volume is coarse.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from arbiter.arms.evaluate import IssuerScore
from arbiter.evaluation.groups import ic_within_groups
from arbiter.evaluation.metrics import ICSummary
from arbiter.packets.schema import EvidencePacket, SessionBar
from arbiter.packets.store import partition_exists, read_packets

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

#: Sessions of bars the size proxy is taken over. Fewer available means no
#: size, never a size from a shorter window.
SIZE_SESSIONS = 20

BUCKETS = ("Small", "Middle", "Large")

#: Size by (day, CIK), in dollars of daily volume.
SizeLookup = Mapping[tuple[date, int], float]


def median_dollar_volume(bars: Sequence[SessionBar]) -> float | None:
    """Return the median of close times volume over the last `SIZE_SESSIONS` sessions.

    Returns `None` when fewer than `SIZE_SESSIONS` bars exist.
    """
    ordered = sorted(bars, key=lambda bar: bar.session)
    if len(ordered) < SIZE_SESSIONS:
        return None
    return float(statistics.median(bar.close * bar.volume for bar in ordered[-SIZE_SESSIONS:]))


def packet_sizes(packets: Sequence[EvidencePacket]) -> dict[int, float]:
    """Return each issuer's size from one day's packets; unsized issuers are absent.

    An issuer-day can have several packets whose bars differ: a filing accepted
    after the close sees one session more than one accepted before it, and an
    issuer with more than one listed security (a common and a preferred share)
    has a price series per security. The largest size decides. Every packet is
    from the one day, so the choice uses nothing later than the filing, and the
    primary listing is the one that carries the volume; taking the earliest
    packet sized a company on its preferred share at a fiftieth of its common's
    traded value.
    """
    sizes: dict[int, float] = {}
    for packet in packets:
        size = median_dollar_volume(packet.bars)
        if size is not None:
            sizes[packet.issuer.cik] = max(size, sizes.get(packet.issuer.cik, size))
    return sizes


def load_sizes(
    root: Path, domain: str, days: Sequence[date]
) -> tuple[dict[tuple[date, int], float], int]:
    """Build the size lookup for `days` from the packet store.

    Returns the lookup and the number of the days that have a packet partition.
    Days without a partition contribute no sizes. Reads go through the store's
    own reader, which verifies each packet against its stored bytes.
    """
    lookup: dict[tuple[date, int], float] = {}
    partitions = 0
    for day in days:
        if not partition_exists(root, domain, day):
            continue
        partitions += 1
        for cik, size in packet_sizes(read_packets(root, domain, day)).items():
            lookup[(day, cik)] = size
    return lookup, partitions


def assign_buckets[Key: SupportsRichComparison](sizes: Mapping[Key, float]) -> dict[Key, int]:
    """Assign each key a tercile index, 0 smallest, by the rule in the module docstring."""
    ranked = sorted(sizes, key=lambda key: (sizes[key], key))
    count = len(ranked)
    return {cik: 3 * rank // count for rank, cik in enumerate(ranked)}


@dataclass(frozen=True)
class BucketIC:
    """One size bucket's information coefficient across days."""

    label: str
    #: `None` when fewer than two bucket-days reached the threshold.
    ic: ICSummary | None
    #: Bucket-days that contributed a coefficient.
    days: int
    #: Issuer-days on those bucket-days only.
    issuer_days: int
    #: Median size of those issuer-days, dollars of daily volume; `None` when no
    #: bucket-day contributed.
    median_size: float | None


@dataclass(frozen=True)
class SizeBreakdown:
    """The coefficient by size bucket and the sample it leaves out.

    The buckets' issuer-days, `dropped` and `unbucketed` add up to
    `scored_issuer_days`.
    """

    buckets: tuple[BucketIC, ...]
    #: Scored issuer-days with no size: no packet, or fewer than `SIZE_SESSIONS` bars.
    unbucketed: int
    #: Sized issuer-days whose bucket-day held fewer than `MIN_EVENTS_PER_DAY`.
    dropped: int
    scored_issuer_days: int


def ic_by_size(
    scores: Mapping[date, Sequence[IssuerScore]],
    sizes: SizeLookup,
    overlap_lags: int = 0,
) -> SizeBreakdown:
    """Compute the information coefficient within size terciles of each day.

    Terciles are formed among the issuers that have a size. A bucket-day
    contributes only with at least `MIN_EVENTS_PER_DAY` issuers in that bucket:
    the headline's minimum applied to each third, so a day needs roughly three
    times as many sized issuers as the headline asks. The per-day coefficients
    are summarised exactly as the headline's are.
    """

    def assign(day: date, issuers: Sequence[IssuerScore]) -> dict[int, int]:
        return assign_buckets(
            {s.cik: sizes[(day, s.cik)] for s in issuers if (day, s.cik) in sizes}
        )

    grouped = ic_within_groups(scores, assign, len(BUCKETS), overlap_lags=overlap_lags)
    buckets = tuple(
        BucketIC(
            label=label,
            ic=group.ic,
            days=group.days,
            issuer_days=group.issuer_days,
            median_size=(
                statistics.median(sizes[(day, m.cik)] for day, m in group.members)
                if group.members
                else None
            ),
        )
        for label, group in zip(BUCKETS, grouped.groups, strict=True)
    )
    return SizeBreakdown(
        buckets,
        unbucketed=grouped.ungrouped,
        dropped=grouped.dropped,
        scored_issuer_days=grouped.scored_issuer_days,
    )
