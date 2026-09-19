"""Packet storage as date-partitioned, gzipped JSON lines.

One file per domain per day, matching the event store's partitioning so that a
re-run replaces its own partition and touches no other. An empty day still
writes a file, because "no packets were built" and "this day was never built"
are different facts and only the second is a defect.

Each line is the exact byte sequence the packet's hash was taken over. That is
the reason this store is JSON rather than Parquet, and the reason it does not
use a second serialiser of its own: a stored packet that round-trips to a
different hash would break the claim that a prediction can be traced to its
input, and the surest way to avoid it is for the stored bytes and the hashed
bytes to be the same bytes.

There is deliberately no checksum file. Integrity here is established by the
join that matters — a prediction records the hash of the packet it read, and a
packet that no longer hashes to that value is detected at the point where the
discrepancy would change a result. A checksum over the compressed payload would
catch bit-rot and nothing else, and a check that can only fail in a way nothing
has ever observed teaches a reader to trust it beyond what it verifies.

Packets are stored whole, including their bars, rather than referencing a shared
price cache. That duplicates prices across events on the same issuer and is
worth it: a packet is the evidence for a published result, and evidence that
depends on another file still being correct is weaker than evidence that does
not.
"""

from __future__ import annotations

import gzip
import json
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from arbiter.packets.hashing import canonical_bytes
from arbiter.packets.schema import EvidencePacket


class PacketIntegrityError(RuntimeError):
    """Raised when a stored packet does not hash to the value it was stored as.

    Either the file was altered after it was written or the packet's canonical
    form has changed between versions. Both invalidate every prediction that
    cites the affected hash, so neither may be repaired silently.
    """


def _partition(root: Path, domain: str, day: date) -> Path:
    return root / domain / f"{day.isoformat()}.jsonl.gz"


def write_packets(
    packets: Sequence[EvidencePacket], root: Path, domain: str, day: date
) -> Path:
    """Write one day's packets, replacing any previous file for that day.

    Args:
        packets: the day's packets, in any order.
        root: the packet store's root directory.
        domain: the event domain, which selects the subdirectory.
        day: the partition date.

    Returns:
        The path written.
    """
    path = _partition(root, domain, day)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = b"".join(canonical_bytes(packet.hashable_fields()) + b"\n" for packet in packets)

    # Written to a temporary file in the same directory and moved into place, so
    # a run interrupted mid-write leaves the previous partition intact rather
    # than a truncated file that reads as a short day.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(gzip.compress(payload))
    temporary.replace(path)
    return path


def partition_exists(root: Path, domain: str, day: date) -> bool:
    """Report whether a day has been built, empty or not."""
    return _partition(root, domain, day).exists()


def read_packets(root: Path, domain: str, day: date) -> list[EvidencePacket]:
    """Read one day's packets, checking each against the bytes it was stored as.

    Raises:
        FileNotFoundError: the day was never built, which is distinct from a day
            that was built and held no packets.
        PacketIntegrityError: a stored packet does not reproduce the identity its
            stored bytes imply.
    """
    path = _partition(root, domain, day)
    if not path.exists():
        msg = (
            f"no packet partition for {domain} on {day.isoformat()}; the day has "
            "not been built, which is not the same as a day with no packets"
        )
        raise FileNotFoundError(msg)

    packets: list[EvidencePacket] = []
    for number, line in enumerate(gzip.decompress(path.read_bytes()).splitlines(), start=1):
        if not line.strip():
            continue
        stored: dict[str, Any] = json.loads(line)
        packet = EvidencePacket.model_validate(stored)

        # The line is what the digest was taken over, so re-deriving it from the
        # reconstructed packet and comparing is a real round-trip check: it fails
        # if parsing lost anything the identity depends on.
        if canonical_bytes(packet.hashable_fields()) != line:
            msg = (
                f"packet on line {number} of {path} does not round-trip: the "
                "stored bytes and the bytes its reconstruction hashes to differ, "
                "so any prediction citing it can no longer be traced to its input"
            )
            raise PacketIntegrityError(msg)

        packets.append(packet)

    return packets
