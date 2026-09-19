"""A stored packet must come back as the same evidence it went in as.

The property worth testing hardest is the round trip, because its failure is
silent: a packet that reloads with a different identity breaks the link between
a prediction and the input that produced it, and every number in the report
still renders. So the store writes the exact bytes the hash was taken over, and
these tests assert that reading them back reproduces both the packet and the
hash. They also pin what the store does not do: a coherent edit to a stored
packet is not detected here, it changes the packet's identity, and the tamper
surfaces at the prediction that recorded the old hash.
"""

import gzip
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.packets.schema import (
    Comparable,
    EvidencePacket,
    FilingSection,
    InsiderExtract,
    IssuerIdentity,
    MarketSummary,
    SessionBar,
)
from arbiter.packets.store import (
    PacketIntegrityError,
    partition_exists,
    read_packets,
    write_packets,
)

DAY = date(2026, 7, 13)
ACCEPTED = datetime(2026, 7, 13, 20, 47, tzinfo=UTC)


def _packet(event_id: str = "0000764180-26-000102") -> EvidencePacket:
    return EvidencePacket(
        event_id=event_id,
        domain="insider",
        as_of=ACCEPTED,
        issuer=IssuerIdentity(
            cik=764180, ticker="MO", company="Altria Group", sector_etf="XLP"
        ),
        sections=(FilingSection(heading="Item 4.01", text="Dismissal of accountant."),),
        extracted=InsiderExtract(
            insider_name="Jane Roe",
            position="Officer",
            transaction_code="P",
            shares=Decimal("1000"),
            price=Decimal("50.00"),
            value_usd=Decimal("50000.00"),
            remaining_shares=None,
        ),
        bars=(
            SessionBar(
                session=date(2026, 7, 10),
                open=Decimal("50.10"),
                high=Decimal("51.00"),
                low=Decimal("49.50"),
                close=Decimal("50.75"),
                volume=Decimal("1200000"),
            ),
        ),
        market=MarketSummary(
            trailing_return_21d=Decimal("-0.043"),
            realised_volatility_21d=Decimal("0.312"),
            volume_percentile_63d=Decimal("0.88"),
        ),
        comparables=(
            Comparable(
                event_id="0000320193-26-000044",
                as_of=datetime(2026, 6, 1, 20, 0, tzinfo=UTC),
                resolved_at=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
                abnormal_return=Decimal("0.031"),
                summary="Officer purchase.",
            ),
        ),
    )


def test_a_packet_survives_the_round_trip_with_its_identity_intact(tmp_path: Path):
    """The property every stored result depends on."""
    original = _packet()
    write_packets([original], tmp_path, "insider", DAY)

    (reloaded,) = read_packets(tmp_path, "insider", DAY)

    assert reloaded.content_hash == original.content_hash
    assert reloaded == original


def test_every_field_survives_the_round_trip(tmp_path: Path):
    """A hash matching is necessary but not sufficient; check the evidence too."""
    write_packets([_packet()], tmp_path, "insider", DAY)

    (reloaded,) = read_packets(tmp_path, "insider", DAY)

    assert reloaded.as_of == ACCEPTED
    assert reloaded.issuer.sector_etf == "XLP"
    assert reloaded.sections[0].heading == "Item 4.01"
    assert reloaded.bars[0].session == date(2026, 7, 10)
    assert reloaded.bars[0].close == Decimal("50.75")
    assert reloaded.market.trailing_return_21d == Decimal("-0.043")
    assert reloaded.comparables[0].abnormal_return == Decimal("0.031")
    assert reloaded.extracted.remaining_shares is None
    assert reloaded.extracted.is_10b5_1 is False


def test_the_stored_line_is_the_bytes_the_hash_was_taken_over(tmp_path: Path):
    """What makes the file itself the evidence rather than a copy of it."""
    packet = _packet()
    path = write_packets([packet], tmp_path, "insider", DAY)

    line = gzip.decompress(path.read_bytes()).splitlines()[0]

    from arbiter.packets.hashing import canonical_bytes, content_hash

    assert line == canonical_bytes(packet.hashable_fields())
    assert content_hash(packet.hashable_fields()) == packet.content_hash


def test_a_day_that_was_built_empty_is_not_a_day_that_was_never_built(tmp_path: Path):
    """The distinction between a complete dataset and a truncated one."""
    assert not partition_exists(tmp_path, "insider", DAY)

    write_packets([], tmp_path, "insider", DAY)

    assert partition_exists(tmp_path, "insider", DAY)
    assert read_packets(tmp_path, "insider", DAY) == []


def test_reading_a_day_that_was_never_built_raises(tmp_path: Path):
    """An unbuilt day must not read as an empty one."""
    with pytest.raises(FileNotFoundError, match="has not been built"):
        read_packets(tmp_path, "insider", DAY)


def test_rewriting_a_day_replaces_it_rather_than_appending(tmp_path: Path):
    """A re-run must be idempotent, not cumulative."""
    write_packets([_packet("a"), _packet("b")], tmp_path, "insider", DAY)
    write_packets([_packet("c")], tmp_path, "insider", DAY)

    assert [packet.event_id for packet in read_packets(tmp_path, "insider", DAY)] == ["c"]


def test_a_day_written_for_one_domain_does_not_disturb_another(tmp_path: Path):
    write_packets([_packet("a")], tmp_path, "insider", DAY)
    write_packets([], tmp_path, "redflag", DAY)

    assert len(read_packets(tmp_path, "insider", DAY)) == 1
    assert read_packets(tmp_path, "redflag", DAY) == []


def test_a_coherent_edit_changes_the_identity_rather_than_being_detected_here(tmp_path: Path):
    """States what this store does *not* do, so nobody relies on it doing it.

    A round-trip check cannot detect tampering: an edit that leaves valid JSON
    re-serialises to itself, so the stored bytes and the re-derived bytes agree
    and the read succeeds. What changes is the packet's identity, and that is
    where the tamper surfaces — a prediction recorded the hash of what it read,
    and the altered packet no longer hashes to it.

    Written as a test because the alternative is a comment nobody checks, and
    because an earlier version of this file asserted the opposite and was wrong.
    """
    original = _packet()
    path = write_packets([original], tmp_path, "insider", DAY)
    tampered = gzip.decompress(path.read_bytes()).replace(
        b'"close":"50.75"', b'"close":"99.99"'
    )
    path.write_bytes(gzip.compress(tampered))

    (reloaded,) = read_packets(tmp_path, "insider", DAY)

    assert reloaded.bars[0].close == Decimal("99.99")
    assert reloaded.content_hash != original.content_hash


def test_a_line_that_does_not_round_trip_is_refused(tmp_path: Path):
    """What the round-trip check does catch: a stored form that parsing changes.

    Spelling a decimal non-canonically is the shape a canonical-form change
    between versions would take, and it must not load silently — every hash in
    the partition would shift under it.
    """
    path = write_packets([_packet()], tmp_path, "insider", DAY)
    drifted = gzip.decompress(path.read_bytes()).replace(
        b'"close":"50.75"', b'"close":"50.750"'
    )
    path.write_bytes(gzip.compress(drifted))

    with pytest.raises(PacketIntegrityError, match="does not round-trip"):
        read_packets(tmp_path, "insider", DAY)


def test_many_packets_keep_their_order_and_their_identities(tmp_path: Path):
    packets = [_packet(f"event-{index}") for index in range(25)]
    write_packets(packets, tmp_path, "insider", DAY)

    reloaded = read_packets(tmp_path, "insider", DAY)

    assert [packet.event_id for packet in reloaded] == [packet.event_id for packet in packets]
    assert [packet.content_hash for packet in reloaded] == [
        packet.content_hash for packet in packets
    ]
