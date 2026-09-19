"""A packet must refuse to hold anything its event could not have known.

The tests that matter here are the ones asserting a packet cannot be built. A
packet carrying a future bar is not a crash and not a wrong number: it is a
forecast that looks exactly like every other forecast and is worth nothing, and
nothing downstream can detect it. So the boundary is enforced at construction,
and these tests pin the boundary rather than the happy path either side of it.
"""

from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from arbiter.packets.schema import (
    Comparable,
    EvidencePacket,
    FilingSection,
    IssuerIdentity,
    LookaheadError,
    MarketSummary,
    SessionBar,
)

# 16:47 Eastern on a Monday in July: after the 16:00 close, so that session's
# bar is knowable and the next session's is not.
ACCEPTED = datetime(2026, 7, 13, 20, 47, tzinfo=UTC)

ISSUER = IssuerIdentity(cik=764180, ticker="MO", company="Altria Group", sector_etf="XLP")


def _bar(day: int) -> SessionBar:
    return SessionBar(
        session=date(2026, 7, day),
        open=Decimal("50.00"),
        high=Decimal("51.00"),
        low=Decimal("49.50"),
        close=Decimal("50.75"),
        volume=Decimal("1200000"),
    )


def _packet(**overrides: object) -> EvidencePacket:
    fields: dict[str, object] = {
        "event_id": "0000764180-26-000102",
        "domain": "insider",
        "as_of": ACCEPTED,
        "issuer": ISSUER,
        "bars": (_bar(10), _bar(13)),
    }
    fields.update(overrides)
    return EvidencePacket(**fields)  # pyright: ignore[reportArgumentType]


def test_a_packet_holding_only_closed_bars_is_accepted():
    """The session of the event itself is knowable once it has closed."""
    assert _packet().bars[-1].session == date(2026, 7, 13)


def test_a_bar_that_had_not_closed_is_refused():
    """The defect this class exists to make impossible."""
    with pytest.raises(LookaheadError, match="2026-07-14"):
        _packet(bars=(_bar(10), _bar(14)))


def test_the_event_session_is_refused_when_the_filing_arrived_before_the_close():
    """10:00 Eastern: the session is open, so its close does not exist yet.

    The same bar and the same session as the accepted case above; only the
    instant differs. Nothing about the bar itself is wrong, which is why this
    cannot be checked by inspecting the bar.
    """
    with pytest.raises(LookaheadError, match="2026-07-13"):
        _packet(as_of=datetime(2026, 7, 13, 14, 0, tzinfo=UTC), bars=(_bar(13),))


def test_a_naive_instant_is_refused():
    with pytest.raises(LookaheadError, match="naive"):
        _packet(as_of=datetime(2026, 7, 13, 16, 47))


def test_a_comparable_resolving_after_the_event_is_refused():
    """A case is useful because its outcome is known — it must have been known."""
    unresolved = Comparable(
        event_id="0000320193-26-000044",
        as_of=datetime(2026, 7, 1, 20, 0, tzinfo=UTC),
        resolved_at=datetime(2026, 7, 20, 20, 0, tzinfo=UTC),
        abnormal_return=Decimal("0.031"),
        summary="Officer purchase, 5-session abnormal return.",
    )

    with pytest.raises(LookaheadError, match="0000320193-26-000044"):
        _packet(comparables=(unresolved,))


def test_a_comparable_resolving_before_the_event_is_accepted():
    """The negative control: the same field one day the other side of the line."""
    resolved = Comparable(
        event_id="0000320193-26-000044",
        as_of=datetime(2026, 6, 1, 20, 0, tzinfo=UTC),
        resolved_at=datetime(2026, 6, 8, 20, 0, tzinfo=UTC),
        abnormal_return=Decimal("0.031"),
        summary="Officer purchase, 5-session abnormal return.",
    )

    assert _packet(comparables=(resolved,)).comparables[0].event_id == "0000320193-26-000044"


def test_a_packet_is_frozen_after_construction():
    packet = _packet()

    with pytest.raises(ValidationError, match="frozen"):
        packet.event_id = "something else"  # type: ignore[misc]


def test_an_unknown_field_is_refused_rather_than_ignored():
    """A misspelled field that is silently dropped is evidence quietly missing."""
    with pytest.raises(ValidationError, match=r"extra_forbidden|Extra inputs"):
        _packet(secotr_etf="XLP")


def test_the_same_contents_hash_the_same():
    assert _packet().content_hash == _packet().content_hash


def test_changing_any_evidence_changes_the_hash():
    """What makes a stored hash proof that two arms read one input."""
    base = _packet().content_hash

    assert _packet(bars=(_bar(10),)).content_hash != base
    assert _packet(domain="redflag").content_hash != base
    assert _packet(extracted={"value_usd": Decimal("50000")}).content_hash != base


def test_a_float_is_normalised_to_a_decimal_at_the_boundary():
    """A float must never reach the digest, and here it cannot.

    The schema converts through the number's decimal spelling rather than its
    binary value, so 0.1 is stored as exactly 0.1 and not as the expansion
    0.1000000000000000055511151231257827 that `Decimal(0.1)` would give. The
    hash is then taken over a value with one stable textual form, which is what
    `hashing` refuses a raw float for.
    """
    packet = _packet(extracted={"value_usd": 0.1})

    assert packet.extracted["value_usd"] == Decimal("0.1")
    assert packet.content_hash == _packet(extracted={"value_usd": Decimal("0.1")}).content_hash


def test_a_lookahead_failure_is_not_reported_as_a_validation_error():
    """The correctness failure must be distinguishable from a typo.

    Pydantic re-raises a `ValueError` from a validator as its own
    `ValidationError`. Were `LookaheadError` a `ValueError`, a caller could not
    tell a packet carrying tomorrow's price from a misspelled field name.
    """
    with pytest.raises(LookaheadError):
        _packet(bars=(_bar(14),))

    with pytest.raises(ValidationError):
        _packet(secotr_etf="XLP")


def test_a_comparable_resolving_exactly_at_the_event_is_accepted():
    """The boundary. A case whose window closed at this instant was knowable.

    Written because a mutation loosening the guard from `>` to `>=` passed the
    whole suite: the cases either side of the line were covered and the line
    itself was not.
    """
    at_the_instant = Comparable(
        event_id="0000320193-26-000044",
        as_of=datetime(2026, 6, 1, 20, 0, tzinfo=UTC),
        resolved_at=ACCEPTED,
        abnormal_return=Decimal("0.031"),
        summary="Resolved at the moment this filing was accepted.",
    )

    assert _packet(comparables=(at_the_instant,)).comparables[0].resolved_at == ACCEPTED


def test_a_decimal_hashes_by_value_and_not_by_how_it_was_written():
    """The packet must not undo the normalisation the digest relies on.

    Written because dumping the packet in Pydantic's JSON mode passed the whole
    suite. That mode loses no precision — it renders a `Decimal` as a string —
    but it renders it before the canonical form can normalise it, so `1.50` and
    `1.5` would become two packets with two identities and two hashes, and every
    test over `hashing` would still pass.
    """
    fifty = _packet(extracted={"value_usd": Decimal("50000.00")})
    fifty_again = _packet(extracted={"value_usd": Decimal("50000")})

    assert fifty.content_hash == fifty_again.content_hash


def test_one_instant_written_in_two_zones_is_one_packet():
    """The same guarantee for time, defeated by the same mutation."""
    eastern = datetime(2026, 7, 13, 16, 47, tzinfo=timezone(timedelta(hours=-4)))

    assert _packet(as_of=eastern).content_hash == _packet().content_hash


def test_summary_statistics_are_absent_rather_than_zero_when_unknown():
    """A zero return claims the stock did not move; None claims we cannot say."""
    assert _packet().market.trailing_return_21d is None


def test_summary_statistics_are_carried_when_present():
    summary = MarketSummary(
        trailing_return_21d=Decimal("-0.043"),
        realised_volatility_21d=Decimal("0.312"),
        volume_percentile_63d=Decimal("0.88"),
    )

    assert _packet(market=summary).content_hash != _packet().content_hash


def test_sections_are_kept_apart_rather_than_concatenated():
    """Section boundaries are where untrusted filing text can be delimited."""
    packet = _packet(
        sections=(
            FilingSection(heading="Item 4.01", text="Dismissal of accountant."),
            FilingSection(heading="Item 9.01", text="Exhibit 16.1 letter."),
        )
    )

    assert [section.heading for section in packet.sections] == ["Item 4.01", "Item 9.01"]


def test_two_sections_joined_differently_are_different_packets():
    """Order is content: the same text in two arrangements is two inputs."""
    forward = _packet(
        sections=(FilingSection(heading="a", text="x"), FilingSection(heading="b", text="y"))
    )
    backward = _packet(
        sections=(FilingSection(heading="b", text="y"), FilingSection(heading="a", text="x"))
    )

    assert forward.content_hash != backward.content_hash
