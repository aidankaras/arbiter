"""A packet's hash is the evidence two arms read the same thing.

Everything the four-arm comparison claims rests on that: a prediction stores the
hash of the packet it came from, and two arms reporting one hash is the proof
they saw one input. A hash that drifts between runs would break the claim while
leaving every number in the report looking exactly as it should.

So the properties worth testing hardest are negative. The same content must hash
alike however it was assembled, spelled, or timezoned; different content must
not; and a value with no stable textual form must be refused rather than
rendered by whatever `repr` it happens to have.
"""

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from arbiter.packets.hashing import UnhashableValueError, canonical_bytes, content_hash

ACCEPTED = datetime(2026, 8, 3, 20, 47, tzinfo=UTC)


def _packet(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "event_id": "0000764180-26-000102",
        "domain": "insider",
        "as_of": ACCEPTED,
        "cik": 764180,
        "ticker": "MO",
        "sector_etf": "XLP",
        "extracted": {"value_usd": Decimal("50000.00"), "is_purchase": True},
    }
    fields.update(overrides)
    return fields


def test_the_same_packet_hashes_the_same_twice():
    assert content_hash(_packet()) == content_hash(_packet())


def test_key_order_is_not_content():
    """A packet assembled in a different order is the same packet."""
    forward = {"a": 1, "b": 2, "c": 3}
    backward = {"c": 3, "b": 2, "a": 1}

    assert content_hash(forward) == content_hash(backward)


def test_the_same_instant_in_two_zones_hashes_alike():
    """One moment is one moment; the zone it was written in is not content."""
    eastern = datetime(2026, 8, 3, 16, 47, tzinfo=timezone(timedelta(hours=-4)))

    assert content_hash(_packet(as_of=eastern)) == content_hash(_packet())


def test_a_decimal_hashes_by_value_not_by_spelling():
    """1.50, 1.5 and 1.500 are one number and must not be three packets."""
    hashes = {content_hash({"x": Decimal(spelling)}) for spelling in ("1.5", "1.50", "1.500")}

    assert len(hashes) == 1


def test_a_decimal_in_exponent_form_hashes_with_its_plain_form():
    assert content_hash({"x": Decimal("1E+2")}) == content_hash({"x": Decimal("100")})


def test_changing_any_field_changes_the_hash():
    """The property that makes a matching hash mean anything at all."""
    base = content_hash(_packet())

    assert content_hash(_packet(ticker="AAPL")) != base
    assert content_hash(_packet(cik=320193)) != base
    assert content_hash(_packet(as_of=ACCEPTED + timedelta(seconds=1))) != base


def test_changing_a_nested_field_changes_the_hash():
    """A packet's evidence lives in nested structure; it must not be skipped."""
    changed = _packet(extracted={"value_usd": Decimal("50000.01"), "is_purchase": True})

    assert content_hash(changed) != content_hash(_packet())


def test_reordering_a_sequence_changes_the_hash():
    """Order is content for a list: two orderings are two different packets."""
    assert content_hash({"items": ["a", "b"]}) != content_hash({"items": ["b", "a"]})


def test_a_float_is_refused_rather_than_hashed():
    """Its shortest round-tripping text is a property of the representation."""
    with pytest.raises(UnhashableValueError, match="float"):
        content_hash({"price": 84.86})


def test_a_naive_datetime_is_refused():
    """It names no instant, so two zones would silently produce one hash."""
    with pytest.raises(UnhashableValueError, match="naive"):
        content_hash({"as_of": datetime(2026, 8, 3, 20, 47)})


def test_a_set_is_refused_because_its_order_is_not_guaranteed():
    with pytest.raises(UnhashableValueError, match="stable textual form"):
        content_hash({"codes": {"4.01", "4.02"}})


def test_an_arbitrary_object_is_refused():
    class Opaque:
        pass

    with pytest.raises(UnhashableValueError, match="Opaque"):
        content_hash({"thing": Opaque()})


def test_the_bytes_the_digest_is_taken_over_are_inspectable():
    """A hash that cannot be explained is not evidence; a mismatch is read here."""
    rendered = canonical_bytes({"b": Decimal("1.50"), "a": ACCEPTED}).decode()

    assert rendered == '{"a":"2026-08-03T20:47:00+00:00","b":"1.5"}'


def test_no_whitespace_reaches_the_digest():
    """Formatting must never alter an identity."""
    assert b" " not in canonical_bytes({"a": 1, "b": 2})


def test_a_true_boolean_does_not_collide_with_one():
    """`True == 1` in Python, and a packet must not confuse them."""
    assert content_hash({"x": True}) != content_hash({"x": 1})


def test_an_empty_packet_still_has_an_identity():
    assert len(content_hash({})) == 64
