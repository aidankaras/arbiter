"""Turning a packet's contents into a stable identity.

A packet hash is the evidence that two arms read the same thing, so the hash has
to depend on the content and on nothing else — not on the order keys happened to
be built in, not on how a decimal was spelled, not on the Python version that
serialised it. Anything that varies across runs would produce two hashes for one
packet and silently break the comparison the hash exists to support.

Three rules follow, and each closes a way a hash could drift:

Keys are sorted, so a packet assembled in a different order hashes the same.

Numbers are rendered as text, never as JSON floats. `json.dumps` on a float
emits the shortest string that round-trips, which is a property of the floating
point representation rather than of the value, and a `Decimal` passed through
float at all loses exactness before it is ever written.

Instants are rendered in UTC with an explicit offset. The same moment expressed
in two zones is one instant and must hash alike, and a naive timestamp is
refused outright rather than assumed to be UTC.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, cast

#: Separators without whitespace, so formatting can never alter a digest.
_COMPACT = (",", ":")


class UnhashableValueError(TypeError):
    """Raised when a packet holds a value with no stable textual form.

    Every type admitted here renders identically on every run. A type that does
    not — a set, whose iteration order is not guaranteed across processes, or an
    arbitrary object whose `repr` carries an address — would produce a different
    hash for the same packet, which is the one failure this module exists to
    prevent. Refusing is the only safe answer.
    """


def _canonical(value: Any) -> Any:
    """Render one value in the form the digest is taken over.

    Raises:
        UnhashableValueError: the value has no stable textual form.
    """
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, Decimal):
        # Normalised so that 1.50 and 1.5 are one value, then rendered without
        # an exponent so that 1E+2 and 100 are too.
        return format(value.normalize(), "f")
    if isinstance(value, float):
        msg = (
            "a float cannot be hashed stably; carry the value as Decimal or a "
            "string, which is how every price and return in this project is held"
        )
        raise UnhashableValueError(msg)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            msg = (
                "a naive datetime has no single instant to hash; attach the zone "
                "it was published in before it reaches a packet"
            )
            raise UnhashableValueError(msg)
        return value.astimezone(UTC).isoformat()
    # Tested after `datetime`, which subclasses `date`: reversed, every instant
    # would hash as its calendar day and two filings hours apart would collide.
    if isinstance(value, date):
        # A session is a calendar day in the market's timezone, not an instant,
        # so it is rendered as one rather than given a spurious midnight.
        return value.isoformat()
    if isinstance(value, Mapping):
        mapping = cast("Mapping[Any, Any]", value)
        return {str(key): _canonical(mapping[key]) for key in sorted(mapping, key=str)}
    if isinstance(value, Sequence):
        # Order is content for a sequence: two different orderings are two
        # different packets, so they are left as they are rather than sorted.
        return [_canonical(item) for item in cast("Sequence[Any]", value)]

    msg = (
        f"{type(value).__name__} has no stable textual form, so a packet holding "
        "one would hash differently between runs"
    )
    raise UnhashableValueError(msg)


def canonical_bytes(fields: Mapping[str, Any]) -> bytes:
    """Return the exact bytes a packet's digest is taken over.

    Exposed rather than kept private because a hash that cannot be explained is
    not evidence of anything: when two packets that should match do not, the
    difference is visible here and nowhere else.
    """
    return json.dumps(
        _canonical(fields), separators=_COMPACT, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")


def content_hash(fields: Mapping[str, Any]) -> str:
    """Return the hex SHA-256 digest of a packet's contents.

    SHA-256 rather than a faster non-cryptographic hash: the digest is stored
    alongside every prediction as the claim that a given arm read given
    evidence, and that claim is worth more if it cannot be forged cheaply.
    """
    return hashlib.sha256(canonical_bytes(fields)).hexdigest()
