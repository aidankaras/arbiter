"""Conversion of SEC filing timestamps into an unambiguous instant.

EDGAR publishes acceptance times in US Eastern with no timezone attached.
Reading one as UTC moves a filing four or five hours, which can carry it across
the 16:00 close and change which session a forecast is allowed to trade. Every
acceptance time therefore passes through this module exactly once, at ingestion,
and no naive datetime travels further into the system.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

SEC_TIMEZONE = ZoneInfo("America/New_York")


class MissingAcceptanceTimeError(ValueError):
    """Raised when a filing carries no acceptance time.

    Such a filing cannot be timestamped, so it is excluded rather than dated by
    a substitute such as the filing date, which carries no time of day and would
    silently place the event at midnight.
    """


def acceptance_time_utc(naive_eastern: datetime | None) -> datetime:
    """Convert an EDGAR acceptance time into a timezone-aware UTC instant.

    Args:
        naive_eastern: the header's acceptance datetime, naive and Eastern as
            EDGAR publishes it.

    Returns:
        The same instant expressed in UTC, with `tzinfo` set.

    Raises:
        MissingAcceptanceTimeError: no acceptance time was supplied.
        ValueError: the datetime already carries a timezone, which would mean
            the caller had already decided how to interpret it.
    """
    if naive_eastern is None:
        msg = "filing has no acceptance time and cannot be timestamped"
        raise MissingAcceptanceTimeError(msg)

    if naive_eastern.tzinfo is not None:
        msg = (
            "expected a naive Eastern datetime as EDGAR publishes it, got one "
            f"carrying {naive_eastern.tzinfo}"
        )
        raise ValueError(msg)

    return naive_eastern.replace(tzinfo=SEC_TIMEZONE).astimezone(UTC)
