from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from arbiter.ingestion.timestamps import (
    MissingAcceptanceTimeError,
    acceptance_time_utc,
)


def test_converts_a_winter_filing_from_eastern_to_utc():
    """January is EST, five hours behind UTC."""
    result = acceptance_time_utc(datetime(2026, 1, 15, 16, 47, 39))
    assert result == datetime(2026, 1, 15, 21, 47, 39, tzinfo=UTC)


def test_converts_a_summer_filing_from_eastern_to_utc():
    """September is EDT, four hours behind UTC."""
    result = acceptance_time_utc(datetime(2026, 9, 11, 16, 47, 39))
    assert result == datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC)


def test_a_filing_accepted_after_the_close_stays_after_the_close():
    """The conversion must not move a filing across the session boundary."""
    result = acceptance_time_utc(datetime(2026, 9, 11, 16, 5, 0))
    assert result.astimezone(ZoneInfo("America/New_York")).hour == 16


def test_result_is_timezone_aware():
    assert acceptance_time_utc(datetime(2026, 9, 11, 16, 47, 39)).tzinfo is not None


def test_an_already_aware_datetime_is_rejected():
    """An aware input means the caller already made an assumption; refuse to guess."""
    aware = datetime(2026, 9, 11, 16, 47, 39, tzinfo=ZoneInfo("Europe/London"))
    with pytest.raises(ValueError, match="naive"):
        acceptance_time_utc(aware)


def test_missing_acceptance_time_raises():
    with pytest.raises(MissingAcceptanceTimeError, match="acceptance"):
        acceptance_time_utc(None)
