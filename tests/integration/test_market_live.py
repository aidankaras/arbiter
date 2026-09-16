from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from arbiter.ingestion.market import (
    RecentSipWindowError,
    daily_bars,
    newest_permitted_end,
    recent_news,
)

pytestmark = pytest.mark.integration

# A settled window, well clear of the feed's current-session restriction.
WINDOW_START = date(2026, 8, 3)
WINDOW_END = date(2026, 8, 28)
ISSUER = "MO"
BENCHMARK = "XLP"


def test_bars_are_returned_for_the_issuer_and_its_benchmark():
    bars = daily_bars([ISSUER, BENCHMARK], WINDOW_START, WINDOW_END, today=date.today())

    assert bars[ISSUER]
    assert bars[BENCHMARK]
    assert len(bars[ISSUER]) == len(bars[BENCHMARK])


def test_live_bars_carry_aware_timestamps_and_exact_prices():
    bars = daily_bars([ISSUER], WINDOW_START, WINDOW_END, today=date.today())

    first = bars[ISSUER][0]
    assert first.timestamp.tzinfo is not None
    assert isinstance(first.close, Decimal)
    assert first.high >= first.low


def test_bars_stay_within_the_requested_window():
    """A window that silently widened would pull prices from outside the event."""
    bars = daily_bars([ISSUER], WINDOW_START, WINDOW_END, today=date.today())

    dates = [bar.timestamp.date() for bar in bars[ISSUER]]
    assert min(dates) >= WINDOW_START
    assert max(dates) <= WINDOW_END


def test_a_window_covering_the_current_session_is_refused_before_the_request():
    """The feed would reject it remotely; this fails first, with a clear reason."""
    today = date.today()

    with pytest.raises(RecentSipWindowError, match="current session"):
        daily_bars([ISSUER], today - timedelta(days=5), today, today=today)


def test_the_newest_permitted_end_is_actually_served():
    """Pins the measured rule against the live service rather than a memory of it."""
    today = date.today()
    end = newest_permitted_end(today)

    bars = daily_bars([ISSUER], end - timedelta(days=10), end, today=today)

    assert bars[ISSUER]


def test_live_news_is_filtered_to_evidence_predating_the_event():
    as_of = datetime(2026, 8, 28, tzinfo=UTC)

    articles = recent_news(ISSUER, start=datetime(2026, 8, 1, tzinfo=UTC), as_of=as_of)

    assert articles
    for article in articles:
        created = datetime.fromisoformat(article["created_at"].replace("Z", "+00:00"))
        updated = datetime.fromisoformat(article["updated_at"].replace("Z", "+00:00"))
        assert created < as_of
        assert updated <= as_of
