import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.ingestion.market import (
    Bar,
    RecentSipWindowError,
    newest_permitted_end,
    parse_bars,
    usable_news,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"
BARS = FIXTURES / "alpaca_bars.json"
NEWS = FIXTURES / "alpaca_news.json"


def _bars_payload() -> dict:
    return json.loads(BARS.read_text())


def _news_items() -> list[dict]:
    return json.loads(NEWS.read_text())["news"]


def test_the_captured_bars_fixture_has_the_fields_the_parser_reads():
    """An upstream field rename must fail here, not yield empty price series."""
    row = _bars_payload()["bars"]["MO"][0]
    assert {"c", "h", "l", "o", "t", "v"} <= set(row)


def test_bars_parse_into_aware_timestamps_and_decimals():
    bars = parse_bars(_bars_payload(), "MO")
    assert bars
    first = bars[0]
    assert isinstance(first, Bar)
    assert first.timestamp.tzinfo is not None
    assert isinstance(first.close, Decimal)


def test_prices_keep_their_reported_digits():
    """Binary float would drift across thousands of bars; money stays exact."""
    reported = _bars_payload()["bars"]["MO"][0]
    parsed = parse_bars(_bars_payload(), "MO")[0]
    assert parsed.close == Decimal(str(reported["c"]))
    assert parsed.open == Decimal(str(reported["o"]))


def test_both_the_issuer_and_its_benchmark_are_available():
    """An abnormal return needs the sector series alongside the stock's."""
    assert parse_bars(_bars_payload(), "MO")
    assert parse_bars(_bars_payload(), "XLP")


def test_an_absent_symbol_returns_no_bars_rather_than_guessing():
    assert parse_bars(_bars_payload(), "NOTLISTED") == []


def test_the_newest_permitted_end_excludes_the_current_session():
    """The consolidated feed refuses a window ending on today's session."""
    assert newest_permitted_end(today=date(2026, 9, 16)) == date(2026, 9, 15)


def test_requesting_a_window_ending_today_is_refused_before_the_call():
    """Fail in our own code with a clear reason, not on a remote 403."""
    with pytest.raises(RecentSipWindowError, match="current session"):
        newest_permitted_end(today=date(2026, 9, 16), requested_end=date(2026, 9, 16))


def test_a_window_ending_yesterday_is_permitted():
    assert newest_permitted_end(
        today=date(2026, 9, 16), requested_end=date(2026, 9, 15)
    ) == date(2026, 9, 15)


def test_news_published_after_the_event_is_excluded():
    """Evidence must predate the decision it informs."""
    as_of = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)
    kept = usable_news(_news_items(), as_of=as_of)
    assert kept
    assert all(
        article["created_at"] < as_of.isoformat().replace("+00:00", "Z") for article in kept
    )


def test_articles_edited_after_the_event_are_excluded_in_backtests():
    """An article revised later can carry information that did not yet exist."""
    items = _news_items()
    edited = dict(items[0])
    edited["created_at"] = "2026-08-01T00:00:00Z"
    edited["updated_at"] = "2026-08-30T00:00:00Z"

    as_of = datetime(2026, 8, 15, tzinfo=UTC)

    assert usable_news([edited], as_of=as_of) == []


def test_an_unedited_article_before_the_event_is_kept():
    items = _news_items()
    clean = dict(items[0])
    clean["created_at"] = "2026-08-01T00:00:00Z"
    clean["updated_at"] = "2026-08-01T00:00:00Z"

    assert usable_news([clean], as_of=datetime(2026, 8, 15, tzinfo=UTC)) == [clean]
