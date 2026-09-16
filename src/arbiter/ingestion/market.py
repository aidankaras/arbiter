"""Market data access: daily bars and news.

Two rules govern everything here, and both exist to keep published numbers
comparable with each other.

The first is the feed rule. The data plan serves the consolidated tape for every
window except one ending on the current session, which it refuses outright.
Falling back to a single-venue feed for recent windows would mean an event's
label depended on when it happened to be computed, so recent windows are not
requested at all: resolution waits a day and every price comes from one tape.

The second is the evidence rule. An article informs a decision only if it
existed beforehand, and only if it has not been revised since. A revision can
carry information that did not exist at the decision point, and the revised text
is what the API returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

BARS_ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
NEWS_ENDPOINT = "https://data.alpaca.markets/v1beta1/news"

#: The consolidated feed. Never substituted, including when it refuses a window.
CONSOLIDATED_FEED = "sip"


class RecentSipWindowError(ValueError):
    """Raised when a price window would extend into the current session.

    The consolidated feed refuses such a window. Raising here, rather than
    letting the request fail remotely, keeps the reason legible and prevents a
    caller from quietly retrying against a single-venue feed.
    """


@dataclass(frozen=True)
class Bar:
    """One daily bar, with prices as exact decimals."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


def _decimal(value: Any) -> Decimal:
    """Convert a reported price to `Decimal` without passing through float."""
    return Decimal(str(value))


def parse_bars(payload: dict[str, Any], symbol: str) -> list[Bar]:
    """Return one symbol's bars from a bars response.

    A symbol absent from the payload yields no bars. That is a real answer for a
    ticker that did not trade in the window, and the caller decides whether an
    event lacking price coverage is resolvable.
    """
    # Annotated explicitly: the payload is untyped JSON, and without these the
    # row type would be an implicit unknown that strict checking rejects.
    by_symbol: dict[str, list[dict[str, Any]]] = payload.get("bars") or {}
    rows: list[dict[str, Any]] = by_symbol.get(symbol) or []
    return [
        Bar(
            timestamp=datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00")),
            open=_decimal(row["o"]),
            high=_decimal(row["h"]),
            low=_decimal(row["l"]),
            close=_decimal(row["c"]),
            volume=_decimal(row["v"]),
        )
        for row in rows
    ]


def newest_permitted_end(today: date, requested_end: date | None = None) -> date:
    """Return the latest window end the consolidated feed will serve.

    With no `requested_end`, returns the newest permitted end, which is the day
    before `today`. With one, returns it unchanged if permitted.

    Raises:
        RecentSipWindowError: the requested end falls on or after `today`.
    """
    latest = date.fromordinal(today.toordinal() - 1)
    if requested_end is None:
        return latest
    if requested_end >= today:
        msg = (
            f"window ending {requested_end.isoformat()} covers the current session; "
            f"the consolidated feed serves through {latest.isoformat()} only"
        )
        raise RecentSipWindowError(msg)
    return requested_end


def _instant(raw: str) -> datetime:
    """Parse an API timestamp into a timezone-aware instant."""
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def usable_news(articles: list[dict[str, Any]], as_of: datetime) -> list[dict[str, Any]]:
    """Return the articles that were available, unrevised, before `as_of`.

    An article is excluded when it was published at or after the event, and also
    when it has been edited since: the API returns current text, so a later
    revision would place information in the packet that did not exist when the
    forecast was made.
    """
    return [
        article
        for article in articles
        if _instant(str(article["created_at"])) < as_of
        and _instant(str(article["updated_at"])) <= as_of
    ]
