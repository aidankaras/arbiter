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

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import httpx

from arbiter.config import get_settings
from arbiter.ingestion.timestamps import SEC_TIMEZONE

BARS_ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
NEWS_ENDPOINT = "https://data.alpaca.markets/v1beta1/news"

#: The consolidated feed. Never substituted, including when it refuses a window.
CONSOLIDATED_FEED = "sip"


class MissingCredentialsError(RuntimeError):
    """Raised when market data is requested without configured credentials."""


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


def current_session_date() -> date:
    """Return today's date in the market's own timezone.

    The feed's restriction is defined against the current US market session, so
    a local civil date is the wrong input: on a UTC host after 20:00 Eastern it
    is already tomorrow, which would let a window cover the live session and
    defeat the rule this module exists to enforce.
    """
    return datetime.now(SEC_TIMEZONE).date()


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


def _credentials() -> dict[str, str]:
    """Return the request headers carrying the market data credentials.

    Raises:
        MissingCredentialsError: the environment supplies no market data keys,
            so a caller would otherwise receive an opaque authentication
            failure from the remote service.
    """
    settings = get_settings()
    if settings.alpaca_api_key is None or settings.alpaca_secret_key is None:
        msg = "ALPACA_API_KEY and ALPACA_SECRET_KEY are required for market data"
        raise MissingCredentialsError(msg)
    return {
        "APCA-API-KEY-ID": settings.alpaca_api_key.get_secret_value(),
        "APCA-API-SECRET-KEY": settings.alpaca_secret_key.get_secret_value(),
    }


def daily_bars(
    symbols: Sequence[str], start: date, end: date, today: date
) -> dict[str, list[Bar]]:
    """Fetch daily bars for each symbol across a closed window.

    The window is validated against the feed rule before any request is made, so
    an attempt to price the current session fails here with a legible reason.

    Raises:
        RecentSipWindowError: the window would cover the current session.
        MissingCredentialsError: no market data credentials are configured.
        httpx.HTTPStatusError: the service rejected the request.
    """
    newest_permitted_end(today, end)

    headers = _credentials()
    collected: dict[str, list[Bar]] = {symbol: [] for symbol in symbols}
    page_token: str | None = None

    # The response is capped and continues under `next_page_token`. Reading only
    # the first page would silently return a short series, or none at all for a
    # symbol that fell past the cap, and `parse_bars` cannot distinguish that
    # from a ticker that did not trade.
    while True:
        params: dict[str, str | int] = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": CONSOLIDATED_FEED,
            "adjustment": "all",
            "limit": 10000,
        }
        if page_token:
            params["page_token"] = page_token

        response = httpx.get(BARS_ENDPOINT, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()

        for symbol in symbols:
            collected[symbol].extend(parse_bars(payload, symbol))

        next_token = payload.get("next_page_token")
        if not next_token:
            return collected
        page_token = str(next_token)


def recent_news(
    symbol: str, start: datetime, as_of: datetime, limit: int = 50
) -> list[dict[str, Any]]:
    """Fetch news for one symbol and return only what predates `as_of` unrevised.

    Filtering happens here rather than at the call site, so no caller can hold a
    list of articles that includes evidence postdating the event.

    Raises:
        MissingCredentialsError: no market data credentials are configured.
        httpx.HTTPStatusError: the service rejected the request.
    """
    response = httpx.get(
        NEWS_ENDPOINT,
        params={
            "symbols": symbol,
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": as_of.isoformat().replace("+00:00", "Z"),
            "limit": limit,
        },
        headers=_credentials(),
        timeout=60,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    articles: list[dict[str, Any]] = payload.get("news") or []
    return usable_news(articles, as_of=as_of)


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
