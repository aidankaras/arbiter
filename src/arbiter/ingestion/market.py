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

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from arbiter.config import get_settings
from arbiter.ingestion.timestamps import SEC_TIMEZONE

BARS_ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
NEWS_ENDPOINT = "https://data.alpaca.markets/v1beta1/news"

#: The consolidated feed. Never substituted, including when it refuses a window.
CONSOLIDATED_FEED = "sip"


#: A plain common-equity symbol: one to five letters, optionally a dotted share
#: class. Preferred issues, warrants, and units carry suffixes this rejects, and
#: they are outside what this study measures. Matching the shape of a valid
#: symbol rather than excluding known-bad spellings is deliberate: the filing
#: sources render unusable identifiers in forms no denylist anticipates.
_TRADEABLE_SYMBOL = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")

#: Placeholder words shaped like symbols, which the rule above cannot exclude.
_PLACEHOLDER_SYMBOLS = frozenset({"NULL", "NONE", "NAN", "NA", "UNKNOWN", "ERROR", "TBD"})


def tradeable_symbol(value: object) -> str | None:
    """Return a symbol in the form the price service uses, or `None` if unusable.

    Returning the normalised symbol rather than a verdict is what keeps the
    check and the stored value from disagreeing. Validation folds case, so a
    lowercase ticker passes; stored and sent as written it then misses a
    response keyed in upper case, and the event resolves to nothing with no
    error — the same silent loss a rejected symbol would cause, from a value
    that was accepted.

    Applied wherever a symbol enters the system, because one unusable symbol in
    a batched price request is rejected by the service and costs every other
    symbol in that request.
    """
    text = str(value or "").strip().upper()
    if _TRADEABLE_SYMBOL.match(text) and text not in _PLACEHOLDER_SYMBOLS:
        return text
    return None


def is_tradeable_symbol(value: object) -> bool:
    """Report whether a value is a symbol this project can price.

    Prefer `tradeable_symbol` anywhere the symbol is then stored or sent: this
    answers the question without handing back the form the answer was based on.
    """
    return tradeable_symbol(value) is not None


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


#: The regular-session close in the market's own timezone. Held as a local time
#: rather than a UTC offset because the offset moves twice a year: 16:00 Eastern
#: is 20:00 UTC under daylight time and 21:00 UTC under standard time.
_REGULAR_CLOSE = time(16, 0)


def session_close_instant(session: date) -> datetime:
    """Return the instant a session's closing price became public.

    Args:
        session: the calendar date of the session, in the market's timezone.

    Returns:
        The close as a timezone-aware UTC instant.

    Early closes are not modelled. The half-days around Thanksgiving and
    Christmas close at 13:00 Eastern, and this reports 16:00 for them, so a
    caller asking what was knowable at 14:00 Eastern on such a day is told
    "not yet" about a price that had in fact printed. That direction is
    deliberate: withholding a published bar weakens the evidence in a packet,
    whereas admitting an unclosed one falsifies it.
    """
    return datetime.combine(session, _REGULAR_CLOSE, tzinfo=SEC_TIMEZONE).astimezone(UTC)


def closed_bars(bars: Sequence[Bar], as_of: datetime) -> list[Bar]:
    """Return the bars whose closing price had been published by `as_of`.

    This is the truncation that makes a packet a point-in-time statement, and it
    is not the complement of `entry_sessions`. That function asks which sessions
    an event could still trade, so it selects sessions beginning after `as_of`.
    This asks which sessions an event could already read, which requires the
    session to have *ended* by then. A filing accepted at 10:00 Eastern falls
    between the two: its own session has begun, so it is not tradeable from the
    open, and has not closed, so its return is not knowable. Both answers
    exclude it, for different reasons.

    Args:
        bars: daily bars, each timestamped at its session's start as the vendor
            publishes them.
        as_of: the instant the packet describes. Must be timezone-aware; a naive
            value would be compared against an aware one and raise.
    """
    return [
        bar
        for bar in bars
        if session_close_instant(bar.timestamp.astimezone(SEC_TIMEZONE).date()) <= as_of
    ]


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


def _is_retryable(exc: BaseException) -> bool:
    """Report whether a failed request is worth attempting again.

    Transport failures and the service asking for patience are retried. A
    rejected symbol, a bad credential, or a refused window are not: retrying
    those turns an immediate, legible failure into a slow one, and a credential
    problem retried into silence would look like a day with no prices.
    """
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
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


#: A batch losing this share of its symbols to individual rejection is a
#: malformed request rather than a few uncovered tickers. Raising then keeps a
#: broken window from being reported as a day on which nothing traded.
_SYSTEMIC_REJECTION_RATE = 0.25


class SystemicRejectionError(RuntimeError):
    """Raised when a price request is rejected for its own sake.

    Distinguishes "these particular symbols are not covered" from "this request
    is wrong", which otherwise look identical: both end with symbols that have
    no prices.
    """


def _bars_isolating_rejections(
    symbols: Sequence[str], start: date, end: date, today: date
) -> tuple[dict[str, list[Bar]], list[str]]:
    """Fetch one batch, splitting it on rejection, and report what was refused.

    The refused symbols are returned rather than inferred from which series came
    back empty: a symbol that simply did not trade in the window is also empty,
    and conflating the two would let a quiet window look like a broken request.
    """
    try:
        return daily_bars(symbols, start, end, today), []
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 400:
            raise
        if len(symbols) == 1:
            return {symbols[0]: []}, [symbols[0]]

    midpoint = len(symbols) // 2
    left_bars, left_rejected = _bars_isolating_rejections(symbols[:midpoint], start, end, today)
    right_bars, right_rejected = _bars_isolating_rejections(
        symbols[midpoint:], start, end, today
    )
    return {**left_bars, **right_bars}, [*left_rejected, *right_rejected]


def daily_bars_tolerating_gaps(
    symbols: Sequence[str], start: date, end: date, today: date
) -> dict[str, list[Bar]]:
    """Fetch daily bars, isolating symbols the service will not serve.

    Batching a day's symbols into one request is what keeps the request count
    inside the per-minute rate limit, but it couples them: the service rejects
    the entire request over a single symbol it does not cover, so one thinly
    traded issuer would cost every label for that day.

    A rejected batch is therefore halved and retried until the refusal is
    attributed to individual symbols, which are then reported as having no
    prices — what callers already treat as unresolvable. A request rejected for
    its own sake fails every split alike, and is raised rather than quietly
    returning a day with no prices.

    Raises:
        SystemicRejectionError: too large a share of the symbols was refused
            individually, which indicts the request rather than the symbols.
        RecentSipWindowError: the window would cover the current session.
        MissingCredentialsError: no market data credentials are configured.
        httpx.HTTPStatusError: the service rejected the request for a reason
            other than its symbols.
    """
    if not symbols:
        return {}

    collected, rejected = _bars_isolating_rejections(symbols, start, end, today)
    if len(rejected) > len(symbols) * _SYSTEMIC_REJECTION_RATE:
        msg = (
            f"{len(rejected)} of {len(symbols)} symbols were refused for "
            f"{start.isoformat()}..{end.isoformat()}, which points at the request "
            f"rather than the symbols; first refused: {', '.join(rejected[:5])}"
        )
        raise SystemicRejectionError(msg)
    return collected


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
