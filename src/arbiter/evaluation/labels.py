"""Outcome labels.

The label is a five- or twenty-session abnormal return: the issuer's simple
return over the window, less its sector benchmark's return over the same window.
Subtracting the benchmark is what makes the number a statement about the event
rather than about the market, so a stock that fell less than its sector counts
as outperformance.

Events without complete price coverage are unresolvable and excluded. They are
never assigned a zero return: substituting a neutral value for a missing one
biases every statistic computed from it toward that substitute, and no
downstream test can detect the difference afterwards.
"""

from __future__ import annotations

from decimal import Decimal


class InsufficientPriceDataError(ValueError):
    """Raised when an event cannot be labelled from the available prices.

    Callers mark the event unresolvable and exclude it from metrics rather than
    supplying a default.
    """


def _require(value: Decimal | None, name: str) -> Decimal:
    """Return a required price, naming it if absent.

    Raises:
        InsufficientPriceDataError: the price is missing.
    """
    if value is None:
        msg = f"{name} is unavailable; the event cannot be labelled"
        raise InsufficientPriceDataError(msg)
    return value


def abnormal_return(
    entry_price: Decimal | None,
    exit_price: Decimal | None,
    benchmark_entry: Decimal | None,
    benchmark_exit: Decimal | None,
) -> Decimal:
    """Return the issuer's excess simple return over its benchmark.

    All four prices come from the same sessions: entry at the first open after
    the decision, exit at the close of the horizon's final session.

    Returns a simple return difference in decimal form, not basis points.

    Raises:
        InsufficientPriceDataError: a price is missing, or an entry price is
            zero, which would make a return undefined rather than extreme.
    """
    stock_entry = _require(entry_price, "entry_price")
    stock_exit = _require(exit_price, "exit_price")
    index_entry = _require(benchmark_entry, "benchmark_entry")
    index_exit = _require(benchmark_exit, "benchmark_exit")

    for name, price in (("entry_price", stock_entry), ("benchmark_entry", index_entry)):
        if price == 0:
            msg = f"{name} is zero, so a return is undefined; the event is unresolvable"
            raise InsufficientPriceDataError(msg)

    stock_return = (stock_exit - stock_entry) / stock_entry
    index_return = (index_exit - index_entry) / index_entry
    return stock_return - index_return
