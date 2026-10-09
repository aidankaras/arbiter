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

#: Returns are reported to twelve decimal places, a hundredth of a basis point.
#: Finer digits come from decimal division rather than from prices.
RETURN_PRECISION = Decimal("0.000000000001")


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
        InsufficientPriceDataError: a price is missing, non-finite, or not
            positive. A zero entry would make the return undefined; a zero exit
            would record a total loss no session printed.
    """
    stock_entry = _require(entry_price, "entry_price")
    stock_exit = _require(exit_price, "exit_price")
    index_entry = _require(benchmark_entry, "benchmark_entry")
    index_exit = _require(benchmark_exit, "benchmark_exit")

    # A traded price is finite and positive. The service's JSON admits bare
    # `NaN` and `Infinity`, and a NaN reaching a label is read downstream as a
    # non-positive outcome rather than as missing. Finiteness is tested first
    # because ordering a NaN raises rather than answering.
    for name, price in (
        ("entry_price", stock_entry),
        ("exit_price", stock_exit),
        ("benchmark_entry", index_entry),
        ("benchmark_exit", index_exit),
    ):
        if not price.is_finite() or price <= 0:
            msg = f"{name} is {price}, which no market prints; the event is unresolvable"
            raise InsufficientPriceDataError(msg)

    stock_return = (stock_exit - stock_entry) / stock_entry
    index_return = (index_exit - index_entry) / index_entry

    # Quantized deliberately, where the number is computed. Dividing decimals
    # yields as many digits as the arithmetic context allows, and past twelve
    # decimal places those digits describe the division rather than the market:
    # they are below a hundredth of a basis point. Keeping them would push false
    # precision into storage and into every statistic computed downstream.
    return (stock_return - index_return).quantize(RETURN_PRECISION)
