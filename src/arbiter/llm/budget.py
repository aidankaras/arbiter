"""Spend and token ceilings.

A prompt asking a model to be efficient is advisory. These checks are the
enforcement: they run before work is dispatched and raise rather than trimming
the workload, because silently doing less work than a run's configuration calls
for would make that run's results incomparable with earlier ones.
"""

from __future__ import annotations

from decimal import Decimal


class BudgetExceededError(RuntimeError):
    """Raised when dispatching more work would cross a configured ceiling."""


def assert_within_spend_ceiling(
    spent_usd: Decimal, pending_usd: Decimal, ceiling_usd: Decimal
) -> None:
    """Check that recorded plus pending spend stays within the daily ceiling.

    Amounts are US dollars. The ceiling is inclusive: a projected total equal to
    it is permitted.

    Raises:
        ValueError: `pending_usd` is negative, which would understate the total.
        BudgetExceededError: the projected total exceeds `ceiling_usd`.
    """
    if pending_usd < 0:
        msg = f"pending_usd must not be negative, got {pending_usd}"
        raise ValueError(msg)

    projected = spent_usd + pending_usd
    if projected > ceiling_usd:
        msg = (
            f"projected spend ${projected} exceeds the daily ceiling "
            f"${ceiling_usd}; halting before dispatch"
        )
        raise BudgetExceededError(msg)


def assert_within_token_ceiling(
    used_tokens: int, pending_tokens: int, ceiling_tokens: int
) -> None:
    """Check that used plus pending tokens stay within the per-run ceiling.

    Raises:
        ValueError: `pending_tokens` is negative, which would understate the total.
        BudgetExceededError: the projected total exceeds `ceiling_tokens`.
    """
    if pending_tokens < 0:
        msg = f"pending_tokens must not be negative, got {pending_tokens}"
        raise ValueError(msg)

    projected = used_tokens + pending_tokens
    if projected > ceiling_tokens:
        msg = (
            f"projected usage {projected} tokens exceeds the per-run ceiling "
            f"{ceiling_tokens}; halting before dispatch"
        )
        raise BudgetExceededError(msg)
