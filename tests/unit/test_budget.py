from decimal import Decimal

import pytest

from arbiter.llm.budget import (
    BudgetExceededError,
    assert_within_spend_ceiling,
    assert_within_token_ceiling,
)


def test_spending_below_the_ceiling_is_allowed():
    assert_within_spend_ceiling(Decimal("1.50"), Decimal("0.25"), Decimal("4.00"))


def test_spending_exactly_to_the_ceiling_is_allowed():
    """The ceiling is the maximum permitted total, not the first forbidden one."""
    assert_within_spend_ceiling(Decimal("3.75"), Decimal("0.25"), Decimal("4.00"))


def test_spending_past_the_ceiling_raises_with_the_amounts_named():
    with pytest.raises(BudgetExceededError, match=r"4\.26.*4\.00"):
        assert_within_spend_ceiling(Decimal("4.01"), Decimal("0.25"), Decimal("4.00"))


def test_token_ceiling_allows_the_exact_total():
    assert_within_token_ceiling(200_000, 50_000, 250_000)


def test_token_ceiling_raises_past_the_limit():
    with pytest.raises(BudgetExceededError, match=r"250001.*250000"):
        assert_within_token_ceiling(200_001, 50_000, 250_000)


def test_negative_pending_cost_is_rejected():
    """A negative estimate would let a caller evade the ceiling."""
    with pytest.raises(ValueError, match="must not be negative"):
        assert_within_spend_ceiling(Decimal("1.00"), Decimal("-5.00"), Decimal("4.00"))


def test_negative_pending_tokens_are_rejected():
    with pytest.raises(ValueError, match="must not be negative"):
        assert_within_token_ceiling(1_000, -5_000, 250_000)
