from decimal import Decimal

import pytest

from arbiter.evaluation.labels import InsufficientPriceDataError, abnormal_return

TOLERANCE = Decimal("0.000001")


def test_abnormal_return_is_the_difference_of_simple_returns():
    """Stock 100 to 110 is +10%; benchmark 50 to 52 is +4%; abnormal is +6%."""
    result = abnormal_return(
        entry_price=Decimal("100"),
        exit_price=Decimal("110"),
        benchmark_entry=Decimal("50"),
        benchmark_exit=Decimal("52"),
    )
    assert abs(result - Decimal("0.06")) < TOLERANCE


def test_a_stock_matching_its_benchmark_has_no_abnormal_return():
    result = abnormal_return(
        entry_price=Decimal("100"),
        exit_price=Decimal("110"),
        benchmark_entry=Decimal("50"),
        benchmark_exit=Decimal("55"),
    )
    assert abs(result) < TOLERANCE


def test_underperforming_the_benchmark_is_negative():
    result = abnormal_return(
        entry_price=Decimal("100"),
        exit_price=Decimal("102"),
        benchmark_entry=Decimal("50"),
        benchmark_exit=Decimal("54"),
    )
    assert abs(result - Decimal("-0.06")) < TOLERANCE


def test_a_falling_market_can_leave_a_falling_stock_with_positive_abnormal_return():
    """The point of benchmarking: down 2% while the sector fell 5% is outperformance."""
    result = abnormal_return(
        entry_price=Decimal("100"),
        exit_price=Decimal("98"),
        benchmark_entry=Decimal("100"),
        benchmark_exit=Decimal("95"),
    )
    assert abs(result - Decimal("0.03")) < TOLERANCE


@pytest.mark.parametrize(
    "missing", ["entry_price", "exit_price", "benchmark_entry", "benchmark_exit"]
)
def test_a_missing_price_makes_the_event_unresolvable(missing: str):
    """A missing outcome is never a neutral one; the event is excluded instead."""
    prices = {
        "entry_price": Decimal("100"),
        "exit_price": Decimal("110"),
        "benchmark_entry": Decimal("50"),
        "benchmark_exit": Decimal("52"),
    }
    prices[missing] = None

    with pytest.raises(InsufficientPriceDataError, match=missing):
        abnormal_return(**prices)


def test_a_zero_entry_price_is_rejected_rather_than_dividing():
    with pytest.raises(InsufficientPriceDataError, match="entry_price is 0"):
        abnormal_return(
            entry_price=Decimal("0"),
            exit_price=Decimal("110"),
            benchmark_entry=Decimal("50"),
            benchmark_exit=Decimal("52"),
        )


def test_the_result_keeps_decimal_precision():
    result = abnormal_return(
        entry_price=Decimal("67.31"),
        exit_price=Decimal("67.19"),
        benchmark_entry=Decimal("80.00"),
        benchmark_exit=Decimal("80.40"),
    )
    assert isinstance(result, Decimal)


@pytest.mark.parametrize(
    "field", ["entry_price", "exit_price", "benchmark_entry", "benchmark_exit"]
)
@pytest.mark.parametrize(
    "raw", ["NaN", "Infinity", "-Infinity", "0", "-1"], ids=lambda raw: f"price={raw}"
)
def test_a_price_no_market_could_print_is_refused(field: str, raw: str):
    """Issue #24: the price service's JSON admits bare `NaN` and `Infinity`.

    A NaN carried into a label is read downstream as a non-positive outcome, so
    an unusable price must make the event unresolvable, never a number.
    """
    prices = {
        "entry_price": Decimal("100"),
        "exit_price": Decimal("110"),
        "benchmark_entry": Decimal("50"),
        "benchmark_exit": Decimal("52"),
    }
    # Premise: the untouched prices produce a label, so a refusal is the bad value's doing.
    assert abnormal_return(**prices).is_finite()
    prices[field] = Decimal(raw)

    with pytest.raises(InsufficientPriceDataError, match=field):
        abnormal_return(**prices)
