"""Returns are quantized where they are computed, not where they are stored.

Dividing decimals yields as many digits as the arithmetic context allows. Those
digits past a hundredth of a basis point describe the division rather than the
market, and carrying them pushed a value into storage that the column could not
represent, which failed the whole day's labels rather than rounding quietly.
"""

from decimal import Decimal

from arbiter.evaluation.labels import RETURN_PRECISION, abnormal_return


def _return(entry: str, exit_: str, benchmark_entry: str, benchmark_exit: str) -> Decimal:
    return abnormal_return(
        entry_price=Decimal(entry),
        exit_price=Decimal(exit_),
        benchmark_entry=Decimal(benchmark_entry),
        benchmark_exit=Decimal(benchmark_exit),
    )


def test_a_repeating_division_is_quantized():
    """A third of a percent has no finite decimal form; storage needs one."""
    result = _return("3", "4", "7", "8")

    assert -result.as_tuple().exponent <= 12


def test_prices_carrying_float_noise_still_produce_a_storable_return():
    """Parsed prices arrive as strings made from floats, with 17 digits."""
    result = _return("63.677400000000006", "67.31", "80.00000000000001", "80.4")

    assert -result.as_tuple().exponent <= 12


def test_the_precision_is_a_hundredth_of_a_basis_point():
    assert Decimal("0.000000000001") == RETURN_PRECISION


def test_quantization_does_not_disturb_an_exact_return():
    """Stock +10%, benchmark +4%: the answer is exact and must stay exact."""
    assert _return("100", "110", "50", "52") == Decimal("0.06")


def test_a_small_real_move_survives_quantization():
    """Twelve places is far below any move a market produces; nothing is lost."""
    result = _return("100", "100.01", "50", "50")

    assert result == Decimal("0.0001")


def test_quantization_keeps_the_sign_of_an_underperforming_return():
    assert _return("100", "98", "50", "51") < 0
