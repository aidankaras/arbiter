"""The decimal arithmetic every published number is computed under.

`Decimal` division, `ln` and `sqrt` round to the thread's current context, which
any caller can change. A statistic computed under precision 12 and the same one
under 28 differ in their final digits, so a packet built inside someone else's
context would hash differently from the same packet built anywhere else.

Every inexact computation therefore runs under `ARITHMETIC`, a context declared
in full here rather than derived from `decimal.DefaultContext`, which is itself
mutable process state. Results are then quantized to `STATISTIC_QUANTUM`.
"""

from __future__ import annotations

import decimal
from decimal import Decimal

#: Every field stated, so nothing is inherited from `decimal.DefaultContext`.
ARITHMETIC = decimal.Context(
    prec=28,
    rounding=decimal.ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow],
)

#: Twelve decimal places, a hundredth of a basis point. Finer digits come from
#: the arithmetic rather than from prices, and presenting them would claim a
#: precision the inputs never had.
STATISTIC_QUANTUM = Decimal("0.000000000001")


def quantized(value: Decimal) -> Decimal:
    """Round a computed statistic to `STATISTIC_QUANTUM` under `ARITHMETIC`."""
    return value.quantize(STATISTIC_QUANTUM, context=ARITHMETIC)
