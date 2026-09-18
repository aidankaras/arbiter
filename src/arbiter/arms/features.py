"""Turning a filed transaction into the numbers a conventional model can use.

Every feature here is computable from the filing itself at the moment it became
public. Nothing reads a price, a later filing, or an outcome. That is the whole
constraint: the four arms are comparable only if each sees the same information
at the same instant, and a feature that quietly encodes the future would make
the baseline unbeatable and the comparison meaningless.

The features follow what the literature has found to separate informative
insider trades from uninformative ones. The strongest is whether the trade was
scheduled: transactions made under a 10b5-1 plan are arranged months ahead and
carry little information, while the discretionary ones carry most of what has
been measured. Size relative to the insider's remaining stake matters for the
same reason — selling a tenth of a holding says less than selling most of it.

Absent values are represented as an explicit indicator alongside a neutral fill
rather than imputed silently. A filing that reports no post-transaction holding
is stating that it does not say, which is different from stating zero, and a
model given a zero would read it as an insider who has sold out entirely.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

#: Feature order is fixed and exported because a model is fitted in one process
#: and scored in another; a reordering between the two would silently mismatch
#: coefficients to columns and produce plausible, wrong forecasts.
FEATURE_NAMES: tuple[str, ...] = (
    "is_purchase",
    "is_sale",
    "log_value_usd",
    "is_10b5_1",
    "fraction_of_holding",
    "reports_holding",
    "is_officer",
    "is_director",
    "is_ten_percent_owner",
    "insiders_trading_same_issuer",
)

#: Transaction codes for open-market trades. Other codes cover grants, exercises
#: and gifts, which are kept as events but are not open-market conviction.
_PURCHASE = "P"
_SALE = "S"

_OFFICER_TITLES = ("officer", "ceo", "cfo", "coo", "president", "chief", "vp", "vice president")
_DIRECTOR_TITLES = ("director",)
_TEN_PERCENT_TITLES = ("10%", "ten percent", "10 percent")


def _role_flags(position: str) -> tuple[float, float, float]:
    """Read officer, director, and large-holder status from a reported title.

    Titles are free text and one insider commonly holds several roles, so these
    are independent indicators rather than one categorical: a director who is
    also an officer is both, and collapsing that would discard the distinction
    the literature draws between them.
    """
    text = position.strip().lower()
    return (
        float(any(title in text for title in _OFFICER_TITLES)),
        float(any(title in text for title in _DIRECTOR_TITLES)),
        float(any(title in text for title in _TEN_PERCENT_TITLES)),
    )


def _fraction_of_holding(
    shares: Decimal | None, remaining: Decimal | None
) -> tuple[float, float]:
    """Return the traded share of the insider's stake, and whether it is known.

    The denominator is the holding before the transaction, reconstructed as the
    shares moved plus those remaining. A filing that reports no remaining
    holding yields a neutral 0.0 paired with a 0.0 indicator, so the model can
    distinguish "did not say" from "sold everything".
    """
    if shares is None or remaining is None:
        return 0.0, 0.0

    before = float(shares) + float(remaining)
    if before <= 0:
        return 0.0, 0.0

    return min(float(shares) / before, 1.0), 1.0


def event_features(row: Mapping[str, Any], insiders_same_issuer: int) -> dict[str, float]:
    """Return one event's features, all knowable when the filing became public.

    Args:
        row: a stored insider event.
        insiders_same_issuer: how many distinct insiders filed for this issuer
            on this day. Clustered filing is the one feature that needs the
            day's other events, and it is passed in rather than looked up so
            this stays a pure function of its inputs.

    Raises:
        KeyError: the row is missing a field every stored event carries, which
            means the reader and the writer disagree about the schema.
    """
    code = str(row["transaction_code"]).strip().upper()
    value = row["value_usd"]
    officer, director, ten_percent = _role_flags(str(row["position"]))
    fraction, reports_holding = _fraction_of_holding(row["shares"], row.get("remaining_shares"))

    return {
        "is_purchase": float(code == _PURCHASE),
        "is_sale": float(code == _SALE),
        # Transaction value spans several orders of magnitude, and what
        # distinguishes trades is closer to their ratio than their difference:
        # a $10m sale is not ten thousand times more informative than a $1k one.
        "log_value_usd": math.log1p(float(value)) if value is not None else 0.0,
        "is_10b5_1": float(bool(row["is_10b5_1"])),
        "fraction_of_holding": fraction,
        "reports_holding": reports_holding,
        "is_officer": officer,
        "is_director": director,
        "is_ten_percent_owner": ten_percent,
        "insiders_trading_same_issuer": float(insiders_same_issuer),
    }


def day_features(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    """Return features for every event of one day, in the order given.

    The clustering feature is computed here because it is the only one defined
    over the day rather than the filing: several insiders at one issuer trading
    on the same day is a stronger signal than any of them alone, and counting
    distinct insiders rather than filings keeps one person's several rows from
    reading as a crowd.
    """
    insiders_by_issuer: dict[int, set[str]] = {}
    for row in rows:
        insiders_by_issuer.setdefault(int(row["cik"]), set()).add(str(row["insider_name"]))

    counts = Counter({cik: len(names) for cik, names in insiders_by_issuer.items()})
    return [event_features(row, counts[int(row["cik"])]) for row in rows]


def as_matrix(features: Sequence[Mapping[str, float]]) -> list[list[float]]:
    """Lay features out in the fixed column order the model was fitted on."""
    return [[row[name] for name in FEATURE_NAMES] for row in features]
