"""Turning a filed transaction into the numbers a conventional model can use.

Every feature here is computable from the filing itself at the moment it became
public. Nothing reads a price, a later filing, or an outcome. That is the whole
constraint: the four arms are comparable only if each sees the same information
at the same instant, and a feature that quietly encodes the future would make
the baseline unbeatable and the comparison meaningless.

What separates informative insider trades from uninformative ones is largely
decided before these features are computed. The screen in `is_candidate` keeps
only discretionary open-market purchases and sales: trades scheduled under a
10b5-1 plan are arranged months ahead and carry little information, and grants,
exercises and gifts are not expressions of conviction at all. Every stored event
has therefore already passed that filter, so neither the plan flag nor the
excluded transaction codes vary here and neither is a feature. What remains to
distinguish events is direction, size relative to the insider's own stake, role,
and whether several insiders moved together.

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
#: `is_sale` is absent deliberately: the screen admits only purchases and sales,
#: so it would be one minus `is_purchase` and collinear with it, splitting one
#: effect across two reported weights. `is_10b5_1` is absent for the stronger
#: reason that the screen admits no scheduled trade at all, leaving the column
#: constant — a standardised constant is zeros, and its fitted weight would be
#: noise printed in the report as though it meant something.
FEATURE_NAMES: tuple[str, ...] = (
    "is_purchase",
    "log_value_usd",
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
        # Value spans orders of magnitude, and what distinguishes trades is
        # closer to their ratio than their difference.
        "log_value_usd": math.log1p(float(value)) if value is not None else 0.0,
        "fraction_of_holding": fraction,
        "reports_holding": reports_holding,
        "is_officer": officer,
        "is_director": director,
        "is_ten_percent_owner": ten_percent,
        "insiders_trading_same_issuer": float(insiders_same_issuer),
    }


class UnsupportedDomainError(NotImplementedError):
    """Raised when features are requested for a domain that has none.

    The baseline arm reads Form 4 fields — a transaction code, a price, an
    insider's title. A current report carries none of them, so the red-flag
    domain needs its own feature set built against what an 8-K actually says:
    which items were triggered, whether a press release accompanied it, and how
    much the filing says. Until that exists the arm cannot score the domain, and
    saying so is better than a `KeyError` from a field that was never there.
    """


def day_features(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    """Return features for every event of one day, in the order given.

    The clustering feature is computed here because it is the only one defined
    over the day rather than the filing: several insiders at one issuer trading
    on the same day is a stronger signal than any of them alone, and counting
    distinct insiders rather than filings keeps one person's several rows from
    reading as a crowd.

    Raises:
        UnsupportedDomainError: the rows carry no Form 4 fields, as red-flag
            events do not, so this arm has no features defined for them.
    """
    if rows and "insider_name" not in rows[0]:
        present = ", ".join(sorted(rows[0]))
        msg = (
            "these events carry no Form 4 fields, so the baseline arm has no "
            f"features for them; the row holds: {present}"
        )
        raise UnsupportedDomainError(msg)

    insiders_by_issuer: dict[int, set[str]] = {}
    for row in rows:
        insiders_by_issuer.setdefault(int(row["cik"]), set()).add(str(row["insider_name"]))

    counts = Counter({cik: len(names) for cik, names in insiders_by_issuer.items()})
    return [event_features(row, counts[int(row["cik"])]) for row in rows]


def as_matrix(features: Sequence[Mapping[str, float]]) -> list[list[float]]:
    """Lay features out in the fixed column order the model was fitted on."""
    return [[row[name] for name in FEATURE_NAMES] for row in features]
