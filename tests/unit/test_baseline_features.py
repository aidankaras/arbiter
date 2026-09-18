"""Features must encode the filing and nothing that came after it.

The comparison between arms is only meaningful if each sees the same
information at the same instant, so the property worth testing hardest is
negative: no feature may depend on a price, a later filing, or an outcome.

The rest of these tests pin the distinctions the literature says matter —
scheduled versus discretionary trades, size relative to the insider's stake,
role, and clustered filing — and the handling of fields that real filings leave
blank.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from arbiter.arms.features import FEATURE_NAMES, as_matrix, day_features, event_features


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "accession_no": "0000764180-26-000102",
        "as_of": datetime(2026, 8, 3, 20, 47, tzinfo=UTC),
        "cik": 764180,
        "ticker": "MO",
        "issuer": "ALTRIA GROUP, INC.",
        "insider_name": "Gifford Kathryn",
        "position": "Director",
        "transaction_code": "P",
        "shares": Decimal("1000"),
        "price": Decimal("50"),
        "value_usd": Decimal("50000"),
        "remaining_shares": Decimal("9000"),
        "is_10b5_1": False,
    }
    row.update(overrides)
    return row


def test_no_feature_reads_the_outcome_or_anything_after_the_filing():
    """The guarantee the whole comparison rests on."""
    with_outcome = _row()
    with_outcome["abnormal_return"] = Decimal("0.25")
    with_outcome["exit_session"] = "2026-08-11"

    assert event_features(with_outcome, 1) == event_features(_row(), 1)


def test_a_purchase_and_a_sale_are_distinguished():
    """One indicator, not two: the screen admits only these two codes, so a
    separate sale flag would be one minus this one and collinear with it."""
    purchase = event_features(_row(transaction_code="P"), 1)
    sale = event_features(_row(transaction_code="S"), 1)

    assert purchase["is_purchase"] == 1.0
    assert sale["is_purchase"] == 0.0
    assert "is_sale" not in purchase


def test_a_code_the_screen_would_reject_is_not_marked_a_purchase():
    """Grants, exercises and gifts never reach the store; nothing reads as a buy."""
    gift = event_features(_row(transaction_code="G"), 1)

    assert gift["is_purchase"] == 0.0


@pytest.mark.parametrize("excluded", ["is_10b5_1", "is_sale"])
def test_a_feature_the_screen_makes_constant_is_not_declared(excluded: str):
    """A constant column standardises to zeros and earns a weight that is noise.

    `is_candidate` admits only discretionary open-market purchases and sales.
    A scheduled-trade indicator is therefore false on every stored row, and a
    sale indicator is exactly one minus the purchase indicator. Both were
    declared features until a fit against real data showed one dead column and
    one collinear pair being printed in a report as though they meant something.
    """
    assert excluded not in FEATURE_NAMES


def test_the_declared_features_vary_across_events_the_screen_admits():
    """Every remaining column must actually distinguish one admitted event from another."""
    varied = [
        event_features(_row(transaction_code="P", position="CEO", cluster_seed=0), 3),
        event_features(
            _row(
                transaction_code="S",
                position="Director, 10% Owner",
                value_usd=Decimal("9000000"),
                remaining_shares=None,
            ),
            1,
        ),
    ]

    for name in FEATURE_NAMES:
        assert varied[0][name] != varied[1][name], f"{name} does not distinguish these events"


def test_value_enters_on_a_log_scale():
    """A ten-million-dollar sale is not ten thousand times the signal of a thousand."""
    small = event_features(_row(value_usd=Decimal("1000")), 1)["log_value_usd"]
    large = event_features(_row(value_usd=Decimal("10000000")), 1)["log_value_usd"]

    assert large > small
    assert large / small < 3


def test_the_traded_fraction_is_measured_against_the_holding_before_the_trade():
    """Selling a tenth of a stake says less than selling most of it."""
    tenth = event_features(_row(shares=Decimal("1000"), remaining_shares=Decimal("9000")), 1)

    assert tenth["fraction_of_holding"] == pytest.approx(0.1)
    assert tenth["reports_holding"] == 1.0


def test_a_filing_that_reports_no_holding_is_not_read_as_having_sold_out():
    """'Did not say' and 'holds nothing' are different statements."""
    silent = event_features(_row(remaining_shares=None), 1)

    assert silent["fraction_of_holding"] == 0.0
    assert silent["reports_holding"] == 0.0, "the model must be able to tell these apart"


def test_selling_the_entire_stake_is_bounded_at_one():
    whole = event_features(_row(shares=Decimal("1000"), remaining_shares=Decimal("0")), 1)

    assert whole["fraction_of_holding"] == 1.0


def test_a_nonsensical_holding_does_not_divide_by_zero():
    nothing = event_features(_row(shares=Decimal("0"), remaining_shares=Decimal("0")), 1)

    assert nothing["fraction_of_holding"] == 0.0
    assert nothing["reports_holding"] == 0.0


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("Director", (0.0, 1.0, 0.0)),
        ("Chief Executive Officer", (1.0, 0.0, 0.0)),
        ("CFO", (1.0, 0.0, 0.0)),
        ("10% Owner", (0.0, 0.0, 1.0)),
        ("Director, 10% Owner", (0.0, 1.0, 1.0)),
        ("President and Director", (1.0, 1.0, 0.0)),
        ("", (0.0, 0.0, 0.0)),
    ],
)
def test_roles_are_read_from_free_text_titles(position: str, expected: tuple[float, ...]):
    """One insider commonly holds several roles, so these are not exclusive."""
    features = event_features(_row(position=position), 1)

    assert (
        features["is_officer"],
        features["is_director"],
        features["is_ten_percent_owner"],
    ) == expected


def test_clustered_filing_counts_distinct_insiders_not_filings():
    """Three rows from one person is not three people acting together."""
    rows = [
        _row(insider_name="Gifford Kathryn", accession_no="a-1"),
        _row(insider_name="Gifford Kathryn", accession_no="a-2"),
        _row(insider_name="Willard Howard", accession_no="a-3"),
    ]

    computed = day_features(rows)

    assert all(row["insiders_trading_same_issuer"] == 2.0 for row in computed)


def test_clustering_is_counted_per_issuer_not_across_the_day():
    rows = [
        _row(cik=764180, insider_name="A"),
        _row(cik=764180, insider_name="B"),
        _row(cik=320193, insider_name="C"),
    ]

    computed = day_features(rows)

    assert [row["insiders_trading_same_issuer"] for row in computed] == [2.0, 2.0, 1.0]


def test_features_are_laid_out_in_the_declared_column_order():
    """A model fitted in one process and scored in another shares only this order."""
    matrix = as_matrix(day_features([_row()]))

    assert len(matrix[0]) == len(FEATURE_NAMES)
    assert matrix[0][FEATURE_NAMES.index("is_purchase")] == 1.0


def test_a_row_missing_a_stored_field_is_refused_rather_than_defaulted():
    """A silent default would mean the reader and writer disagree unnoticed."""
    incomplete = _row()
    del incomplete["position"]

    with pytest.raises(KeyError):
        event_features(incomplete, 1)


def test_no_events_yield_no_features():
    assert day_features([]) == []
