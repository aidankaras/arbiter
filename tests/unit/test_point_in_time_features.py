"""A feature must not depend on which events turned out to be measurable.

One feature counts the insiders trading the same issuer on the same day. It is
computed over the day's filings, and the day's filings are all visible at the
moment a forecast is made. Which of them eventually resolves to a label is not:
that depends on whether the issuer could be priced over the following week,
which is information from after the filing.

Narrowing the day to its labelled events before computing features would let
that leak in, and it would do so invisibly — the feature would still look
plausible, and the model would score better than it deserves.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from arbiter.arms.evaluate import load_day
from arbiter.arms.features import FEATURE_NAMES
from arbiter.evaluation.resolution import Label
from arbiter.events.insider import InsiderEvent
from arbiter.ingestion.store import write_events

DAY = date(2026, 8, 3)


def _event(accession: str, insider: str, cik: int = 764180) -> InsiderEvent:
    return InsiderEvent(
        accession_no=accession,
        as_of=datetime(2026, 8, 3, 20, 47, tzinfo=UTC),
        cik=cik,
        ticker="MO",
        issuer="ALTRIA GROUP, INC.",
        insider_name=insider,
        position="Director",
        transaction_code="P",
        shares=Decimal("1000"),
        price=Decimal("50"),
        value_usd=Decimal("50000"),
        remaining_shares=Decimal("9000"),
        is_10b5_1=False,
    )


def _label(accession: str) -> Label:
    return Label(
        accession_no=accession,
        ticker="MO",
        benchmark_symbol="XLP",
        abnormal_return=Decimal("0.012"),
        horizon_days=5,
        entry_session=date(2026, 8, 4),
        exit_session=date(2026, 8, 11),
    )


def test_clustering_counts_the_days_filings_not_the_ones_that_resolved(tmp_path: Path):
    """Three insiders filed; only one event could be priced.

    The forecast for that one event still saw three insiders trading the issuer
    that day, because that is what was visible when it was made.
    """
    write_events(
        [
            _event("a-1", "Gifford Kathryn"),
            _event("a-2", "Willard Howard"),
            _event("a-3", "Casteen John"),
        ],
        tmp_path,
        "insider",
        DAY,
    )
    write_events([_label("a-1")], tmp_path, "labels-insider", DAY)

    loaded = load_day(tmp_path, "insider", DAY)

    assert loaded is not None
    assert len(loaded) == 1, "only the priceable event is scored"
    assert loaded.features[0]["insiders_trading_same_issuer"] == 3.0, (
        "the count must reflect the day as it looked when the forecast was made"
    )


def test_an_unlabelled_day_is_distinguishable_from_an_unprocessed_one(tmp_path: Path):
    write_events([_event("a-1", "Gifford Kathryn")], tmp_path, "insider", DAY)
    write_events([], tmp_path, "labels-insider", DAY)

    loaded = load_day(tmp_path, "insider", DAY)

    assert loaded is not None, "the day was processed; it simply resolved nothing"
    assert len(loaded) == 0


def test_a_day_never_processed_reads_as_absent(tmp_path: Path):
    assert load_day(tmp_path, "insider", DAY) is None


def test_every_declared_feature_survives_the_round_trip(tmp_path: Path):
    """A stored event must produce the same columns the model was fitted on."""
    write_events([_event("a-1", "Gifford Kathryn")], tmp_path, "insider", DAY)
    write_events([_label("a-1")], tmp_path, "labels-insider", DAY)

    loaded = load_day(tmp_path, "insider", DAY)

    assert loaded is not None
    assert set(loaded.features[0]) == set(FEATURE_NAMES)


def test_the_stored_outcome_decides_the_label_sign(tmp_path: Path):
    write_events([_event("a-1", "Gifford Kathryn")], tmp_path, "insider", DAY)
    write_events([_label("a-1")], tmp_path, "labels-insider", DAY)

    loaded = load_day(tmp_path, "insider", DAY)

    assert loaded is not None
    assert loaded.outcomes == [1]
    assert loaded.returns[0] > 0
