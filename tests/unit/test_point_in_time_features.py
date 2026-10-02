"""A feature must not depend on which events turned out to be measurable.

One feature counts the insiders trading the same issuer on the same day. For
each event it is computed over the filings accepted by that event's own
acceptance, all of which are visible when its forecast is made. Which of them
eventually resolves to a label is not:
that depends on whether the issuer could be priced over the following week,
which is information from after the filing.

Narrowing the day to its labelled events before computing features would let
that leak in, and it would do so invisibly — the feature would still look
plausible, and the model would score better than it deserves.
"""

from datetime import date
from pathlib import Path

from arbiter.arms.evaluate import load_day
from arbiter.arms.features import FEATURE_NAMES
from arbiter.ingestion.store import write_events, write_unpriceable
from tests.unit.stored_rows import event, label, unpriced

DAY = date(2026, 8, 3)


def test_clustering_counts_the_days_filings_not_the_ones_that_resolved(tmp_path: Path):
    """Three insiders filed; only one event could be priced.

    The forecast for that one event still saw three insiders trading the issuer
    that day, because that is what was visible when it was made.
    """
    write_events(
        [
            event("a-1", "Gifford Kathryn"),
            event("a-2", "Willard Howard"),
            event("a-3", "Casteen John"),
        ],
        tmp_path,
        "insider",
        DAY,
    )
    write_events([label("a-1")], tmp_path, "labels-insider", DAY)
    write_unpriceable(unpriced("a-2", "a-3"), tmp_path, "insider", DAY, stage="labeling")

    loaded = load_day(tmp_path, "insider", DAY)

    assert loaded is not None
    assert len(loaded) == 1, "only the priceable event is scored"
    assert loaded.unlabelled == 2
    assert loaded.features[0]["insiders_trading_same_issuer"] == 3.0, (
        "the count must reflect the day as it looked when the forecast was made"
    )


def test_an_unlabelled_day_is_distinguishable_from_an_unprocessed_one(tmp_path: Path):
    write_events([event("a-1", "Gifford Kathryn")], tmp_path, "insider", DAY)
    write_events([], tmp_path, "labels-insider", DAY)
    write_unpriceable(unpriced("a-1"), tmp_path, "insider", DAY, stage="labeling")

    loaded = load_day(tmp_path, "insider", DAY)

    assert loaded is not None, "the day was processed; it simply resolved nothing"
    assert len(loaded) == 0
    assert loaded.unlabelled == 1


def test_a_day_never_processed_reads_as_absent(tmp_path: Path):
    assert load_day(tmp_path, "insider", DAY) is None


def test_every_declared_feature_survives_the_round_trip(tmp_path: Path):
    """A stored event must produce the same columns the model was fitted on."""
    write_events([event("a-1", "Gifford Kathryn")], tmp_path, "insider", DAY)
    write_events([label("a-1")], tmp_path, "labels-insider", DAY)

    loaded = load_day(tmp_path, "insider", DAY)

    assert loaded is not None
    assert set(loaded.features[0]) == set(FEATURE_NAMES)


def test_the_stored_outcome_decides_the_label_sign(tmp_path: Path):
    write_events([event("a-1", "Gifford Kathryn")], tmp_path, "insider", DAY)
    write_events([label("a-1")], tmp_path, "labels-insider", DAY)

    loaded = load_day(tmp_path, "insider", DAY)

    assert loaded is not None
    assert loaded.outcomes == [1]
    assert loaded.returns[0] > 0
