"""A day is scored only when every event is labelled or recorded as excluded.

Events without a label are dropped from the evaluation, which is right when the
labeling pass recorded why. An event it did not record means the label partition
no longer describes the events beside it — written before an outcome window
closed, or left behind when the day was re-ingested — and scoring the day would
score a subset without saying so.

Each refusal below is paired with a case identical except for the one record
that makes it legitimate, so the test proves the guard reacts to that record
and not to something else in the fixture.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.arms.evaluate import LabelCoverageError, load_day
from arbiter.evaluation.resolution import Label
from arbiter.events.insider import InsiderEvent
from arbiter.ingestion.store import write_events, write_unpriceable

DAY = date(2026, 8, 3)


def _event(accession: str) -> InsiderEvent:
    return InsiderEvent(
        accession_no=accession,
        as_of=datetime(2026, 8, 3, 20, 47, tzinfo=UTC),
        cik=764180,
        ticker="MO",
        issuer="ALTRIA GROUP, INC.",
        insider_name="Gifford Kathryn",
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


def _store(root: Path, *, events: list[str], labels: list[str], excluded: list[str]) -> None:
    write_events([_event(accession) for accession in events], root, "insider", DAY)
    write_events([_label(accession) for accession in labels], root, "labels-insider", DAY)
    write_unpriceable(
        [
            {"accession_no": accession, "reason": "issuer has no listed ticker"}
            for accession in excluded
        ],
        root,
        "insider",
        DAY,
        stage="labeling",
    )


def test_an_unlabelled_event_with_a_recorded_exclusion_is_counted(tmp_path: Path):
    _store(tmp_path, events=["a-1", "a-2"], labels=["a-1"], excluded=["a-2"])

    loaded = load_day(tmp_path, "insider", DAY)

    assert loaded is not None
    assert len(loaded) == 1
    assert loaded.unlabelled == 1


def test_an_unlabelled_event_without_a_recorded_exclusion_is_refused(tmp_path: Path):
    _store(tmp_path, events=["a-1", "a-2"], labels=["a-1"], excluded=[])

    with pytest.raises(LabelCoverageError, match="a-2"):
        load_day(tmp_path, "insider", DAY)


def test_an_exclusion_recorded_for_another_event_does_not_cover_this_one(tmp_path: Path):
    _store(tmp_path, events=["a-1", "a-2"], labels=["a-1"], excluded=["a-9"])

    with pytest.raises(LabelCoverageError, match="a-2"):
        load_day(tmp_path, "insider", DAY)


def test_a_day_whose_labeling_pass_left_no_record_is_refused(tmp_path: Path):
    """A missing record file is not an empty one: nothing says the stage ran."""
    write_events([_event("a-1"), _event("a-2")], tmp_path, "insider", DAY)
    write_events([_label("a-1")], tmp_path, "labels-insider", DAY)

    with pytest.raises(LabelCoverageError):
        load_day(tmp_path, "insider", DAY)


def test_a_label_for_an_event_the_day_no_longer_holds_is_refused(tmp_path: Path):
    """The shape a re-ingestion leaves behind: labels from the old event set."""
    _store(tmp_path, events=["a-1"], labels=["a-1", "a-gone"], excluded=[])

    with pytest.raises(LabelCoverageError, match="a-gone"):
        load_day(tmp_path, "insider", DAY)


def test_a_fully_labelled_day_reports_nothing_unlabelled(tmp_path: Path):
    _store(tmp_path, events=["a-1", "a-2"], labels=["a-1", "a-2"], excluded=[])

    loaded = load_day(tmp_path, "insider", DAY)

    assert loaded is not None
    assert loaded.unlabelled == 0
