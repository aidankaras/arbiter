"""Labels persist through the same store as events, under their own domain.

A label is produced days or weeks after the event it measures, so it is written
in a separate pass and must survive the round trip with its identity and its
exact return intact.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.evaluation.resolution import Label
from arbiter.ingestion.store import read_events, write_events

DAY = date(2026, 8, 3)


def _label(accession: str, abnormal: str = "0.0612345678") -> Label:
    return Label(
        accession_no=accession,
        abnormal_return=Decimal(abnormal),
        benchmark_symbol="XLP",
        horizon_days=5,
        entry_session=date(2026, 8, 4),
        exit_session=date(2026, 8, 11),
    )


def test_labels_round_trip_with_their_identity(tmp_path: Path):
    write_events([_label("a-1"), _label("a-2")], tmp_path, "labels-insider", DAY)

    rows = read_events(tmp_path, "labels-insider", DAY)

    assert {row["accession_no"] for row in rows} == {"a-1", "a-2"}
    assert all(row["benchmark_symbol"] == "XLP" for row in rows)


def test_the_return_keeps_every_digit_it_was_measured_with(tmp_path: Path):
    """A return rounded in storage would silently change the measurement."""
    write_events([_label("a-1", "0.0612345678")], tmp_path, "labels-insider", DAY)

    stored = read_events(tmp_path, "labels-insider", DAY)[0]

    assert stored["abnormal_return"] == Decimal("0.0612345678")


def test_a_negative_return_survives_unchanged(tmp_path: Path):
    write_events([_label("a-1", "-0.0925")], tmp_path, "labels-insider", DAY)

    assert read_events(tmp_path, "labels-insider", DAY)[0]["abnormal_return"] == Decimal(
        "-0.0925"
    )


def test_sessions_round_trip_as_dates(tmp_path: Path):
    write_events([_label("a-1")], tmp_path, "labels-insider", DAY)

    stored = read_events(tmp_path, "labels-insider", DAY)[0]

    assert stored["entry_session"] == date(2026, 8, 4)
    assert stored["exit_session"] == date(2026, 8, 11)


def test_two_days_of_labels_share_one_schema(tmp_path: Path):
    """Differing precision between days must not make the set unreadable."""
    import pyarrow.parquet as pq

    write_events([_label("a-1", "0.06")], tmp_path, "labels-insider", DAY)
    write_events([_label("b-1", "0.0612345678")], tmp_path, "labels-insider", date(2026, 8, 4))

    combined = pq.read_table(tmp_path / "labels-insider")

    assert combined.num_rows == 2


def test_labels_are_immutable(tmp_path: Path):
    label = _label("a-1")

    with pytest.raises(ValueError, match="frozen"):
        label.abnormal_return = Decimal("1")
