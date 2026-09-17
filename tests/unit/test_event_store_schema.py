"""The store must stay readable as a whole, across days and across crashes.

Both properties below were silently absent: the schema was inferred per day, so
two days could disagree and the dataset would fail to open together; and writes
went straight to the target path, so an interrupted write left a truncated file
that read as a complete day.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from arbiter.events.insider import InsiderEvent
from arbiter.events.redflags import RedFlagEvent
from arbiter.ingestion.store import read_events, write_events

DAY_ONE = date(2026, 9, 10)
DAY_TWO = date(2026, 9, 11)


def _event(accession: str, price: str, value: str) -> InsiderEvent:
    return InsiderEvent(
        accession_no=accession,
        as_of=datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        cik=764180,
        ticker="MO",
        issuer="ALTRIA GROUP, INC.",
        insider_name="Kathryn B. Mcquade",
        position="Director",
        transaction_code="P",
        shares=Decimal("1500"),
        price=Decimal(price),
        value_usd=Decimal(value),
        remaining_shares=Decimal("114929"),
        is_10b5_1=False,
    )


def test_days_with_different_price_precision_remain_readable_together(tmp_path: Path):
    """Inferred schemas disagreed on decimal scale and broke the whole dataset."""
    write_events([_event("a-1", "67.62", "101430.0")], tmp_path, "insider", DAY_ONE)
    write_events([_event("b-1", "1.23456789", "1851.85")], tmp_path, "insider", DAY_TWO)

    combined = pq.read_table(tmp_path / "insider")

    assert combined.num_rows == 2
    assert {str(row) for row in combined.column("accession_no").to_pylist()} == {"a-1", "b-1"}


def test_the_stored_schema_does_not_depend_on_the_day(tmp_path: Path):
    write_events([_event("a-1", "67.62", "101430.0")], tmp_path, "insider", DAY_ONE)
    write_events([_event("b-1", "1.23456789", "1851.85")], tmp_path, "insider", DAY_TWO)

    first = pq.read_schema(tmp_path / "insider" / f"{DAY_ONE.isoformat()}.parquet")
    second = pq.read_schema(tmp_path / "insider" / f"{DAY_TWO.isoformat()}.parquet")

    assert first.field("price").type == second.field("price").type
    assert first.field("as_of").type == second.field("as_of").type


def test_high_precision_values_survive_the_round_trip(tmp_path: Path):
    """A declared scale must not silently truncate what a filing reported."""
    write_events([_event("a-1", "1.23456789", "1851.85")], tmp_path, "insider", DAY_ONE)

    stored = read_events(tmp_path, "insider", DAY_ONE)[0]

    assert stored["price"] == Decimal("1.23456789")


def _redflag(accession: str) -> RedFlagEvent:
    return RedFlagEvent(
        accession_no=accession,
        as_of=datetime(2026, 9, 11, 20, 47, 39, tzinfo=UTC),
        cik=1423689,
        issuer="AGNC Investment Corp.",
        item_codes=("4.02", "5.02"),
        item_text="Item 5.02 Departure of Directors...",
        has_press_release=False,
    )


def test_events_carrying_a_collection_field_round_trip(tmp_path: Path):
    """A tuple column must be inferred, not declared a string and handed a tuple."""
    write_events([_redflag("r-1")], tmp_path, "redflag", DAY_ONE)

    stored = read_events(tmp_path, "redflag", DAY_ONE)[0]

    assert list(stored["item_codes"]) == ["4.02", "5.02"]
    assert stored["has_press_release"] is False


def test_scalar_columns_are_still_declared_on_a_collection_bearing_model(tmp_path: Path):
    """The tuple falls through to inference; the scalars beside it must not."""
    write_events([_redflag("r-1")], tmp_path, "redflag", DAY_ONE)
    write_events([_redflag("r-2")], tmp_path, "redflag", DAY_TWO)

    first = pq.read_schema(tmp_path / "redflag" / f"{DAY_ONE.isoformat()}.parquet")
    second = pq.read_schema(tmp_path / "redflag" / f"{DAY_TWO.isoformat()}.parquet")

    assert first.field("as_of").type == second.field("as_of").type
    assert first.field("cik").type == pa.int64()


def test_a_failed_write_leaves_no_partial_partition(tmp_path: Path, monkeypatch):
    """A truncated file would read as a complete day, which is worse than none."""
    import pyarrow.parquet as pq_module

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pq_module, "write_table", explode)

    with pytest.raises(OSError, match="disk full"):
        write_events([_event("a-1", "67.62", "101430.0")], tmp_path, "insider", DAY_ONE)

    assert not (tmp_path / "insider" / f"{DAY_ONE.isoformat()}.parquet").exists()
    leftovers = list((tmp_path / "insider").glob("*.tmp"))
    assert leftovers == [], "a failed write must not leave staging files behind"
