"""Partitions written before a field existed must still read and label.

`ticker`/`ticker_source` on red-flag events and `owner_ciks` on insider events
were added after days had been ingested. Those partitions lack the columns until
a re-ingest, and every consumer reads rows as plain dicts, so an absent key must
mean "not recorded" rather than a crash.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from arbiter.evaluation.resolution import requests_from_rows, with_tickers
from arbiter.events.insider import InsiderEvent
from arbiter.events.redflags import RedFlagEvent
from arbiter.ingestion.store import read_events, write_events

DAY = date(2026, 9, 11)
AS_OF = datetime(2026, 9, 11, 20, 47, tzinfo=UTC)


def _write_old_redflag(root: Path) -> None:
    """The red-flag columns as they were before `ticker` existed."""
    table = pa.Table.from_pylist(
        [
            {
                "accession_no": "r-1",
                "as_of": AS_OF,
                "cik": 1423689,
                "issuer": "AGNC Investment Corp.",
                "item_codes": ["5.02"],
                "item_text": "Item 5.02",
                "has_press_release": False,
            }
        ]
    )
    path = root / "redflag" / f"{DAY.isoformat()}.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(table, path)


def _insider(**extra: object) -> InsiderEvent:
    fields: dict[str, object] = {
        "accession_no": "a-1",
        "as_of": AS_OF,
        "cik": 764180,
        "ticker": "MO",
        "issuer": "Altria",
        "insider_name": "A / B",
        "position": "Officer",
        "transaction_code": "P",
        "shares": Decimal(1000),
        "price": Decimal(50),
        "value_usd": Decimal(50000),
        "remaining_shares": None,
        "is_10b5_1": False,
    }
    return InsiderEvent.model_validate(fields | extra)


def test_an_old_redflag_partition_labels_through_the_current_map(tmp_path: Path):
    _write_old_redflag(tmp_path)
    (row,) = read_events(tmp_path, "redflag", DAY)
    assert "ticker" not in row, "premise: the old partition has no ticker column"

    priced, unpriceable = with_tickers([row], ticker_for=lambda cik: "AGNC")

    assert unpriceable == []
    assert (priced[0]["ticker"], priced[0]["ticker_source"]) == ("AGNC", "current_map")
    assert len(requests_from_rows(priced, "redflag")) == 1


def test_an_old_insider_partition_reads_without_owner_ciks(tmp_path: Path):
    row = _insider().model_dump(mode="python", exclude={"owner_ciks"})
    path = tmp_path / "insider" / f"{DAY.isoformat()}.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), path)

    (stored,) = read_events(tmp_path, "insider", DAY)
    assert "owner_ciks" not in stored, "premise: the old partition has no owner_ciks column"

    assert stored.get("owner_ciks") is None
    assert len(requests_from_rows([stored], "insider")) == 1


def test_new_fields_round_trip_through_the_store(tmp_path: Path):
    stated = RedFlagEvent(
        accession_no="r-1",
        as_of=AS_OF,
        cik=1423689,
        issuer="AGNC Investment Corp.",
        item_codes=("5.02",),
        item_text="Item 5.02",
        has_press_release=False,
        ticker="AGNC",
        ticker_source="filing",
    )
    silent = stated.model_copy(
        update={"accession_no": "r-2", "ticker": None, "ticker_source": None}
    )

    write_events([stated, silent], tmp_path, "redflag", DAY)
    write_events([_insider(owner_ciks=(100, 900))], tmp_path, "insider", DAY)

    rows = {r["accession_no"]: r for r in read_events(tmp_path, "redflag", DAY)}
    assert (rows["r-1"]["ticker"], rows["r-1"]["ticker_source"]) == ("AGNC", "filing")
    assert (rows["r-2"]["ticker"], rows["r-2"]["ticker_source"]) == (None, None)
    assert list(read_events(tmp_path, "insider", DAY)[0]["owner_ciks"]) == [100, 900]


def test_a_day_where_no_filing_stated_a_ticker_still_stores_a_string_column(tmp_path: Path):
    """An all-null column infers as a null type; the declared string type must win."""
    event = RedFlagEvent(
        accession_no="r-1",
        as_of=AS_OF,
        cik=1,
        issuer="X",
        item_codes=("5.02",),
        item_text="",
        has_press_release=False,
    )

    write_events([event], tmp_path, "redflag", DAY)

    schema = pq.read_schema(tmp_path / "redflag" / f"{DAY.isoformat()}.parquet")
    assert schema.field("ticker").type == pa.string()
