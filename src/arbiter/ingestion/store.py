"""Event storage as date-partitioned Parquet.

One file per domain per day. A re-run replaces its own partition and touches no
other, which is what makes the daily job idempotent: re-running a date produces
the same dataset rather than duplicate rows.

An empty day still writes a file. The distinction between "no events occurred"
and "this day was never processed" is the difference between a complete dataset
and a silently truncated one, and only the second is a defect.

Parquet is used directly through PyArrow rather than through a dataframe
library, because writing rows and reading them back needs nothing more, and
PyArrow preserves both timezone-aware timestamps and `Decimal` values without
passing them through binary floating point.
"""

# PyArrow ships no type information, so strict checking reports every call into
# it as unknown. The suppressions are scoped to this module, the only place the
# untyped library is touched, and the functions below declare their own types so
# the package's public surface stays fully checked.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel


def _partition(root: Path, domain: str, day: date) -> Path:
    """Return the file holding one domain's events for one day."""
    return root / domain / f"{day.isoformat()}.parquet"


def write_events(events: Sequence[BaseModel], root: Path, domain: str, day: date) -> Path:
    """Write one day's events, replacing any existing partition for that day.

    Returns the path written, including for a day on which no events occurred.
    """
    path = _partition(root, domain, day)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [event.model_dump(mode="python") for event in events]
    table = pa.Table.from_pylist(rows) if rows else pa.table({})
    pq.write_table(table, path)
    return path


def read_events(root: Path, domain: str, day: date) -> list[dict[str, Any]]:
    """Return one day's stored events.

    Raises:
        FileNotFoundError: the day has never been processed. That is distinct
            from a day on which no events occurred, which returns an empty list,
            and conflating the two would hide a gap in the dataset.
    """
    path = _partition(root, domain, day)
    if not path.exists():
        msg = f"no partition for {domain} on {day.isoformat()}; the day was never processed"
        raise FileNotFoundError(msg)

    return pq.read_table(path).to_pylist()
