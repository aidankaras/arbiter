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
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

#: Python types on an event model, mapped to the stored column type.
#:
#: The decimal scale is wide because parsed cells reach this module as strings
#: produced from floats, which carry up to seventeen significant digits. A
#: narrower scale rejects those values outright rather than rounding them, so
#: the column must hold every digit the parser can hand it.
_ARROW_TYPES: dict[str, pa.DataType] = {
    "str": pa.string(),
    "int": pa.int64(),
    "bool": pa.bool_(),
    "datetime": pa.timestamp("us", tz="UTC"),
    "date": pa.date32(),
    "Decimal": pa.decimal128(38, 18),
}


def _scalar_name(annotation: object) -> str | None:
    """Return the scalar type name an annotation denotes, or `None`.

    Matching is exact rather than by substring. `tuple[str, ...]` contains the
    text "str" while being nothing of the kind, and declaring it a string column
    hands Arrow a tuple where it expects bytes.
    """
    text = str(annotation)
    for prefix in ("<class '", "typing.Optional["):
        text = text.removeprefix(prefix)
    text = text.removesuffix("'>").removesuffix("]")
    # Optional fields arrive as a union; every column is nullable anyway.
    candidates = [part.strip() for part in text.split("|")]
    names = [part.rsplit(".", 1)[-1] for part in candidates if part.strip() != "None"]
    if len(names) != 1:
        return None
    return names[0] if names[0] in _ARROW_TYPES else None


def _schema_for(model: type[BaseModel]) -> pa.Schema:
    """Derive a stable Arrow schema from an event model's fields.

    Only scalar fields are declared. A collection field, such as a tuple of item
    codes, is left for Arrow to infer, because inference is safe for a shape
    that carries no precision to lose. Scalars are declared precisely because
    inference reads their width from whichever values a given day happened to
    contain.
    """
    fields: list[pa.Field] = []
    for name, info in model.model_fields.items():
        scalar = _scalar_name(info.annotation)
        if scalar is None:
            continue
        fields.append(pa.field(name, _ARROW_TYPES[scalar], nullable=True))
    return pa.schema(fields)


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
    # Scalar columns are declared from the event model rather than inferred from
    # the day's values. Inference reads decimal precision from whatever happened
    # to arrive, so two days can disagree and the dataset stops being readable as
    # a whole. Collection columns are left to inference and appended here, since
    # a declared subset of columns would drop the rest.
    declared = _schema_for(type(events[0])) if events else None
    table = pa.Table.from_pylist(rows) if rows else pa.table({})
    if declared is not None:
        table = table.cast(
            pa.schema(
                [
                    declared.field(field.name) if field.name in declared.names else field
                    for field in table.schema
                ]
            )
        )

    # Written beside the target and moved into place, because a crash midway
    # through a direct write leaves a truncated file that reads as a complete day.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        staged = Path(handle.name)
    try:
        pq.write_table(table, staged)
        staged.replace(path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return path


def write_unpriceable(
    records: Sequence[Mapping[str, str]], root: Path, domain: str, day: date, stage: str
) -> Path:
    """Record the day's events that could not be priced, and return the path.

    `stage` names the pass that excluded them — extraction rejects a filer with
    no listed stock, labeling rejects an issuer whose ticker cannot be resolved.
    The two run back to back over the same day, so writing them to one path
    meant the second silently erased the first.

    Kept beside the labels rather than inside them: an event dropped for want of
    a listed security is not a measurement, but neither is it nothing. Without
    this record a day thinned by unlistable issuers looks identical to a quiet
    one, and the difference decides whether a gap in the dataset is a finding or
    a defect.

    Written for every processed day, including days with nothing to report, for
    the same reason an empty partition is still written.
    """
    path = root / "unpriceable" / stage / domain / f"{day.isoformat()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        staged = Path(handle.name)
    try:
        staged.write_text(json.dumps(list(records), indent=1) + "\n", encoding="utf-8")
        staged.replace(path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return path


def partition_exists(root: Path, domain: str, day: date) -> bool:
    """Report whether a day has been processed for a domain.

    Distinct from whether it holds events: an empty partition is written for a
    day that was processed and yielded nothing, and that is exactly the case
    this must report as done so a resumed run does not fetch it again.
    """
    return _partition(root, domain, day).exists()


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
