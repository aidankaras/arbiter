"""The information coefficient by insider type: routine versus opportunistic.

Replicates the split at the centre of Cohen, Malloy and Pomorski (2012), "Decoding
Inside Information", Journal of Finance 67(3). Insiders who trade in the same
calendar month year after year are placing trades for reasons unrelated to news
(liquidity, diversification, a habit), and the paper finds the predictive content
of insider trading sits in the remainder. This module reports the baseline's
information coefficient within each type; the model and its features do not change,
and the report presents a breakdown rather than a test of the difference.

Definitions, fixed in advance:

* An insider is a (reporting-owner CIK, issuer CIK) pair, not a person. The paper
  classifies each insider across their whole career; keying by pair means a person
  who trades at two companies has two independent histories. That is the conservative
  choice here: history at one company never vouches for routine behaviour at another,
  at the cost of leaving more insiders unclassified.
* History is the open-market purchases and sales (transaction codes P and S) on
  Forms 4 and 4/A in SEC's Insider Transactions Data Sets. Every owner named on a
  filing is credited with its transactions, because one accession can list several.
* For an event accepted in year Y, month M (US/Eastern): ROUTINE if the pair traded
  in month M of each of Y-1, Y-2 and Y-3; OPPORTUNISTIC if it traded in each of those
  three years but not in month M of all three; otherwise UNCLASSIFIED, including any
  pair with a year missing. A history trade counts only if its filing date precedes the
  event's acceptance date, and trades in year Y itself are never used.
* An event with several owners is classified only if every owner gets the same class.
* The information coefficient is credited per issuer-day, so an issuer-day takes the
  class shared by all its classified events; any disagreement makes it MIXED and it
  is excluded.

The history covers only the years it is loaded for, so an insider's classification
needs three prior years of data. The shipped history is 2023-2025, which classifies
events from 2026 only and leaves a short, pair-by-pair record.
"""

# pyarrow ships no usable stubs; the checks are relaxed for the cache read and write only,
# and the functions below declare their own types so the public surface stays checked.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import IO, Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

from arbiter.arms.evaluate import IssuerScore
from arbiter.evaluation.groups import ic_within_groups
from arbiter.evaluation.metrics import ICSummary

ROUTINE = "routine"
OPPORTUNISTIC = "opportunistic"
UNCLASSIFIED = "unclassified"
MIXED = "mixed"

#: Rows of the report, in order. The index is the group index in `ic_by_insider_type`.
GROUPS = (OPPORTUNISTIC, ROUTINE)

#: Years of prior history an insider needs, and the only years consulted.
LOOKBACK_YEARS = 3

_SEC_TZ = ZoneInfo("America/New_York")
_FORM_TYPES = frozenset({"4", "4/A"})
_OPEN_MARKET = frozenset({"P", "S"})
_ZIP_GLOB = "*_form345.zip"
#: Bump when the cached table's contents change for the same zips.
_CACHE_VERSION = "1"
_MONTHS = {
    name: number
    for number, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"), 1
    )
}

#: (owner CIK, issuer CIK).
Pair = tuple[int, int]
#: One trade: year and month of the transaction, and the filing date that made it public.
Trade = tuple[int, int, date]
#: Trade history by pair.
History = Mapping[Pair, frozenset[Trade]]
#: Issuer-day class by (day, issuer CIK).
IssuerDayClasses = Mapping[tuple[date, int], str]
#: (owner, issuer, year, month, filed).
_Row = tuple[int, int, int, int, date]


def _parse_date(text: str) -> date:
    """Parse the data sets' `DD-MON-YYYY` without depending on the locale."""
    day, month, year = text.strip().upper().split("-")
    return date(int(year), _MONTHS[month], int(day))


def _rows(handle: IO[bytes]) -> Iterator[dict[str, str]]:
    """Stream a tab-separated member as dicts. Quotes are literal characters in these files."""
    text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace", newline="")
    yield from csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE)


def _read_zip(path: Path) -> set[_Row]:
    """Return a row per owner and open-market month for every P or S on a Form 4 or 4/A.

    Filters at each member so only the accessions that matter are held: submissions
    to Forms 4 and 4/A, then their open-market rows, then those filings' owners.
    """
    with zipfile.ZipFile(path) as archive:
        filings: dict[str, tuple[int, date]] = {}
        with archive.open("SUBMISSION.tsv") as handle:
            for row in _rows(handle):
                if row["DOCUMENT_TYPE"] in _FORM_TYPES:
                    filings[row["ACCESSION_NUMBER"]] = (
                        int(row["ISSUERCIK"]),
                        _parse_date(row["FILING_DATE"]),
                    )
        trades: dict[str, set[tuple[int, int]]] = defaultdict(set)
        with archive.open("NONDERIV_TRANS.tsv") as handle:
            for row in _rows(handle):
                accession = row["ACCESSION_NUMBER"]
                if row["TRANS_CODE"] in _OPEN_MARKET and accession in filings:
                    traded = _parse_date(row["TRANS_DATE"])
                    trades[accession].add((traded.year, traded.month))
        found: set[_Row] = set()
        with archive.open("REPORTINGOWNER.tsv") as handle:
            for row in _rows(handle):
                accession = row["ACCESSION_NUMBER"]
                if accession not in trades:
                    continue
                issuer, filed = filings[accession]
                owner = int(row["RPTOWNERCIK"])
                found.update((owner, issuer, y, m, filed) for y, m in trades[accession])
    return found


def available_archives(directory: Path) -> list[Path]:
    """The history zips under `directory`, sorted; empty when there are none."""
    return sorted(directory.glob(_ZIP_GLOB)) if directory.is_dir() else []


def _cache_path(directory: Path, archives: Sequence[Path]) -> Path:
    """A cache file named by the zip set, so a changed or added zip never reuses a stale one."""
    stamp = "|".join(f"{a.name}:{a.stat().st_size}:{a.stat().st_mtime_ns}" for a in archives)
    digest = hashlib.sha256(f"{_CACHE_VERSION}|{stamp}".encode()).hexdigest()[:16]
    return directory / f"insider_history_{digest}.parquet"


def load_history(directory: Path) -> History:
    """Build the pair -> trades history from the zips under `directory`.

    The first call streams each zip; the result is written to a parquet file beside
    the zips, named by their names, sizes and modification times, and later calls read
    that instead. Raises `FileNotFoundError` when the directory holds no zips: an empty
    history would classify nothing and look like a result.
    """
    archives = available_archives(directory)
    if not archives:
        msg = f"no {_ZIP_GLOB} under {directory}"
        raise FileNotFoundError(msg)
    cache = _cache_path(directory, archives)
    rows: list[_Row]
    if cache.exists():
        table = pq.read_table(cache).to_pydict()
        rows = list(
            zip(
                table["owner"],
                table["issuer"],
                table["year"],
                table["month"],
                table["filed"],
                strict=True,
            )
        )
    else:
        union: set[_Row] = set()
        for archive in archives:
            union |= _read_zip(archive)
        rows = sorted(union)
        pq.write_table(
            pa.table(
                {
                    "owner": pa.array([r[0] for r in rows], pa.int64()),
                    "issuer": pa.array([r[1] for r in rows], pa.int64()),
                    "year": pa.array([r[2] for r in rows], pa.int32()),
                    "month": pa.array([r[3] for r in rows], pa.int32()),
                    "filed": pa.array([r[4] for r in rows], pa.date32()),
                }
            ),
            cache,
        )
    grouped: dict[Pair, set[Trade]] = defaultdict(set)
    for owner, issuer, year, month, filed in rows:
        grouped[(owner, issuer)].add((year, month, filed))
    return {pair: frozenset(trades) for pair, trades in grouped.items()}


def classify_pair(history: History, owner: int, issuer: int, accepted: datetime) -> str:
    """Classify one (owner, issuer) pair for an event accepted at `accepted`.

    Returns `ROUTINE`, `OPPORTUNISTIC` or `UNCLASSIFIED`. A naive `accepted` is read as UTC.
    """
    local = (accepted if accepted.tzinfo else accepted.replace(tzinfo=UTC)).astimezone(_SEC_TZ)
    year, month, cutoff = local.year, local.month, local.date()
    prior = range(year - LOOKBACK_YEARS, year)
    # Point-in-time: a trade is usable only once its filing was public.
    usable = [t for t in history.get((owner, issuer), ()) if t[2] < cutoff and t[0] in prior]
    if {t[0] for t in usable} != set(prior):
        return UNCLASSIFIED
    if all(any(t[0] == y and t[1] == month for t in usable) for y in prior):
        return ROUTINE
    return OPPORTUNISTIC


def classify_event(
    history: History, owners: Sequence[int], issuer: int, accepted: datetime
) -> str:
    """Classify an event by its owners; classified only when every owner agrees."""
    if not owners:
        return UNCLASSIFIED
    classes = {classify_pair(history, owner, issuer, accepted) for owner in owners}
    return classes.pop() if len(classes) == 1 else UNCLASSIFIED


def issuer_day_class(event_classes: Iterable[str]) -> str:
    """Collapse one issuer-day's event classes to a single class.

    The class shared by every classified event; `MIXED` if two classified events
    differ; `UNCLASSIFIED` if none is classified.
    """
    classified = {c for c in event_classes if c != UNCLASSIFIED}
    if not classified:
        return UNCLASSIFIED
    return classified.pop() if len(classified) == 1 else MIXED


def issuer_day_classes(
    history: History, events: Iterable[tuple[date, Mapping[str, Any]]]
) -> dict[tuple[date, int], str]:
    """Class per (day, issuer) from stored event rows.

    Rows need `cik`, `as_of` (datetime) and, when stored, `owner_ciks`; a row with no
    owners (a partition written before the field existed) is unclassified.
    """
    per: dict[tuple[date, int], list[str]] = defaultdict(list)
    for day, row in events:
        issuer = int(row["cik"])
        owners = tuple(int(owner) for owner in row.get("owner_ciks") or ())
        per[(day, issuer)].append(classify_event(history, owners, issuer, row["as_of"]))
    return {key: issuer_day_class(classes) for key, classes in per.items()}


def load_issuer_day_classes(
    root: Path, domain: str, days: Sequence[date], history: History
) -> dict[tuple[date, int], str]:
    """Classify the scored events of `days` from the event store.

    Only events with a stored label are considered, because only they reach the
    scores: an event excluded for want of a price must not make an issuer-day mixed.
    A day without an event or label partition contributes nothing.
    """
    from arbiter.ingestion.store import partition_exists, read_events

    chosen: list[tuple[date, Mapping[str, Any]]] = []
    for day in days:
        if not partition_exists(root, domain, day) or not partition_exists(
            root, f"labels-{domain}", day
        ):
            continue
        labelled = {
            str(label["accession_no"]) for label in read_events(root, f"labels-{domain}", day)
        }
        chosen.extend(
            (day, row)
            for row in read_events(root, domain, day)
            if str(row["accession_no"]) in labelled
        )
    return issuer_day_classes(history, chosen)


@dataclass(frozen=True)
class TypeRow:
    """One insider type's information coefficient across days."""

    label: str
    #: `None` when fewer than two type-days reached the threshold.
    ic: ICSummary | None
    days: int
    issuer_days: int


@dataclass(frozen=True)
class TypeBreakdown:
    """The coefficient by insider type and the sample it leaves out.

    The rows' issuer-days, `unclassified`, `mixed` and `dropped` add up to
    `scored_issuer_days`.
    """

    rows: tuple[TypeRow, ...]
    unclassified: int
    mixed: int
    #: Classified issuer-days whose type-day held fewer than `MIN_EVENTS_PER_DAY`.
    dropped: int
    scored_issuer_days: int


def ic_by_insider_type(
    scores: Mapping[date, Sequence[IssuerScore]],
    classes: IssuerDayClasses,
    overlap_lags: int = 0,
) -> TypeBreakdown:
    """Compute the information coefficient within each insider type of each day.

    Uses the same per-group machinery as the size breakdown: a type-day counts only
    with at least `MIN_EVENTS_PER_DAY` issuers, and per-day coefficients are summarised
    like the headline's. An issuer-day absent from `classes` is unclassified.
    """

    def assign(day: date, issuers: Sequence[IssuerScore]) -> dict[int, int]:
        return {
            s.cik: GROUPS.index(classes[(day, s.cik)])
            for s in issuers
            if classes.get((day, s.cik), UNCLASSIFIED) in GROUPS
        }

    grouped = ic_within_groups(scores, assign, len(GROUPS), overlap_lags=overlap_lags)
    mixed = sum(
        1
        for day, issuers in scores.items()
        for s in issuers
        if classes.get((day, s.cik)) == MIXED
    )
    return TypeBreakdown(
        rows=tuple(
            TypeRow(label, g.ic, g.days, g.issuer_days)
            for label, g in zip(GROUPS, grouped.groups, strict=True)
        ),
        unclassified=grouped.ungrouped - mixed,
        mixed=mixed,
        dropped=grouped.dropped,
        scored_issuer_days=grouped.scored_issuer_days,
    )
