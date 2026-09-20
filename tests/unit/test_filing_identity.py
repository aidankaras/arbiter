"""One filing is one filing, and its CIK is the issuer's.

EDGAR's daily index lists a Form 4 once per reporting-owner CIK, so a purchase
filed jointly by three related entities appears three times in a day's index.
Ingested once per appearance, one transaction becomes three events.

The CIK that arrives with each of those entries is the entry's own filer, not
the issuer. That matters beyond identity: the sector benchmark is looked up by
CIK, and a reporting owner resolves to no sector and falls back to the broad
market — so the duplicate rows were also being measured against SPY while the
real issuer resolved to its sector ETF. An abnormal return computed against the
wrong benchmark is not comparable with one computed against the right one, and
nothing downstream can tell them apart.
"""

from datetime import date, datetime

from arbiter.ingestion.edgar import FilingRecord, records_from_filings


class _Header:
    def __init__(self, accepted: datetime | None) -> None:
        self.acceptance_datetime = accepted


class _IndexEntry:
    """One appearance of a filing in a day's index, under one filer CIK."""

    def __init__(self, accession_no: str, cik: int) -> None:
        self.accession_no = accession_no
        self.cik = cik
        self.form = "4"
        self.company = "NET Power Inc."
        self.filing_date = date(2026, 3, 2)
        self.header = _Header(datetime(2026, 3, 2, 18, 0))


ACCESSION = "0001104659-26-022377"


def test_one_filing_listed_under_three_owners_becomes_one_record():
    """The duplication that inflated the event store.

    The index entries differ only in filer CIK; the filing is the same document
    and describes the same transactions.
    """
    entries = [_IndexEntry(ACCESSION, cik) for cik in (1981100, 1973442, 1845437)]

    records, untimestamped = records_from_filings(entries)

    assert len(records) == 1
    assert untimestamped == []


def test_two_genuinely_different_filings_are_both_kept():
    """The negative control: collapsing by accession must not collapse by day."""
    entries = [_IndexEntry(ACCESSION, 1845437), _IndexEntry("0000320193-26-000044", 320193)]

    records, _ = records_from_filings(entries)

    assert {record.accession_no for record, _ in records} == {
        ACCESSION,
        "0000320193-26-000044",
    }


def test_a_repeated_accession_is_reported_once_even_across_many_owners():
    entries = [_IndexEntry(ACCESSION, cik) for cik in range(1000, 1012)]

    records, _ = records_from_filings(entries)

    assert len(records) == 1


def test_an_untimestamped_filing_is_still_excluded_and_named():
    """Collapsing duplicates must not swallow the exclusion record."""

    class _Undated(_IndexEntry):
        def __init__(self) -> None:
            super().__init__("0000000000-26-000001", 1)
            self.header = _Header(None)

    records, untimestamped = records_from_filings([_Undated()])

    assert records == []
    assert untimestamped == ["0000000000-26-000001"]


def test_a_filing_record_carries_the_accession_it_was_built_from():
    entries = [_IndexEntry(ACCESSION, 1845437)]

    records, _ = records_from_filings(entries)
    record, _source = records[0]

    assert isinstance(record, FilingRecord)
    assert record.accession_no == ACCESSION
