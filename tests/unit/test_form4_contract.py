"""Contract tests: real filings, recorded, with their expected events pinned.

Every defect this project has found in ingestion surfaced in a live run rather
than in a test, and the reason is visible in what the other tests construct.
They build a Form 4 table by hand, which means they encode what a filing was
assumed to look like. A filing that does not match the assumption is exactly the
case no hand-built fixture contains.

These fixtures are recorded from EDGAR and committed, so the table each test
reads is the table the SEC actually served. Recording them is what makes the
expected counts below meaningful: the defect that withdrew the first baseline
report stored one qualifying transaction once per line on its form, and the
fixture for that filing is the first case here.

Re-record with `tests/fixtures/form4/record.py` if a filing's parse changes; a
diff in a fixture is a change in the upstream contract and is reviewed as one.
"""

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from arbiter.events.insider import extract_insider_events, is_candidate
from arbiter.ingestion.edgar import FilingRecord
from tests.unit.form4_stub import Form4Stub

FIXTURES = Path(__file__).parent.parent / "fixtures" / "form4"

THRESHOLD = Decimal("50000")


def _load(accession: str) -> tuple[FilingRecord, Form4Stub]:
    payload = json.loads((FIXTURES / f"{accession}.json").read_text())
    record = FilingRecord(
        accession_no=payload["accession_no"],
        form=payload["form"],
        cik=payload["cik"],
        company=payload["company"],
        as_of=pd.Timestamp(payload["as_of"]).to_pydatetime(),
        filing_date=pd.Timestamp(payload["filing_date"]).date(),
    )
    return record, Form4Stub(payload)


def _candidates(accession: str):
    record, form4 = _load(accession)
    events = extract_insider_events(record, form4)
    return events, [event for event in events if is_candidate(event, THRESHOLD)]


def test_a_filing_with_three_transactions_yields_three_events():
    """NET Power: 1,881 at $2.00, 4,744 at $2.0003, 79,858 at $2.0508.

    Three genuinely different transactions on one form. They must stay three
    events; collapsing them would lose two real trades.
    """
    events, _ = _candidates("0001104659-26-022377")

    assert len(events) == 3
    assert sorted(event.shares for event in events) == [
        Decimal("1881"),
        Decimal("4744"),
        Decimal("79858"),
    ]


def test_only_the_transaction_over_the_threshold_becomes_a_candidate():
    """The regression on the defect that withdrew the first baseline report.

    Two of that filing's three transactions are worth $3,762 and $9,489 and fall
    under the $50,000 threshold. Exactly one qualifies — and the stored data held
    three identical copies of it, one per line on the form.
    """
    _, candidates = _candidates("0001104659-26-022377")

    assert len(candidates) == 1
    assert candidates[0].shares == Decimal("79858")
    assert candidates[0].value_usd == Decimal("163772.7864")


def test_one_transaction_filed_by_joint_owners_is_one_event():
    """Flagship: a $25,000,000 purchase reported once per reporting owner.

    The table carries three rows identical in every field, because the owners
    are collapsed into a single name string. Counting them separately weights one
    transaction three times, and joint filing is how funds and ten-percent owners
    file while officers file alone — so the over-weighting would track filer
    type, which is what the study asks the data to discriminate on.
    """
    _, candidates = _candidates("0001193125-26-085855")

    assert len(candidates) == 1
    assert candidates[0].value_usd == Decimal("25000000.0")


def test_conversions_without_a_price_are_not_candidates():
    """That same filing's other sixteen rows are conversions and a grant.

    A code C carries no price and is not an expression of conviction; admitting
    it would put the largest share counts in the dataset into a population
    defined as open-market trades.
    """
    events, candidates = _candidates("0001193125-26-085855")

    assert {event.transaction_code for event in events} == {"P", "C", "A"}
    assert {event.transaction_code for event in candidates} == {"P"}


def test_every_recorded_fixture_parses_and_prices_what_it_should():
    """A sweep, so adding a fixture extends the contract without new code."""
    recorded = sorted(FIXTURES.glob("*.json"))
    assert recorded, "no recorded filings; the contract tests would pass vacuously"

    for path in recorded:
        events, candidates = _candidates(path.stem)
        assert events, f"{path.stem} produced no events at all"
        for event in candidates:
            assert event.value_usd is not None
            assert event.value_usd >= THRESHOLD
            assert event.ticker == event.ticker.upper()


@pytest.mark.parametrize(
    ("accession", "expected"),
    [("0001104659-26-022377", 1), ("0001193125-26-085855", 1)],
)
def test_candidate_counts_are_pinned_per_filing(accession: str, expected: int):
    """The single assertion that would have caught the duplication.

    Pinned per filing rather than in aggregate: a count that drifts names the
    filing whose parse changed, which is what makes it actionable.
    """
    _, candidates = _candidates(accession)

    assert len(candidates) == expected


def test_two_trades_differing_only_by_date_stay_two_events():
    """A constructed table, deliberately, and the reason is worth stating.

    The recorded fixtures test what filings really contain; this tests what the
    collapsing rule must not do, and no filing recorded so far has the shape. An
    insider buying the same size at the same price on two days, ending at the
    same holding, is two trades — dropping the date from the identity would make
    them one, silently, and in the direction that loses real observations rather
    than the direction that duplicates them.
    """
    record, _ = _load("0001104659-26-022377")
    rows = [
        {
            "Insider": "Jane Roe",
            "Position": "Officer",
            "Code": "P",
            "Shares": 1000,
            "Price": 50.0,
            "Value": 50000.0,
            "Remaining Shares": 4000,
            "Date": day,
            "Issuer": "Altria Group Inc.",
            "Ticker": "MO",
        }
        for day in ("2026-03-02", "2026-03-03")
    ]
    form4 = Form4Stub({"rows": rows, "aff10b5_one": False, "issuer_cik": 764180})

    events = extract_insider_events(record, form4)

    assert len(events) == 2


def test_no_two_candidates_from_one_filing_are_identical():
    """Two events indistinguishable in every field are one event counted twice.

    Stated as a property over every fixture rather than a case, because the
    shapes that produce it — joint owners, repeated lines — are upstream
    decisions no hand-written fixture would anticipate.
    """
    for path in sorted(FIXTURES.glob("*.json")):
        _, candidates = _candidates(path.stem)
        identities = [
            (
                event.insider_name,
                event.transaction_code,
                event.shares,
                event.price,
                event.remaining_shares,
            )
            for event in candidates
        ]

        assert len(identities) == len(set(identities)), (
            f"{path.stem} produced duplicate candidates, which would weight one "
            "transaction more than once"
        )


def test_an_event_carries_the_issuers_cik_not_the_index_entrys():
    """The CIK decides the benchmark, so the wrong one changes the measurement.

    NET Power's Form 4 appears in the day's index three times, under CIKs
    1981100, 1973442 and 1845437. Only the last is the issuer; the others are
    reporting owners, and `benchmark_for_issuer` resolves them to no sector and
    falls back to the broad market. Events from the same filing were therefore
    measured against SPY or XLK depending on which index entry they arrived
    through, and an abnormal return against the wrong benchmark is not
    comparable with one against the right one.
    """
    events, _ = _candidates("0001104659-26-022377")

    assert {event.cik for event in events} == {1845437}


def test_the_issuer_cik_resolves_to_a_sector_rather_than_the_broad_market():
    """The consequence, asserted rather than assumed.

    A fallback to SPY is a legitimate answer for an issuer with no sector
    mapping, which is exactly why it cannot be distinguished from a lookup made
    with the wrong CIK unless something pins it.
    """
    from arbiter.ingestion.sectors import benchmark_for_issuer

    events, _ = _candidates("0001104659-26-022377")
    (cik,) = {event.cik for event in events}

    assert benchmark_for_issuer(cik) == "XLK"
    assert benchmark_for_issuer(1973442) == "SPY", "a reporting owner, for contrast"
