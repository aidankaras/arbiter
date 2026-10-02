"""Stored events, labels and exclusion records, shared by the evaluation tests.

One definition rather than one per file, so a field added to `InsiderEvent` or
`Label` is added once. The defaults describe one Altria director purchase that
resolved to a small positive return; tests vary only what they are about.

Not named `test_*`, so pytest does not collect it as a module of tests.
"""

from datetime import UTC, date, datetime
from decimal import Decimal

from arbiter.evaluation.resolution import Label
from arbiter.events.insider import InsiderEvent


def event(accession: str, insider: str = "Gifford Kathryn", cik: int = 764180) -> InsiderEvent:
    return InsiderEvent(
        accession_no=accession,
        as_of=datetime(2026, 8, 3, 20, 47, tzinfo=UTC),
        cik=cik,
        ticker="MO",
        issuer="ALTRIA GROUP, INC.",
        insider_name=insider,
        position="Director",
        transaction_code="P",
        shares=Decimal("1000"),
        price=Decimal("50"),
        value_usd=Decimal("50000"),
        remaining_shares=Decimal("9000"),
        is_10b5_1=False,
    )


def label(accession: str) -> Label:
    return Label(
        accession_no=accession,
        ticker="MO",
        benchmark_symbol="XLP",
        abnormal_return=Decimal("0.012"),
        horizon_days=5,
        entry_session=date(2026, 8, 4),
        exit_session=date(2026, 8, 11),
    )


def unpriced(*accessions: str) -> list[dict[str, str]]:
    """Exclusion records as the labeling pass writes them for an unlisted issuer."""
    return [
        {"accession_no": accession, "reason": "issuer has no listed ticker"}
        for accession in accessions
    ]
