"""Record a Form 4 from EDGAR as a contract-test fixture.

Run from the repository root:

    uv run python tests/fixtures/form4/record.py 0001104659-26-022377

A fixture is the table the SEC actually served, so a diff in one is a change in
the upstream contract and is reviewed as a change rather than refreshed away.
Record a filing when it exhibits a shape the suite does not yet cover — several
transactions on one form, joint reporting owners, a missing price, a holdings
row — and name in the test what that shape is.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from arbiter.ingestion.edgar import configure_identity, to_record


def record(accession_no: str, into: Path) -> Path:
    """Write one filing's parsed table and identity to a JSON fixture."""
    from edgar import find

    configure_identity()
    filing = find(accession_no)
    parsed = filing.obj()
    identity = to_record(filing)

    payload = {
        "accession_no": identity.accession_no,
        "form": identity.form,
        "cik": identity.cik,
        "company": identity.company,
        "as_of": identity.as_of.isoformat(),
        "filing_date": identity.filing_date.isoformat(),
        "aff10b5_one": bool(parsed.aff10b5_one),
        "rows": json.loads(parsed.to_dataframe().to_json(orient="records", date_format="iso")),
    }

    path = into / f"{accession_no}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    written = record(sys.argv[1], Path(__file__).parent)
    # A command-line tool reports on stdout; the lint rule against `print` is
    # written for library code and this file is only ever run directly.
    sys.stdout.write(f"recorded {written}\n")
