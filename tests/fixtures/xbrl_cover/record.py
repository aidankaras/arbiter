"""Record an 8-K's XBRL cover-page facts as a contract-test fixture.

Run from the repository root:

    uv run python tests/fixtures/xbrl_cover/record.py 0001423689-26-000124

A fixture is the `dei:TradingSymbol` and `dei:Security12bTitle` rows the SEC
served, with the context each was tagged in, because pairing a symbol with its
title depends on that context and a hand-built table would encode the pairing
its author assumed. A diff in a fixture is a change in the upstream contract and
is reviewed as one rather than refreshed away.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from arbiter.ingestion.edgar import configure_identity

CONCEPTS = ("dei:TradingSymbol", "dei:Security12bTitle")


def record(accession_no: str, into: Path) -> Path:
    """Write one filing's cover-page symbol and title facts to a JSON fixture."""
    from edgar import find

    configure_identity()
    xbrl = find(accession_no).xbrl()
    rows: list[dict[str, object]] = []
    for concept in CONCEPTS:
        frame = xbrl.facts.query().by_concept(concept).to_dataframe()
        if frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            rows.append(
                {
                    "concept": row["concept"],
                    "value": row["value"],
                    "context_ref": row["context_ref"],
                    "is_dimensioned": bool(row["is_dimensioned"]),
                }
            )

    path = into / f"{accession_no}.json"
    path.write_text(json.dumps({"accession_no": accession_no, "rows": rows}, indent=2) + "\n")
    return path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    written = record(sys.argv[1], Path(__file__).parent)
    sys.stdout.write(f"recorded {written}\n")
