"""A stand-in for a parsed Form 4, shared by the extractor's unit tests.

One definition rather than five. The extractor reads three things from a parsed
filing — its table, its 10b5-1 flag, and its issuer — and a stub that omits any
of them fails in a way that looks like a code defect rather than a stale test.
The issuer was added after real filings showed the filing record's CIK is the
index entry's filer, which for a jointly filed Form 4 is a reporting owner
rather than the company.

Not named `test_*`, so pytest does not collect it as a module of tests.
"""

from __future__ import annotations

from typing import Any


class Issuer:
    """The company a Form 4 is filed about."""

    def __init__(
        self, cik: int | str, name: str = "ALTRIA GROUP, INC.", ticker: str = "MO"
    ) -> None:
        self.cik = cik
        self.name = name
        self.ticker = ticker


class Form4Stub:
    """Stands in for the parsed filing, exposing only what the extractor reads."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.aff10b5_one = payload["aff10b5_one"]
        # Defaults to the payload's own CIK so a fixture that predates the
        # issuer distinction still describes one consistent company.
        self.issuer = Issuer(payload.get("issuer_cik", payload.get("cik", 0)))

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(self._payload["rows"])
