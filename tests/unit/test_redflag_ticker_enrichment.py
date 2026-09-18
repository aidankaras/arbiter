"""Red-flag events name their filer by CIK, so a ticker must be looked up.

Form 4 reports a ticker; an 8-K does not. Until the lookup is applied, every
red-flag event is unpriceable for a reason that has nothing to do with the
market. An issuer that genuinely has no listed ticker is recorded as such rather
than dropped, because a silently missing event and an event that could not be
priced are different facts.
"""

from datetime import UTC, datetime

import pytest

from arbiter.evaluation.resolution import with_tickers


def _row(accession: str, cik: int = 1423689, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "accession_no": accession,
        "as_of": datetime(2026, 8, 3, 20, 47, tzinfo=UTC),
        "cik": cik,
        "issuer": "AGNC Investment Corp.",
    }
    row.update(extra)
    return row


def test_a_row_without_a_ticker_gains_one_from_the_lookup():
    priced, unpriceable = with_tickers([_row("r-1")], ticker_for=lambda cik: "AGNC")

    assert unpriceable == []
    assert priced[0]["ticker"] == "AGNC"


def test_a_row_that_already_has_a_ticker_is_left_alone():
    """Form 4 rows carry their own ticker; looking it up again would cost a request."""
    asked: list[int] = []

    def lookup(cik: int) -> str | None:
        asked.append(cik)
        return "WRONG"

    priced, _ = with_tickers([_row("a-1", ticker="MO")], ticker_for=lookup)

    assert priced[0]["ticker"] == "MO"
    assert asked == [], "a row with a ticker must not trigger a lookup"


def test_an_issuer_with_no_listed_ticker_is_recorded_not_dropped():
    """A filer with no listed security is a fact about the filer, not a defect."""
    priced, unpriceable = with_tickers([_row("r-1")], ticker_for=lambda cik: None)

    assert priced == []
    assert unpriceable == [{"accession_no": "r-1", "reason": "issuer has no listed ticker"}]


def test_one_unpriceable_issuer_does_not_cost_the_others():
    def lookup(cik: int) -> str | None:
        return None if cik == 999 else "AGNC"

    priced, unpriceable = with_tickers(
        [_row("r-1"), _row("r-2", cik=999), _row("r-3")], ticker_for=lookup
    )

    assert [row["accession_no"] for row in priced] == ["r-1", "r-3"]
    assert [entry["accession_no"] for entry in unpriceable] == ["r-2"]


def test_each_issuer_is_looked_up_once_however_many_events_it_filed():
    """A day's filings concentrate in far fewer issuers than filings."""
    asked: list[int] = []

    def lookup(cik: int) -> str | None:
        asked.append(cik)
        return "AGNC"

    with_tickers([_row("r-1"), _row("r-2"), _row("r-3")], ticker_for=lookup)

    assert asked == [1423689], "the same issuer must not be looked up per event"


def test_a_row_without_a_cik_is_refused():
    """Without an identity there is nothing to look up and nothing to price."""
    row = _row("r-1")
    del row["cik"]

    with pytest.raises(KeyError, match="cik"):
        with_tickers([row], ticker_for=lambda cik: "AGNC")


def test_no_rows_yield_nothing_and_ask_nothing():
    asked: list[int] = []

    priced, unpriceable = with_tickers([], ticker_for=lambda cik: asked.append(cik) or "X")

    assert priced == []
    assert unpriceable == []
    assert asked == []


def test_a_blank_ticker_on_a_row_is_treated_as_absent():
    """Stored rows can carry an empty string where a ticker was never parsed."""
    priced, _ = with_tickers([_row("r-1", ticker="")], ticker_for=lambda cik: "AGNC")

    assert priced[0]["ticker"] == "AGNC"
