"""Measuring a real day of red-flag events end to end.

This domain was unmeasurable until the ticker lookup existed: an 8-K names its
filer by CIK alone, so every event in it failed for a reason unrelated to the
market. The horizon here is twenty sessions, since drift after an auditor change
or a restatement runs about a month.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.evaluation.resolution import HORIZONS, resolve_stored_day
from arbiter.ingestion.market import current_session_date, daily_bars_tolerating_gaps
from arbiter.ingestion.pipeline import ingest_day
from arbiter.ingestion.sectors import benchmark_for_issuer, issuer_ticker
from arbiter.ingestion.store import read_events
from arbiter.ingestion.timestamps import SEC_TIMEZONE

pytestmark = pytest.mark.integration

# Far enough back that a twenty-session window has closed and the tape serves it.
SETTLED_DAY = date(2026, 8, 3)
MIN_VALUE = Decimal("50000")


def _fetch(symbols, start: date, end: date):
    """Fetch as production does: one batch, tolerating symbols the tape refuses.

    8-K filers include trusts, shells, and recently listed issuers, so a batch
    covering a whole day will meet a symbol the tape does not serve.
    """
    return daily_bars_tolerating_gaps(list(symbols), start, end, today=current_session_date())


@pytest.fixture(scope="module")
def resolved(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int]:
    """Ingest and resolve one settled day of red flags, once for the module."""
    root = tmp_path_factory.mktemp("events")
    ingest_day(SETTLED_DAY, root, MIN_VALUE)
    written = resolve_stored_day(
        day=SETTLED_DAY,
        domain="redflag",
        root=root,
        fetch_bars=_fetch,
        benchmark_for=benchmark_for_issuer,
        today=current_session_date(),
        ticker_for=issuer_ticker,
    )
    return root, written


def test_red_flag_events_now_produce_labels(resolved: tuple[Path, int]):
    """Before the ticker lookup this domain produced none, ever."""
    _, written = resolved

    assert written > 0


def test_labels_carry_the_longer_red_flag_horizon(resolved: tuple[Path, int]):
    root, _ = resolved

    labels = read_events(root, "labels-redflag", SETTLED_DAY)

    assert labels
    assert all(label["horizon_days"] == HORIZONS["redflag"] for label in labels)
    assert HORIZONS["redflag"] == 20


def test_no_label_enters_before_its_filing_was_public(resolved: tuple[Path, int]):
    """The entry session must open after the filing was accepted, not after its index date.

    These are not the same day. EDGAR indexes a filing accepted after 17:30
    Eastern under the next business day, so a filing in the day's index can have
    become public the previous Friday evening — and entering at this day's open
    is then correct rather than early. Asserting against the index date instead
    of the acceptance instant would reject that valid observation.
    """
    root, _ = resolved
    accepted = {
        event["accession_no"]: event["as_of"]
        for event in read_events(root, "redflag", SETTLED_DAY)
    }

    for label in read_events(root, "labels-redflag", SETTLED_DAY):
        public_on = accepted[label["accession_no"]].astimezone(SEC_TIMEZONE).date()

        assert label["benchmark_symbol"]
        assert label["entry_session"] > public_on
        assert label["exit_session"] > label["entry_session"]


def test_the_index_contains_filings_accepted_before_its_own_date(resolved: tuple[Path, int]):
    """Keeps the entry-timing invariant from passing only because no filing was late.

    A day's index is not a day's filings. If this ever finds none, the invariant
    above is no longer testing the case it was written for.
    """
    root, _ = resolved

    accepted = [
        event["as_of"].astimezone(SEC_TIMEZONE).date()
        for event in read_events(root, "redflag", SETTLED_DAY)
    ]

    assert any(day < SETTLED_DAY for day in accepted), (
        "no filing in this index predates it, so after-hours acceptance is untested here"
    )


def test_returns_are_plausible_rather_than_degenerate(resolved: tuple[Path, int]):
    root, _ = resolved

    returns = [
        label["abnormal_return"] for label in read_events(root, "labels-redflag", SETTLED_DAY)
    ]

    assert len(set(returns)) > 1, "identical returns across issuers indicate a wiring fault"
    assert all(abs(value) < Decimal("1") for value in returns)


def test_unpriceable_issuers_are_recorded_rather_than_dropped(resolved: tuple[Path, int]):
    """A filer with no listed security is a fact, and the record proves it was seen."""
    root, _ = resolved

    record = root / "unpriceable" / "redflag" / f"{SETTLED_DAY.isoformat()}.json"

    assert record.exists(), "every run records what it could not price, including none"
