"""Measuring a real ingested day end to end.

Everything below runs against filings and prices as they actually are: events
ingested from EDGAR, benchmarks derived from SIC, and price windows fetched from
the consolidated tape. Unit tests over canned series cannot establish that this
path works, and in this project they repeatedly have not.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.evaluation.resolution import HORIZONS, resolve_stored_day
from arbiter.ingestion.market import current_session_date, daily_bars_tolerating_gaps
from arbiter.ingestion.pipeline import ingest_day
from arbiter.ingestion.sectors import benchmark_for_issuer
from arbiter.ingestion.store import read_events

pytestmark = pytest.mark.integration

# A settled day, far enough back that a five-session window has closed and the
# consolidated tape serves it.
SETTLED_DAY = date(2026, 8, 3)
MIN_VALUE = Decimal("50000")


def _fetch(symbols, start: date, end: date):
    """Fetch as production does: one batch, tolerating symbols the tape refuses."""
    return daily_bars_tolerating_gaps(list(symbols), start, end, today=current_session_date())


@pytest.fixture(scope="module")
def resolved(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int]:
    """Ingest and resolve one settled day once for the whole module.

    Resolution reaches live price endpoints, so repeating it per test would
    multiply the request count against a per-minute limit for no added coverage.
    """
    root = tmp_path_factory.mktemp("events")
    ingest_day(SETTLED_DAY, root, MIN_VALUE)
    written = resolve_stored_day(
        day=SETTLED_DAY,
        domain="insider",
        root=root,
        fetch_bars=_fetch,
        benchmark_for=benchmark_for_issuer,
        today=current_session_date(),
    )
    return root, written


def test_a_settled_day_of_insider_events_produces_labels(resolved: tuple[Path, int]):
    _, written = resolved

    assert written > 0, "a settled day with events should yield measurable outcomes"


def test_every_stored_label_names_its_benchmark_and_sessions(resolved: tuple[Path, int]):
    root, _ = resolved

    labels = read_events(root, "labels-insider", SETTLED_DAY)

    assert labels
    for label in labels:
        assert label["benchmark_symbol"]
        assert label["horizon_days"] == HORIZONS["insider"]
        assert label["entry_session"] > SETTLED_DAY, "entry follows the filing"
        assert label["exit_session"] > label["entry_session"]


def test_returns_are_plausible_rather_than_degenerate(resolved: tuple[Path, int]):
    """Every return identical, or every one zero, would signal a wiring fault."""
    root, _ = resolved

    returns = [
        label["abnormal_return"] for label in read_events(root, "labels-insider", SETTLED_DAY)
    ]

    assert len(set(returns)) > 1, "identical returns across issuers indicate a wiring fault"
    assert all(abs(value) < Decimal("1") for value in returns), "a 100% abnormal move is a bug"


def test_resolving_again_replaces_rather_than_duplicates(resolved: tuple[Path, int]):
    """The label pass is idempotent for the same reason ingestion is."""
    root, first = resolved

    second = resolve_stored_day(
        day=SETTLED_DAY,
        domain="insider",
        root=root,
        fetch_bars=_fetch,
        benchmark_for=benchmark_for_issuer,
        today=current_session_date(),
    )

    assert first == second
    assert len(read_events(root, "labels-insider", SETTLED_DAY)) == first
