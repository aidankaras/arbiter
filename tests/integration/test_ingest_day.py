from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.ingestion.pipeline import ingest_day
from arbiter.ingestion.store import read_events

pytestmark = pytest.mark.integration

# A settled trading day: EDGAR does not revise the filings accepted on a
# completed day, so these assertions stay stable over time.
SETTLED_DAY = date(2026, 9, 11)
MIN_VALUE = Decimal("50000")


@pytest.fixture(scope="module")
def ingested(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, int], Path]:
    """Ingest the settled day once for the whole module.

    Ingestion fetches and parses every Form 4 and 8-K accepted that day, which
    is minutes of work. Repeating it per test would buy no additional coverage.
    """
    root = tmp_path_factory.mktemp("events")
    counts = ingest_day(SETTLED_DAY, root, MIN_VALUE)
    return counts, root


def test_ingesting_a_settled_day_produces_events_in_both_domains(
    ingested: tuple[dict[str, int], Path],
):
    counts, _ = ingested
    assert counts["insider"] > 0
    assert counts["redflag"] > 0
    assert set(counts) == {"insider", "redflag", "rejected", "unpriceable"}


def test_stored_insider_events_respect_the_filter_rules(
    ingested: tuple[dict[str, int], Path],
):
    """Every stored trade must be an open-market trade above the size floor."""
    _, root = ingested
    rows = read_events(root, "insider", SETTLED_DAY)

    assert rows
    assert all(row["transaction_code"] in {"P", "S"} for row in rows)
    assert all(row["is_10b5_1"] is False for row in rows)
    assert all(row["value_usd"] is not None for row in rows)
    assert all(row["value_usd"] >= MIN_VALUE for row in rows)


def test_stored_redflag_events_all_carry_a_trigger_item(
    ingested: tuple[dict[str, int], Path],
):
    _, root = ingested
    rows = read_events(root, "redflag", SETTLED_DAY)

    assert rows
    for row in rows:
        assert row["item_codes"]
        assert set(row["item_codes"]) <= {"4.01", "4.02", "5.02"}


def test_events_carry_the_acceptance_instant_not_midnight(
    ingested: tuple[dict[str, int], Path],
):
    """A date-only timestamp would place every event at midnight and mis-sequence them."""
    _, root = ingested
    rows = read_events(root, "insider", SETTLED_DAY)

    assert all(row["as_of"].tzinfo is not None for row in rows)
    assert any(row["as_of"].hour != 0 for row in rows)


def test_a_real_day_parses_without_systemic_rejections(
    ingested: tuple[dict[str, int], Path],
):
    """A rejection rate above a few percent means a format break, not odd filers.

    Issuers with no listed common stock are excluded from both sides of this
    ratio. They are a property of the filing population rather than a parsing
    problem, and on a real day they ran to 4.3% of Form 4 filings on their own.
    """
    counts, root = ingested

    attempted = counts["insider"] + counts["redflag"] + counts["rejected"]
    assert counts["rejected"] / attempted < 0.05

    quarantine = root / "rejected" / "insider" / f"{SETTLED_DAY.isoformat()}.json"
    assert quarantine.exists(), "every run records its rejections, including none"

    unpriceable = root / "unpriceable" / "insider" / f"{SETTLED_DAY.isoformat()}.json"
    assert unpriceable.exists(), "every run records the filers it could not price"


def test_rerunning_the_day_is_idempotent(ingested: tuple[dict[str, int], Path], tmp_path: Path):
    """A repeated run must produce the same dataset rather than duplicate events."""
    counts, root = ingested
    before = len(read_events(root, "insider", SETTLED_DAY))

    repeated = ingest_day(SETTLED_DAY, root, MIN_VALUE)

    assert repeated == counts
    assert len(read_events(root, "insider", SETTLED_DAY)) == before
