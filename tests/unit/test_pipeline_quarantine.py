"""A filing that cannot be parsed must neither abort the day nor vanish."""

import json
from datetime import date
from pathlib import Path

from arbiter.events.insider import UnpriceableIssuerError
from arbiter.ingestion.pipeline import (
    _PARSE_FAILURES,
    _quarantine,
)

DAY = date(2026, 9, 11)


def test_a_clean_day_still_records_an_empty_rejection_file(tmp_path: Path):
    """An absent file and a day with no rejections must not look the same."""
    _quarantine([], tmp_path, "insider", DAY, attempted=900)

    path = tmp_path / "rejected" / "insider" / "2026-09-11.json"
    assert json.loads(path.read_text()) == []


def test_a_few_bad_filings_are_recorded_with_their_accession_numbers(tmp_path: Path):
    rejected = [{"accession_no": "0001-26-000001", "error": "Ticker is blank"}]

    _quarantine(rejected, tmp_path, "insider", DAY, attempted=900)

    stored = json.loads((tmp_path / "rejected" / "insider" / "2026-09-11.json").read_text())
    assert stored[0]["accession_no"] == "0001-26-000001"
    assert "Ticker" in stored[0]["error"]


def test_a_day_that_mostly_failed_is_reported_as_broken(tmp_path: Path):
    """Ten percent failing is a format or client break, not a few odd filers.

    Reported rather than raised: every domain is judged before any of them stops
    the day, so a day broken in both is not described as broken in one.
    """
    rejected = [{"accession_no": f"a-{i}", "error": "unparseable"} for i in range(10)]

    breakage = _quarantine(rejected, tmp_path, "insider", DAY, attempted=100)

    assert breakage is not None
    assert "10 of 100 insider filings" in breakage


def test_a_healthy_day_reports_no_breakage(tmp_path: Path):
    """The negative control: a clean day must return nothing to report."""
    rejected = [{"accession_no": "a-1", "error": "unparseable"}]

    assert _quarantine(rejected, tmp_path, "insider", DAY, attempted=100) is None


def test_the_report_names_the_quarantine_file(tmp_path: Path):
    """The operator needs to know where to look, not just that something broke."""
    rejected = [{"accession_no": f"a-{i}", "error": "unparseable"} for i in range(10)]

    breakage = _quarantine(rejected, tmp_path, "insider", DAY, attempted=100)

    assert breakage is not None
    assert "rejected" in breakage


def test_a_day_with_no_filings_at_all_does_not_divide_by_zero(tmp_path: Path):
    _quarantine([], tmp_path, "redflag", DAY, attempted=0)

    assert (tmp_path / "rejected" / "redflag" / "2026-09-11.json").exists()


def test_an_unlistable_issuer_cannot_be_quarantined_as_a_parse_failure():
    """The guard exists to catch format breakage, not ordinary filers.

    A real day carried 34 Form 4 filings whose issuer has no listed common
    stock — 4.3% of that day, against a 5% threshold. Were these counted as
    parse failures, a day with slightly more of them would abort ingestion
    while nothing whatsoever was wrong.
    """
    assert not issubclass(UnpriceableIssuerError, _PARSE_FAILURES)


def test_excluding_unlistable_issuers_does_not_hide_a_real_break(tmp_path: Path):
    """They leave the denominator as well as the numerator.

    Ten unparseable filings out of a hundred attempted is a break whether or
    not the day also carried unlistable issuers, so removing those from what
    was attempted must not dilute the share below the threshold.
    """
    rejected = [{"accession_no": f"a-{i}", "error": "unparseable"} for i in range(10)]

    assert _quarantine(rejected, tmp_path, "insider", DAY, attempted=130 - 30) is not None
