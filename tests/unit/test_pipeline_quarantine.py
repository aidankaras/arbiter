"""A filing that cannot be parsed must neither abort the day nor vanish."""

import json
from datetime import date
from pathlib import Path

import pytest

from arbiter.ingestion.pipeline import SystemicParseFailureError, _quarantine

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


def test_a_day_that_mostly_failed_is_refused(tmp_path: Path):
    """Ten percent failing is a format or client break; the day is not recorded."""
    rejected = [{"accession_no": f"a-{i}", "error": "unparseable"} for i in range(10)]

    with pytest.raises(SystemicParseFailureError, match="format or client breakage"):
        _quarantine(rejected, tmp_path, "insider", DAY, attempted=100)


def test_the_refusal_names_the_quarantine_file(tmp_path: Path):
    """The operator needs to know where to look, not just that something broke."""
    rejected = [{"accession_no": f"a-{i}", "error": "unparseable"} for i in range(10)]

    with pytest.raises(SystemicParseFailureError, match="rejected"):
        _quarantine(rejected, tmp_path, "insider", DAY, attempted=100)


def test_a_day_with_no_filings_at_all_does_not_divide_by_zero(tmp_path: Path):
    _quarantine([], tmp_path, "redflag", DAY, attempted=0)

    assert (tmp_path / "rejected" / "redflag" / "2026-09-11.json").exists()
