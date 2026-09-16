from arbiter.ingestion.pipeline import summarize_counts


def test_counts_report_each_domain():
    assert summarize_counts(insider=3, redflag=2) == {"insider": 3, "redflag": 2}


def test_zero_counts_are_reported_not_omitted():
    """A broken extractor must not be indistinguishable from a quiet day."""
    assert summarize_counts(insider=0, redflag=0) == {"insider": 0, "redflag": 0}


def test_every_configured_domain_appears_in_the_summary():
    """The summary is the run's record; a missing domain would hide a failure."""
    assert set(summarize_counts(insider=1, redflag=1)) == {"insider", "redflag"}
