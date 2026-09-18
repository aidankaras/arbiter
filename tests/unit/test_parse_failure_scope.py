"""What the per-filing quarantine catches, and what it deliberately does not.

One unparseable filing must not cost a day's other events. The opposite risk is
worse: a quarantine that swallowed every exception would turn a systemic break
into a quiet day of zero events, which no later check could distinguish from a
day on which nothing happened.
"""

import pytest

from arbiter.ingestion.pipeline import _PARSE_FAILURES


@pytest.mark.parametrize(
    ("failure", "why"),
    [
        (ValueError, "a malformed value in a filing"),
        (KeyError, "a column the upstream format no longer provides"),
        (AttributeError, "the client raising from inside its own rendering path"),
    ],
)
def test_single_filing_failures_are_quarantined(failure: type[Exception], why: str):
    assert issubclass(failure, _PARSE_FAILURES), why


@pytest.mark.parametrize(
    ("failure", "why"),
    [
        (MemoryError, "the process is out of resources, not the filing malformed"),
        (KeyboardInterrupt, "the operator asked the run to stop"),
        (SystemExit, "the process is shutting down"),
        (OSError, "the disk or network is failing, which affects every filing"),
    ],
)
def test_systemic_failures_are_not_quarantined(failure: type[BaseException], why: str):
    """These must abort the run rather than be recorded as rejected filings."""
    assert not issubclass(failure, _PARSE_FAILURES), why


def test_the_quarantine_set_is_explicit_rather_than_broad():
    """`except Exception` would hide the systemic failures above."""
    assert Exception not in _PARSE_FAILURES
    assert BaseException not in _PARSE_FAILURES
