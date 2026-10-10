"""Reading a filing's own ticker and owners from live EDGAR.

The unit tests read recorded cover-page rows; this checks the recording still
matches what the SEC serves through the client, which is where an upstream
change to the XBRL facts API would show first.
"""

import pytest
from edgar import find

from arbiter.events.redflags import choose_filing_ticker
from arbiter.ingestion.edgar import configure_identity, cover_securities

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def _identity() -> None:
    configure_identity()


@pytest.mark.parametrize(
    ("accession", "expected"),
    [
        ("0001628280-26-039277", "COLD"),
        # An issuer since delisted: the current company map no longer lists it,
        # the filing's cover page still does (issue #43).
        ("0001193125-26-253436", "FBRX"),
        # Common stock plus six listed depositary preferreds.
        ("0001423689-26-000124", "AGNC"),
    ],
)
def test_the_filings_own_symbol_is_read(accession: str, expected: str):
    assert choose_filing_ticker(cover_securities(find(accession))) == expected


def test_a_multi_symbol_cover_page_lists_every_security():
    securities = cover_securities(find("0001423689-26-000124"))

    assert len(securities) == 7
    assert all(security.title for security in securities)


def test_a_form4_exposes_its_reporting_owners():
    owners = find("0001104659-26-079686").obj().reporting_owners

    assert [(owner.cik, owner.name) for owner in getattr(owners, "owners", owners)] == [
        ("0002068530", "Coastlands Capital LP")
    ]
