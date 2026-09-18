"""Resolving an issuer's ticker from its CIK, against live EDGAR.

An 8-K identifies its filer by CIK alone, so without this lookup the red-flag
domain cannot be priced and every event in it would be recorded unresolvable for
a reason unrelated to the market.
"""

import pytest

from arbiter.ingestion.sectors import benchmark_for_issuer, issuer_ticker

pytestmark = pytest.mark.integration

ALTRIA_CIK = 764180
APPLE_CIK = 320193
AGNC_CIK = 1423689  # the red-flag fixture issuer


def test_tickers_are_resolved_for_real_issuers():
    assert issuer_ticker(ALTRIA_CIK) == "MO"
    assert issuer_ticker(APPLE_CIK) == "AAPL"


def test_the_red_flag_fixture_issuer_can_be_priced():
    """The domain was unmeasurable until this lookup existed."""
    assert issuer_ticker(AGNC_CIK) == "AGNC"
    assert benchmark_for_issuer(AGNC_CIK) == "XLRE"


def test_repeated_lookups_are_served_from_cache():
    """A day of red flags touches many issuers; each must be fetched once."""
    issuer_ticker(ALTRIA_CIK)
    before = issuer_ticker.cache_info()

    issuer_ticker(ALTRIA_CIK)

    assert issuer_ticker.cache_info().hits > before.hits


def test_an_issuer_and_its_benchmark_are_different_symbols():
    """A self-benchmarked event would measure a return of exactly zero, forever."""
    assert issuer_ticker(ALTRIA_CIK) != benchmark_for_issuer(ALTRIA_CIK)
