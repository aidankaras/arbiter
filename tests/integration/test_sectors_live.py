import pytest

from arbiter.ingestion.sectors import benchmark_for_issuer, issuer_sic

pytestmark = pytest.mark.integration

# Issuers whose sector classification is unambiguous and stable.
ALTRIA_CIK = 764180
APPLE_CIK = 320193
AGNC_CIK = 1423689


def test_sic_codes_are_retrieved_for_real_issuers():
    assert issuer_sic(ALTRIA_CIK) == "2111"
    assert issuer_sic(APPLE_CIK) == "3571"


def test_issuers_resolve_to_their_sector_benchmark():
    assert benchmark_for_issuer(ALTRIA_CIK) == "XLP"
    assert benchmark_for_issuer(APPLE_CIK) == "XLK"
    assert benchmark_for_issuer(AGNC_CIK) == "XLRE"


def test_repeated_lookups_are_served_from_cache():
    """A day's ingestion touches hundreds of issuers; each must be fetched once."""
    issuer_sic(ALTRIA_CIK)
    cached = issuer_sic.cache_info()

    issuer_sic(ALTRIA_CIK)

    assert issuer_sic.cache_info().hits > cached.hits
