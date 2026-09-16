import pytest

from arbiter.ingestion.sectors import (
    BROAD_MARKET_ETF,
    SECTOR_ETFS,
    sector_etf_for_sic,
)


@pytest.mark.parametrize(
    ("sic", "expected", "why"),
    [
        ("2111", "XLP", "cigarettes are consumer staples"),
        ("3571", "XLK", "electronic computers are technology"),
        ("7372", "XLK", "prepackaged software is technology"),
        ("6798", "XLRE", "real estate investment trusts"),
        ("6022", "XLF", "state commercial banks are financials"),
        ("1311", "XLE", "crude petroleum and natural gas is energy"),
        ("4911", "XLU", "electric services are utilities"),
        ("3711", "XLY", "motor vehicles are consumer discretionary"),
        ("3721", "XLI", "aircraft manufacturing is industrials"),
        ("2820", "XLB", "plastics and resins are materials"),
        ("4813", "XLC", "telephone communications"),
        ("3826", "XLV", "laboratory analytical instruments are health care"),
    ],
)
def test_representative_codes_map_to_their_sector(sic: str, expected: str, why: str):
    assert sector_etf_for_sic(sic) == expected, why


def test_an_unmapped_code_falls_back_to_the_broad_market():
    """A deliberate, recorded fallback: the label still has a benchmark."""
    assert sector_etf_for_sic("9995") == BROAD_MARKET_ETF


def test_a_missing_code_falls_back_to_the_broad_market():
    assert sector_etf_for_sic(None) == BROAD_MARKET_ETF


def test_codes_are_accepted_with_or_without_leading_zeros():
    """EDGAR reports some codes three digits wide; both forms must agree."""
    assert sector_etf_for_sic("0100") == sector_etf_for_sic("100")


def test_every_mapped_symbol_is_a_known_benchmark():
    """Guards against a typo silently creating a benchmark that does not trade."""
    assert BROAD_MARKET_ETF in SECTOR_ETFS or BROAD_MARKET_ETF == "SPY"
    for symbol in SECTOR_ETFS:
        assert symbol.isupper()
        assert 2 <= len(symbol) <= 4


def test_the_mapping_covers_the_eleven_sectors():
    """Fewer would mean a sector silently collapses into the broad market."""
    assert len(SECTOR_ETFS) == 11
