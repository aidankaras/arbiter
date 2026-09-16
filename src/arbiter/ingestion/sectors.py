"""Benchmark selection.

An abnormal return is a statement about an event only if the benchmark
subtracted from it represents what the issuer would have done anyway. Issuers
are therefore benchmarked against their sector rather than the whole market.

EDGAR classifies filers by SIC, so the mapping runs from SIC major group to the
corresponding sector fund. The table below is a judgment: SIC predates the
sector schemes these funds track, and a handful of major groups sit plausibly in
two sectors. Two consequences follow, and both are deliberate.

First, an issuer whose major group is not mapped is benchmarked against the
broad market rather than dropped. A weaker benchmark still measures something;
no benchmark measures nothing.

Second, every label records the benchmark it used. A later revision of this
table therefore shows up as a visible difference between labels rather than as a
silent change in what past results meant.
"""

from __future__ import annotations

from functools import lru_cache

from edgar import Company

from arbiter.ingestion.edgar import configure_identity

#: The eleven sector funds, one per sector in the scheme these map onto.
SECTOR_ETFS = (
    "XLB",  # materials
    "XLC",  # communication services
    "XLE",  # energy
    "XLF",  # financials
    "XLI",  # industrials
    "XLK",  # technology
    "XLP",  # consumer staples
    "XLRE",  # real estate
    "XLU",  # utilities
    "XLV",  # health care
    "XLY",  # consumer discretionary
)

#: Used when an issuer's major group has no sector mapping.
BROAD_MARKET_ETF = "SPY"

# SIC major group (first two digits) to sector fund. Ranges follow the SEC's own
# division structure, with manufacturing and services split by major group
# because those divisions span several sectors.
_MAJOR_GROUP_TO_ETF: dict[int, str] = {
    # Agriculture, forestry, fishing, mining, construction
    **dict.fromkeys(range(1, 10), "XLP"),  # agricultural production
    **dict.fromkeys(range(10, 15), "XLB"),  # metal and nonmetal mining
    13: "XLE",  # oil and gas extraction
    **dict.fromkeys(range(15, 18), "XLI"),  # construction
    # Manufacturing
    **dict.fromkeys(range(20, 22), "XLP"),  # food and tobacco
    **dict.fromkeys(range(22, 24), "XLY"),  # textiles and apparel
    **dict.fromkeys(range(24, 28), "XLB"),  # lumber, furniture, paper, printing
    28: "XLV",  # chemicals and pharmaceuticals, dominated by drugs
    29: "XLE",  # petroleum refining
    **dict.fromkeys(range(30, 35), "XLB"),  # rubber, stone, primary metals
    35: "XLK",  # industrial and computer equipment
    36: "XLK",  # electronic equipment
    37: "XLI",  # transportation equipment
    38: "XLV",  # instruments, dominated by medical and analytical
    39: "XLY",  # miscellaneous manufacturing
    # Transportation, communications, utilities
    **dict.fromkeys(range(40, 48), "XLI"),  # rail, trucking, air, pipelines
    48: "XLC",  # communications
    49: "XLU",  # electric, gas, sanitary services
    # Trade
    **dict.fromkeys(range(50, 52), "XLI"),  # wholesale distribution
    **dict.fromkeys(range(52, 60), "XLY"),  # retail
    # Finance, insurance, real estate
    **dict.fromkeys(range(60, 65), "XLF"),  # banks, brokers, insurers
    **dict.fromkeys(range(65, 68), "XLRE"),  # real estate
    # Services
    **dict.fromkeys(range(70, 73), "XLY"),  # hotels, personal services
    73: "XLK",  # business services, dominated by software
    **dict.fromkeys(range(74, 79), "XLY"),  # repair, entertainment
    78: "XLC",  # motion pictures
    **dict.fromkeys(range(80, 81), "XLV"),  # health services
    **dict.fromkeys(range(81, 90), "XLY"),  # legal, educational, social
}

# Chemicals split: 28 covers both drugs and industrial chemicals. Drugs dominate
# filing volume, so the major group maps to health care and the industrial
# subgroups are corrected individually.
_SUBGROUP_OVERRIDES: dict[int, str] = {
    2800: "XLB",  # industrial inorganic chemicals
    2810: "XLB",
    2820: "XLB",  # plastics and synthetic resins
    2860: "XLB",  # industrial organic chemicals
    2870: "XLB",  # agricultural chemicals
    2890: "XLB",
    3826: "XLV",  # laboratory analytical instruments
    # Transportation equipment spans two sectors that move differently. Aircraft
    # and railroad stock track industrials; vehicles sold to households track
    # consumer demand, so the automakers are corrected individually rather than
    # by moving the whole major group and misclassifying aerospace.
    3711: "XLY",  # motor vehicles and passenger car bodies
    3713: "XLY",  # truck and bus bodies
    3714: "XLY",  # motor vehicle parts and accessories
    3715: "XLY",  # truck trailers
    3716: "XLY",  # motor homes
    3751: "XLY",  # motorcycles and bicycles
}


def sector_etf_for_sic(sic: str | None) -> str:
    """Return the benchmark symbol for an issuer's SIC code.

    Accepts codes with or without leading zeros. An unknown or absent code
    yields the broad market fund, which is recorded on the label like any other
    benchmark.
    """
    if sic is None:
        return BROAD_MARKET_ETF

    digits = str(sic).strip()
    if not digits.isdigit():
        return BROAD_MARKET_ETF

    code = int(digits)
    if code in _SUBGROUP_OVERRIDES:
        return _SUBGROUP_OVERRIDES[code]

    return _MAJOR_GROUP_TO_ETF.get(code // 100, BROAD_MARKET_ETF)


@lru_cache(maxsize=4096)
def issuer_sic(cik: int) -> str | None:
    """Return an issuer's SIC code, or `None` when EDGAR reports none.

    Identity is configured here rather than assumed. The SEC refuses requests
    without a contact string, and relying on some earlier call to have set it
    would make this function work or fail according to what ran before it.

    Cached because a day's filings concentrate in far fewer issuers than
    filings, and each distinct issuer would otherwise cost its own request.
    """
    configure_identity()
    sic = getattr(Company(cik), "sic", None)
    return str(sic) if sic else None


def benchmark_for_issuer(cik: int) -> str:
    """Return the benchmark symbol an issuer's returns are measured against."""
    return sector_etf_for_sic(issuer_sic(cik))
