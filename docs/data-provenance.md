# Data provenance

Every input this project uses, where it comes from, and the terms it arrives
under. Anyone reproducing a result needs to know what the numbers were derived
from; anyone in finance reading this repository will want to know the licensing
position before anything else.

## SEC EDGAR

**What:** Form 4 statements of changes in beneficial ownership, and 8-K current
reports carrying items 4.01, 4.02, and 5.02.

**Terms:** EDGAR filings are public domain works of the US government. The SEC's
fair-access policy requires every automated request to carry a descriptive
User-Agent identifying the requester, and rate-limits or refuses traffic without
one. That string is configuration, not a constant: `SEC_USER_AGENT` is validated
at startup and the process refuses to run without it, so the requirement cannot
be skipped by forgetting it.

**Accessed through:** `edgartools`, which resolves filings from EDGAR's daily
indices. Acceptance timestamps come from each filing's SEC header.

## Market data and news

**What:** daily bars for issuers and sector benchmark funds, and news articles
used as point-in-time context.

**Source:** Alpaca's market data API on its Basic plan. Bars come from the
consolidated tape, which the plan serves for every window except one ending on
the current session. News is sourced by Alpaca from Benzinga.

**Terms:** market data is licensed to the account holder for their own use.
**No price or news content is redistributed by this repository.** Committed
fixtures contain a small number of records captured for regression testing, not
a dataset.

**Credentials** are supplied through the environment and never committed. The
broker endpoint is validated against an exact host allowlist, so a live-trading
endpoint fails at startup rather than at order submission.

## Test fixtures

`tests/fixtures/` holds real captured payloads rather than synthesized mocks,
because a parser tested only against invented data is tested against the author's
assumptions. They therefore contain real accession numbers, real issuer names,
and the names of real corporate insiders — all of which are already public
record in the filings themselves, published by the SEC.

Fixtures are captured at a fixed past date so their content does not drift.

## Derived data

Event partitions under `data/` and anything computed from them are derived works
of the sources above. They are not committed, and the directory is ignored.

## What is not used

No paid analyst consensus data, no alternative data vendors, no scraped content,
and no data obtained outside the documented APIs. The earnings surprise measure
is deliberately constructed from a firm's own reported history precisely so the
project carries no dependency on licensed consensus estimates.
