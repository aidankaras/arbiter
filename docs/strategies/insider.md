# Insider transactions

## Trigger

Form 4 statements of changes in beneficial ownership. 351,055 were filed in 2025,
making this the highest-volume strategy by two orders of magnitude.

Form 4 is submitted as structured XML, so extraction requires no inference. This
strategy is almost entirely a parsing and feature-engineering problem.

## Features

| Feature | Notes |
|---|---|
| Filer role | Officer, director, or ten-percent owner |
| Transaction code | Open-market purchase and sale are informative; option exercises and gifts largely are not |
| Size relative to existing holdings | A purchase doubling a position differs from one adding a percent |
| Size relative to the filer's own history | Normalizes across individuals |
| Cluster flag | Multiple insiders transacting within a rolling window |
| Days to the next scheduled earnings event | Timing relative to known information releases |
| Rule 10b5-1 plan indicator | Pre-scheduled trades carry little information |

**The plan indicator is read at filing level, not per transaction.** The parsed
filing exposes one flag, so a Form 4 reporting both a scheduled sale and a
discretionary purchase marks both as scheduled, and the candidate filter drops
both. That is a deliberate approximation in the conservative direction: it
discards informative trades rather than admitting uninformative ones. It is not
neutral, because filers who mix both kinds in a single filing are not a random
subset, so the excluded population carries a bias worth measuring before any
claim rests on it. Reading the per-transaction footnote would remove the
approximation.

The last is important and frequently overlooked. Transactions executed under a
pre-established trading plan were scheduled months earlier and say nothing about
current information. Treating them as signal is a well-documented way to
manufacture apparent alpha that does not survive correction.

## Volume and cost

At roughly 1,400 filings per trading day, this strategy alone would exhaust the
entire inference budget several times over if each filing reached a model.

It does not. Parsing, classification, and feature construction are deterministic.
Only events surviving the screening rule reach an LLM arm, which is a small
fraction of a percent of daily volume.

## Expected result

Structured signal, almost no prose. The baseline arm should be competitive with
everything else, and the agent arm should show no meaningful advantage. A finding
that it does would be more interesting than a finding that it does not, and would
warrant investigating whether the advantage is real or an artifact of screening.
