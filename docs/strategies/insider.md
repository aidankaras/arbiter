# Insider transactions

## Trigger

Form 4 statements of changes in beneficial ownership. 351,055 were filed in 2025,
making this the highest-volume strategy by two orders of magnitude.

Form 4 is submitted as structured XML, so extraction requires no inference. This
strategy is almost entirely a parsing and feature-engineering problem.

## Features

The baseline arm uses these features, defined in `src/arbiter/arms/features.py`:

| Feature | Definition |
|---|---|
| `is_purchase` | Open-market purchase (code P) rather than sale (code S) |
| `log_value_usd` | Log of the transaction's dollar value |
| `fraction_of_holding` | Shares traded as a fraction of the stake held before the trade, reconstructed from the post-trade holding Form 4 reports (added back for a sale, subtracted for a purchase) |
| `reports_holding` | Whether the filing reported a remaining holding, so a missing value is distinguishable from zero |
| `is_officer`, `is_director`, `is_ten_percent_owner` | Filer role, as independent flags because one insider often holds several |
| `insiders_trading_same_issuer` | Distinct insiders who had filed for the same issuer that day by this filing's acceptance, itself included |

Rule 10b5-1 status is a screening criterion rather than a feature: scheduled
trades are excluded before scoring, so the column would be constant. Size
relative to the filer's own history and timing relative to the next earnings
date are planned; neither is computed yet.

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
