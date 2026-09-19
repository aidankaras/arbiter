# The red-flag domain is not yet measurable

A survey of the red-flag population before building a feature set for it, and
the reason no feature set was built.

Generated 2026-09-18 from 1,312 events across 26 sampled trading days,
2026-03-02 to 2026-08-11.

## What the population actually contains

| Trigger | Events | Share |
|---|---|---|
| Item 5.02 — departure or appointment of an officer or director | 1,237 | 94.3% |
| Item 4.01 — change of certifying accountant | 73 | 5.6% |
| Item 4.02 — non-reliance on previously issued financial statements | 13 | 1.0% |

Other properties are close to constant: 99.2% of filings trigger exactly one
item, and 4.8% are accompanied by a press release.

## Why this stops a baseline arm

The three characteristics an 8-K makes available — which item fired, whether a
press release accompanied it, how much the filing says — have almost no
variation across this population. An indicator for Item 5.02 is true for 94 of
every 100 events; one for a press release is false for 95 of every 100. A
feature that rarely varies contributes a coefficient fitted on a handful of
observations, and a feature that never varies contributes a coefficient that is
noise. Publishing weights for either would state a finding where none exists.

This is the same defect that removed two features from the insider arm, found
the same way: by looking at the distribution of the actual stored data rather
than at what the filing format permits.

## Why the interesting items are the rare ones

The literature associates large abnormal returns with restatements and auditor
changes, not with routine officer transitions. Palmrose, Richardson and Scholz
report roughly −9% around restatement announcements; departures are frequent,
heterogeneous, and mostly uninformative.

The population inverts that. The events worth measuring are 1% and 5.6% of what
the screen admits, and the events that dominate it are the ones with the least
reason to move a price.

## What this would take

At 13 restatement events across 26 sampled days — roughly one every other
trading day — a sample large enough to estimate an effect at this horizon needs
on the order of several hundred such events. That is years of calendar coverage
at the current sampling rate, and still over a thousand trading days ingested
even sampling every day.

Three options, in the order they should be considered:

1. **Study Items 4.01 and 4.02 as their own population**, ingesting a long
   historical range for those items alone. The events are rare but cheap to
   isolate, and this is the version of the domain the literature actually
   supports.
2. **Treat Item 5.02 as a separate question** with its own hypothesis, since a
   population of officer transitions is not a red-flag study and should not be
   reported as one.
3. **Leave the domain unmeasured** and say so, which is the current state.

## What was verified

Ingestion, labelling and exclusion recording work correctly for this domain —
1,312 events across 26 days resolved to labels through the same path as the
insider arm, with the same point-in-time discipline and the same twenty-session
horizon. The gap is not in the pipeline. It is that the screened population
cannot support the measurement the domain was defined to make.
