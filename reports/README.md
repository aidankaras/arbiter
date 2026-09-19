# Results

Measurements committed as they are produced, so that what was claimed at a given
commit stays inspectable. A result recomputed against more data is a new report
rather than an edit to the old one.

## What is here now

`baseline-insider.md` — **withdrawn 2026-09-19; no figure in it should be
cited.** The event store it was computed from held duplicated rows, from two
separate causes. A filing's single qualifying transaction had been stored once
per line on its form, inflating events by a factor averaging 2.94 across days
and ranging from 2.34 to 4.59. Separately, one transaction filed by several
joint reporting owners was counted once per owner — an over-weighting that
tracked filer type, since funds and large holders file jointly while officers
file alone.

Both causes are fixed and pinned by contract tests over recorded filings. The
report is kept rather than deleted so that what was claimed at its commit stays
inspectable, which is the same reason results are committed at all. It will be
replaced by a new report computed from re-ingested data rather than edited in
place.

`redflag-population.md` — why the red-flag domain has no baseline arm yet. The
screened population is 94% officer transitions, while the restatements and
auditor changes the literature associates with large abnormal returns are 1%
and 5.6% of it. The pipeline works for this domain; the population cannot
support the measurement.

## What is planned

A weekly research log covering events processed, per-arm performance, where the
arms disagreed and which was right, representative decisions with their
reasoning, merged changes with their benchmark deltas, and cost against budget.
That requires the three language-model arms and the shadow portfolios, none of
which exist yet.

## How to read any figure here

Every number is generated from the stored event and label data by a command in
this repository, never transcribed by hand. Results that did not work are
reported alongside those that did — a null result measured correctly is the
expected outcome for most of what this project tests, and reporting only the
favourable ones would make the whole exercise unreadable.
