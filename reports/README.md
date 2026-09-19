# Results

Measurements committed as they are produced, so that what was claimed at a given
commit stays inspectable. A result recomputed against more data is a new report
rather than an edit to the old one.

## What is here now

`baseline-insider.md` — the conventional arm measured out of sample, written by
`arbiter report`. It states the information coefficient with its standard error,
the sample the estimate rests on, the calibration of its stated probabilities,
and the fitted weights.

Read the verdict at the top before the tables. Over a few dozen days the
standard error on an information coefficient is large enough that no result
should be read as settled, and the report says so in words rather than leaving a
table to be interpreted generously.

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
