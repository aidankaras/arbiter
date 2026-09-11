# Earnings events

## Trigger

8-K filings carrying Item 2.02 (Results of Operations and Financial Condition),
joined to the issuer's XBRL company facts for the reported period.

Roughly 16,000 qualifying events per year.

## Surprise definition

Surprise is computed as standardized unexpected earnings against a seasonal
random walk:

```
SUE = (EPS_q − EPS_{q−4}) / stdev(EPS_q − EPS_{q−4} over the trailing 8 quarters)
```

This deliberately avoids analyst consensus data, which is not freely available at
the coverage this system needs. The seasonal random walk is the standard
construction in the post-earnings-announcement drift literature, so results
remain comparable to published work, and it removes a paid dependency that would
otherwise gate the entire strategy.

The tradeoff is real and worth stating: a consensus-based surprise measures the
deviation from market expectation, while this measures deviation from the firm's
own recent trajectory. They diverge most for firms undergoing rapid change.

## Features

| Feature | Source |
|---|---|
| Standardized unexpected earnings | XBRL facts |
| Revenue surprise, same construction | XBRL facts |
| Pre-announcement drift over the prior 20 sessions | Prices |
| Realized volatility, trailing 20 and 60 sessions | Prices |
| Market capitalization decile at the event timestamp | Prices, shares outstanding |
| Sector | SIC classification |
| Days since the prior earnings event | Event history |
| Guidance language flags | Filing text |

## Restatement hazard

XBRL company facts reflect restatements, so querying a historical period today
can return a figure revised after the fact. Facts are therefore filtered to those
whose own filing date precedes the event timestamp. See
[`../lessons/01-point-in-time-correctness.md`](../lessons/01-point-in-time-correctness.md).

## Expected result

Dense signal, mostly numeric. The baseline and neural arms should perform
comparably here, and the agent arm's advantage, if any, should be small. This is
the control against which the red-flag strategy's text dependence is measured.
