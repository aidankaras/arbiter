# Baseline arm: insider transactions

Logistic regression on filing characteristics, forecasting the sign of the
five-session abnormal return. This is the control the language-model arms
are measured against, not a proposed strategy.

Generated 2026-09-19 from the event store.

## Result

**No measurable skill.** The mean information coefficient is +0.0308 against a standard error of 0.0581 (t = +0.53), which is indistinguishable from zero at the conventional threshold. With 11 scored days this is the expected outcome whether or not an edge exists; it is a statement about the sample, not evidence that no signal is there. Calibration is worse than that: a Brier skill of -0.0367 means the stated probabilities were less useful than forecasting the base rate of 46.2% for every event. The forecasts run low by 0.011 across the reliability bands below, weighted by the events in each, on balance, with 2 of 6 bands running the other way.

## What this rests on

| | |
|---|---|
| Fitted on | 15 days, 7,296 events |
| Scored on | 11 days, 4,712 events |
| Days contributing a rank correlation | 11 of 11 |
| Scored observations | 1,732 issuer-days |
| Fitting period | 2026-03-02 to 2026-05-27 |
| Scoring period | 2026-06-08 to 2026-08-11 |
| Base rate | 46.2% of events had a positive abnormal return |
| Brier skill | -0.0367 against forecasting the base rate |

The split is chronological: every scored day falls after every fitted day.
Discrimination is credited once per issuer per day, because several insiders
at one company on one day resolve to a single outcome.

## Information coefficient by day

| Day | Rank correlation |
|---|---|
| 2026-06-08 | -0.0669 |
| 2026-06-12 | -0.0291 |
| 2026-06-18 | -0.3556 |
| 2026-07-01 | +0.2678 |
| 2026-07-08 | -0.1171 |
| 2026-07-14 | +0.3405 |
| 2026-07-20 | +0.0236 |
| 2026-07-24 | -0.0211 |
| 2026-07-30 | +0.0475 |
| 2026-08-05 | +0.2108 |
| 2026-08-11 | +0.0385 |
| **Mean** | **+0.0308** |
| Standard error | 0.0581 |
| t | +0.53 |

## Calibration

Whether a stated probability means what it says. A well-calibrated forecast
has the realised column tracking the forecast column down the table.

| Forecast band | Mean forecast | Realised | Events | |
|---|---|---|---|---|
| 0.2 to 0.3 | 0.273 | 0.831 | 65 | `████████████████████····` |
| 0.3 to 0.4 | 0.363 | 0.425 | 294 | `██████████··············` |
| 0.4 to 0.5 | 0.457 | 0.452 | 1,003 | `███████████·············` |
| 0.5 to 0.6 | 0.529 | 0.454 | 355 | `███████████·············` |
| 0.6 to 0.7 | 0.633 | 0.500 | 14 | `████████████············` |
| 0.7 to 0.8 | 0.725 | 0.000 | 1 | `························` |

## Fitted weights

On the standardised scale, so magnitudes are comparable across features.
A weight is what the model leaned on, not evidence that the characteristic
predicts anything on its own.

| Feature | Weight |
|---|---|
| `fraction_of_holding` | -0.1962 |
| `log_value_usd` | +0.1670 |
| `is_officer` | +0.1507 |
| `is_ten_percent_owner` | -0.1306 |
| `is_purchase` | +0.1289 |
| `insiders_trading_same_issuer` | +0.1058 |
| `is_director` | +0.0704 |
| `reports_holding` | +0.0562 |

## Reading this honestly

- An information coefficient measures ranking, not profit. Nothing here
  accounts for transaction costs, market impact, or borrow availability.
- The scored sample is one contiguous period. A result from one regime is
  not evidence about another.
- Insider-trading returns documented in the literature have decayed
  substantially since publication, so a weak or absent signal here is the
  expected finding rather than a surprising one.
