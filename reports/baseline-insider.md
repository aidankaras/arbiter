# Baseline arm: insider transactions

Logistic regression on filing characteristics, forecasting the sign of the
five-session abnormal return. This is the control the language-model arms
are measured against, not a proposed strategy.

Generated 2026-10-01 from the event store.

## Result

**A positive information coefficient of +0.1209** (standard error 0.0391, t = +3.09) across 11 days. This clears the conventional two-standard-error threshold, which is a weak bar: it is one test on one sample, and the estimate should be expected to shrink as more days are added. Calibration is worse than that: a Brier skill of -0.0003 means the stated probabilities were less useful than forecasting the base rate of 42.9% for every event. The forecasts run high by 0.019 across the reliability bands below, weighted by the events in each, on balance, with 1 of 3 bands running the other way.

## What this rests on

| | |
|---|---|
| Fitted on | 17 days, 2,988 events |
| Scored on | 11 days, 1,318 events |
| Days contributing a rank correlation | 11 of 11 |
| Scored observations | 592 issuer-days |
| Fitting period | 2026-03-02 to 2026-06-02 |
| Scoring period | 2026-06-12 to 2026-08-11 |
| Base rate | 42.9% of events had a positive abnormal return |
| Brier skill | -0.0003 against forecasting the base rate |

The split is chronological: every scored day falls after every fitted day.
Withheld from both sides: 2026-06-08. Fitted outcomes were still open on
those days, so forecasts made then would share market moves with returns
the model had already been fitted on.
Discrimination is credited once per issuer per day, because several insiders
at one company on one day resolve to a single outcome.

## Information coefficient by day

| Day | Rank correlation |
|---|---|
| 2026-06-12 | +0.1088 |
| 2026-06-18 | -0.1167 |
| 2026-06-25 | +0.0769 |
| 2026-07-01 | +0.1125 |
| 2026-07-08 | +0.0876 |
| 2026-07-14 | +0.3956 |
| 2026-07-20 | +0.1500 |
| 2026-07-24 | +0.2496 |
| 2026-07-30 | +0.0164 |
| 2026-08-05 | +0.1707 |
| 2026-08-11 | +0.0779 |
| **Mean** | **+0.1209** |
| Standard error | 0.0391 |
| t | +3.09 |

## Calibration

Whether a stated probability means what it says. A well-calibrated forecast
has the realised column tracking the forecast column down the table.

| Forecast band | Mean forecast | Realised | Events | |
|---|---|---|---|---|
| 0.3 to 0.4 | 0.371 | 0.377 | 61 | `█████████···············` |
| 0.4 to 0.5 | 0.442 | 0.422 | 445 | `██████████··············` |
| 0.5 to 0.6 | 0.535 | 0.500 | 86 | `████████████············` |

## Fitted weights

On the standardised scale, so magnitudes are comparable across features.
A weight is what the model leaned on, not evidence that the characteristic
predicts anything on its own.

| Feature | Weight |
|---|---|
| `is_purchase` | +0.1697 |
| `fraction_of_holding` | -0.1218 |
| `is_officer` | +0.0934 |
| `is_director` | +0.0786 |
| `is_ten_percent_owner` | +0.0508 |
| `reports_holding` | +0.0446 |
| `log_value_usd` | +0.0371 |
| `insiders_trading_same_issuer` | +0.0240 |

## Reading this honestly

- An information coefficient measures ranking, not profit. Nothing here
  accounts for transaction costs, market impact, or borrow availability.
- The scored sample is one contiguous period. A result from one regime is
  not evidence about another.
- Insider-trading returns documented in the literature have decayed
  substantially since publication, so a weak or absent signal here is the
  expected finding rather than a surprising one.
