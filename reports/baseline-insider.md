# Baseline arm: insider transactions

Logistic regression on filing characteristics, forecasting the sign of the
five-session abnormal return. This is the control the language-model arms
are measured against, not a proposed strategy.

Generated 2026-10-10 from the event store.

## Result

**No measurable skill.** The mean information coefficient is +0.0116 against a standard error of 0.0204 (t = +0.57), which is indistinguishable from zero at the conventional threshold. With 29 scored days this is the expected outcome whether or not an edge exists; it is a statement about the sample, not evidence that no signal is there. The stated probabilities are worth no more than forecasting the fitting period's rate of 47.6% for every issuer-day: a Brier skill of -0.0011 is indistinguishable from it. Against the scored period's own rate of 46.1%, which was not knowable in advance, the Brier skill is -0.0020. The forecasts run high by 0.016 across the reliability bands below, weighted by the issuer-days in each, on balance, with 1 of 3 bands running the other way.

## What this rests on

| | |
|---|---|
| Fitted on | 51 days, 7,887 events |
| Scored on | 29 days, 5,779 labelled events; 82 more could not be priced and are recorded as exclusions |
| Days contributing a rank correlation | 29 of 29 |
| Scored observations | 2,655 issuer-days |
| Fitting period | 2026-03-02 to 2026-07-21 |
| Scoring period | 2026-07-30 to 2026-09-09 |
| Up-rate, fitting period | 47.6% of issuer-days had a positive abnormal return |
| Up-rate, scoring period | 46.1% of issuer-days |
| Brier skill | -0.0011 against forecasting the fitting period's rate |
| Brier skill, hindsight reference | -0.0020 against forecasting the scoring period's own rate, not knowable in advance |

The split is chronological: every scored day falls after every fitted day.
Withheld from both sides: 2026-07-22, 2026-07-23, 2026-07-24, 2026-07-27, 2026-07-28, 2026-07-29. Fitted outcomes were still open on
those days, so forecasts made then would share market moves with returns
the model had already been fitted on.
Discrimination is credited once per issuer per day, because several insiders
at one company on one day resolve to a single outcome.

## Information coefficient by day

| Day | Rank correlation |
|---|---|
| 2026-07-30 | -0.1623 |
| 2026-07-31 | -0.1008 |
| 2026-08-03 | -0.0055 |
| 2026-08-04 | +0.0605 |
| 2026-08-05 | +0.1192 |
| 2026-08-06 | -0.0524 |
| 2026-08-07 | -0.0876 |
| 2026-08-10 | +0.1052 |
| 2026-08-11 | +0.0782 |
| 2026-08-12 | -0.0417 |
| 2026-08-13 | +0.0334 |
| 2026-08-14 | -0.0262 |
| 2026-08-17 | -0.0780 |
| 2026-08-18 | +0.0909 |
| 2026-08-19 | +0.2810 |
| 2026-08-20 | +0.0624 |
| 2026-08-21 | -0.0493 |
| 2026-08-24 | +0.1734 |
| 2026-08-25 | +0.1058 |
| 2026-08-26 | +0.2213 |
| 2026-08-27 | +0.1064 |
| 2026-08-28 | -0.1303 |
| 2026-08-31 | -0.0170 |
| 2026-09-01 | -0.1063 |
| 2026-09-02 | +0.0223 |
| 2026-09-03 | -0.1111 |
| 2026-09-04 | +0.0047 |
| 2026-09-08 | -0.1106 |
| 2026-09-09 | -0.0497 |
| **Mean** | **+0.0116** |
| Standard error | 0.0204 |
| t | +0.57 |

## Calibration

Whether a stated probability means what it says. A well-calibrated forecast
has the realised column tracking the forecast column down the table.
Pooled over all 2,655 scored issuer-days, including
any on days with too few issuers to rank.

| Forecast band | Mean forecast | Realised | Issuer-days | |
|---|---|---|---|---|
| 0.3 to 0.4 | 0.391 | 0.406 | 69 | `██████████··············` |
| 0.4 to 0.5 | 0.474 | 0.456 | 2,219 | `███████████·············` |
| 0.5 to 0.6 | 0.509 | 0.501 | 367 | `████████████············` |

## Fitted weights

On the standardised scale, so magnitudes are comparable across features.
A weight is what the model leaned on, not evidence that the characteristic
predicts anything on its own.

| Feature | Weight |
|---|---|
| `fraction_of_holding` | -0.1254 |
| `reports_holding` | +0.0552 |
| `is_purchase` | +0.0365 |
| `is_ten_percent_owner` | -0.0361 |
| `log_value_usd` | +0.0336 |
| `insiders_trading_same_issuer` | +0.0115 |
| `is_director` | +0.0094 |
| `is_officer` | -0.0058 |

## Reading this honestly

- An information coefficient measures ranking, not profit. Nothing here
  accounts for transaction costs, market impact, or borrow availability.
- The scored sample is one contiguous period. A result from one regime is
  not evidence about another.
- Insider-trading returns documented in the literature have decayed
  substantially since publication, so a weak or absent signal here is the
  expected finding rather than a surprising one.
