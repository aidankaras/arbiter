# Baseline arm: insider transactions

Logistic regression on filing characteristics, forecasting the sign of the
five-session abnormal return. This is the control the language-model arms
are measured against, not a proposed strategy.

Generated 2026-10-10 from the event store.

## Result

**No measurable skill.** The mean information coefficient is +0.0285 against a standard error of 0.0187, the larger of the error treating days as independent and one allowing for each day's outcome window overlapping the next 5 (t = +1.52), which is indistinguishable from zero at the conventional threshold. With 60 scored days this is the expected outcome whether or not an edge exists; it is a statement about the sample, not evidence that no signal is there. The stated probabilities are worth no more than forecasting the up-rate each refit had observed, 47.3% on average for every issuer-day: a Brier skill of -0.0015 is indistinguishable from it. Against the scored period's own rate of 47.0%, which was not knowable in advance, the Brier skill is -0.0013. The bands below disagree in direction: 2 run high and 1 run low, so the miss is dispersion rather than a shift.

## What this rests on

| | |
|---|---|
| Initial fitting window | 26 days, 2026-03-02 to 2026-06-12; never scored |
| Refits | 6, one per block of up to 10 scored days |
| Scored on | 60 days, 8,715 labelled events; 159 more could not be priced and are recorded as exclusions |
| Days contributing a rank correlation | 60 of 60 |
| Scored observations | 3,909 issuer-days |
| Scoring period | 2026-06-15 to 2026-09-09 |
| First refit | fitted on 20 days, 3,815 events: only the initial-window events whose outcomes had closed |
| Final refit | fitted on 70 days, 10,846 events |
| Up-rate, refits | 47.3% of fitted issuer-days had a positive abnormal return, averaged over scored issuer-days by the refit that scored them |
| Up-rate, scoring period | 47.0% of issuer-days |
| Brier skill | -0.0015 against forecasting the up-rate of the refit that made each forecast |
| Brier skill, hindsight reference | -0.0013 against forecasting the scoring period's own rate, not knowable in advance |

Evaluation is walk-forward: each block of scored days is forecast by a model
fitted only on events whose outcomes had closed before the block's first day.
Discrimination is credited once per issuer per day, because several insiders
at one company on one day resolve to a single outcome.

## Information coefficient by day

| Day | Rank correlation |
|---|---|
| 2026-06-15 | -0.0646 |
| 2026-06-16 | +0.0448 |
| 2026-06-17 | +0.0249 |
| 2026-06-18 | -0.0410 |
| 2026-06-22 | -0.0573 |
| 2026-06-23 | +0.0822 |
| 2026-06-24 | +0.0830 |
| 2026-06-25 | +0.0126 |
| 2026-06-26 | -0.0332 |
| 2026-06-29 | +0.2755 |
| 2026-06-30 | -0.0328 |
| 2026-07-01 | -0.1714 |
| 2026-07-02 | +0.1164 |
| 2026-07-06 | +0.4024 |
| 2026-07-07 | -0.0030 |
| 2026-07-08 | +0.2282 |
| 2026-07-09 | -0.1195 |
| 2026-07-10 | +0.1887 |
| 2026-07-13 | +0.2401 |
| 2026-07-14 | +0.2088 |
| 2026-07-15 | +0.2031 |
| 2026-07-16 | -0.3230 |
| 2026-07-17 | +0.0208 |
| 2026-07-20 | +0.2107 |
| 2026-07-21 | +0.0138 |
| 2026-07-22 | +0.0629 |
| 2026-07-23 | -0.2931 |
| 2026-07-24 | +0.1228 |
| 2026-07-27 | -0.1551 |
| 2026-07-28 | +0.2353 |
| 2026-07-29 | +0.0927 |
| 2026-07-30 | -0.1711 |
| 2026-07-31 | -0.1015 |
| 2026-08-03 | -0.0134 |
| 2026-08-04 | +0.0614 |
| 2026-08-05 | +0.1157 |
| 2026-08-06 | -0.0598 |
| 2026-08-07 | -0.0894 |
| 2026-08-10 | +0.0865 |
| 2026-08-11 | +0.0869 |
| 2026-08-12 | -0.0202 |
| 2026-08-13 | -0.0072 |
| 2026-08-14 | -0.0418 |
| 2026-08-17 | -0.1052 |
| 2026-08-18 | +0.0565 |
| 2026-08-19 | +0.3037 |
| 2026-08-20 | +0.0828 |
| 2026-08-21 | -0.0331 |
| 2026-08-24 | +0.1338 |
| 2026-08-25 | +0.1079 |
| 2026-08-26 | +0.2291 |
| 2026-08-27 | +0.1315 |
| 2026-08-28 | -0.1216 |
| 2026-08-31 | -0.0229 |
| 2026-09-01 | -0.1602 |
| 2026-09-02 | +0.0454 |
| 2026-09-03 | -0.1203 |
| 2026-09-04 | +0.0331 |
| 2026-09-08 | -0.1518 |
| 2026-09-09 | -0.1205 |
| **Mean** | **+0.0285** |
| Standard error | 0.0187, the larger of the error treating days as independent and one allowing for each day's outcome window overlapping the next 5 |
| t | +1.52 |

## Calibration

Whether a stated probability means what it says. A well-calibrated forecast
has the realised column tracking the forecast column down the table.
Pooled over all 3,909 scored issuer-days, including
any on days with too few issuers to rank.

| Forecast band | Mean forecast | Realised | Issuer-days | |
|---|---|---|---|---|
| 0.3 to 0.4 | 0.388 | 0.415 | 106 | `██████████··············` |
| 0.4 to 0.5 | 0.470 | 0.468 | 3,338 | `███████████·············` |
| 0.5 to 0.6 | 0.517 | 0.503 | 465 | `████████████············` |

## Fitted weights

From the final refit, which saw the most data; earlier refits' weights are
not shown. On the standardised scale, so magnitudes are comparable across
features.
A weight is what the model leaned on, not evidence that the characteristic
predicts anything on its own.

| Feature | Weight |
|---|---|
| `fraction_of_holding` | -0.0835 |
| `is_purchase` | +0.0373 |
| `reports_holding` | +0.0285 |
| `is_ten_percent_owner` | -0.0257 |
| `log_value_usd` | +0.0225 |
| `is_officer` | -0.0150 |
| `is_director` | -0.0086 |
| `insiders_trading_same_issuer` | -0.0035 |

## Reading this honestly

- An information coefficient measures ranking, not profit. Nothing here
  accounts for transaction costs, market impact, or borrow availability.
- The scored sample is one contiguous period. A result from one regime is
  not evidence about another.
- Insider-trading returns documented in the literature have decayed
  substantially since publication, so a weak or absent signal here is the
  expected finding rather than a surprising one.
