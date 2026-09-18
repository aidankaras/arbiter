# Methodology

How Arbiter constructs events, labels them, controls information leakage, and
evaluates four forecasting approaches against each other. This document defines
the properties the rest of the system exists to preserve.

## The comparison being made

Four approaches forecast the same quantity from the same inputs:

| Arm | Information used | Inference |
|---|---|---|
| `baseline` | Structured features only | None |
| `dl` | Structured features and filing text | Fine-tuned encoder |
| `agent` | Structured features and filing text | LLM agent team |
| `arbiter` | The other three arms' outputs and reasoning | LLM, different family |

`baseline` is the null hypothesis. It is deliberately simple, because a complex
baseline can obscure the absence of an effect just as easily as a weak one can
manufacture the appearance of one.

`arbiter` uses a different model family from `agent`. A judge sharing a family
with the system it evaluates shares its blind spots, and agreement between
correlated models is not evidence of correctness.

## Events

An event is a filing that meets a strategy's trigger condition.

| Strategy | Trigger | Approximate annual volume |
|---|---|---|
| Earnings | 8-K Item 2.02, joined to XBRL company facts | 16,000 |
| Insider | Form 4 ownership filings | 350,000 |
| Red flags | 8-K Items 4.01, 4.02, 5.02 | low thousands |

Volumes are drawn from EDGAR's full-year filing indices. The spread across three
orders of magnitude is intentional: it lets the same pipeline test whether an
approach's advantage depends on data density.

Events are timestamped with the filing's **EDGAR acceptance datetime**, not its
period-of-report or its filing date. Acceptance time is when the information
became publicly available, which is the only timestamp that makes a forecast
meaningful.

## The evidence packet

Each event produces exactly one packet, which is immutable and content-hashed.

Contents: the filing's sectioned text, issuer identity and classification, market
context truncated at the event timestamp, strategy-specific extracted fields, and
comparable historical cases.

Three properties are enforced in code rather than by convention:

**Immutability.** A packet is frozen after construction. Every prediction stores
the packet's hash, so a result can be traced to the exact input that produced it,
and any change to packet construction is detectable rather than silent.

**Information truncation.** Market context is computed from data available at the
event timestamp. Price windows, volatility estimates, and volume statistics all
terminate there.

**No downstream fetching.** The three forecasting arms that consume a packet
receive no capability to retrieve anything else. The agent analyst is constructed
with an empty tool list, so the constraint holds structurally rather than
depending on a prompt being obeyed.

Without these, the four arms cannot be shown to have seen the same information,
and the comparison has no meaning.

## Comparable retrieval and the as-of constraint

Arms may retrieve similar historical cases, including how those cases resolved.
This is what allows an approach to reason from precedent.

Every such retrieval filters to cases that had **already resolved strictly before
the event's own timestamp.** The filter lives in the query.

This is the most easily violated property in the system and the most damaging
when violated. Retrieving a case whose outcome had not yet occurred inserts
future information into a past decision, and the resulting performance is
fictitious in a way that no downstream test detects. Every retrieval path will
carry a test asserting the constraint holds; no retrieval code exists yet, and
this note records the requirement it must satisfy when it does.

## Labeling

The primary label is five-trading-day abnormal return:

```
r_abnormal = r_stock(next_open → close at t+5) − r_benchmark(same window)
```

The benchmark is the sector ETF mapped from the issuer's SIC classification.

**The window opens at the first market open strictly after the event
timestamp.** Filings frequently arrive after the close, and a same-session entry
would attribute a full session of price movement to information that was not yet
public.

Each arm emits both a continuous point estimate and a distribution over three
classes defined by historical terciles of that event type's abnormal return
distribution. The class distribution makes calibration measurable for the
non-LLM arms, which would otherwise be excluded from the most informative
comparison.

Secondary labels derived from the same packets: twenty-day abnormal return, and
the change in realized volatility around the event.

Events with insufficient price coverage are marked unresolvable and excluded.
They are never assigned a zero return.

### Exclusions from the measured population

Three classes of event carry no label, and each is recorded per day beside the
labels rather than dropped, so that a thin day stays distinguishable from a day
whose issuers were unlistable:

- **No listed security.** A current report may be filed by a trust, a shell, or
  an issuer whose securities are not exchange-listed. There is no price series
  to measure and no benchmark to measure against.
- **No plain equity symbol.** Issuers whose only listed security is a preferred
  issue, warrant, or unit are outside the comparison, which is defined over
  common equity. Their return series is driven by rate and structure effects
  that the sector benchmark does not span.
- **Insufficient price coverage.** The horizon window has not closed, or the
  security did not trade across enough of it.

These exclusions are properties of the issuer, not of any arm's forecast, and
they are applied before any arm sees the event. All four arms therefore measure
the same population.

One consequence deserves stating plainly: excluding unlistable and non-common
issuers removes the least liquid tail of filers. Reported results describe
filings by issuers with listed common stock, which is a narrower claim than
"filings," and comparisons against published event-study results should account
for it.

## Screening

Running every arm on every event is not affordable, so `agent` and `arbiter` run
on a subset while `baseline` and `dl` run on everything.

An event is screened in when the baseline and neural arms disagree beyond a
threshold, or when predicted magnitude is large, **or when it is selected by a
random control sample drawn at a fixed rate.**

The control sample is what makes the screen analyzable. Without it, the screened
population is defined by model disagreement, and any comparison on that
population would confound the arms' skill with the screen's selection. With it,
selection effects can be measured directly. The screening rule is versioned and
applied identically in backtest and live evaluation.

Headline comparisons state which population they are computed over.

## Evaluation

Reported per arm and per strategy:

- **Skill** — information coefficient, directional accuracy, RMSE.
- **Calibration** — reliability curves, Brier score, log loss, sharpness.
- **Economic** — shadow portfolio return with modeled transaction costs, hit
  rate, turnover.
- **Cost** — dollars and tokens per decision.
- **Disagreement** — pairwise agreement, and accuracy conditional on
  disagreement.

The last is the most informative. Two approaches with similar aggregate accuracy
that disagree often are doing different things, and which one is right when they
diverge says more than either one's headline number.

## Backtest contamination

The LLM arms were trained on data that covers the backtest period. They may
recall how specific events resolved rather than inferring it, and there is no
way to remove that knowledge from a model already trained.

Backtest and forward results are therefore reported separately and labeled
wherever they appear. Backtested performance for the LLM arms is treated as an
upper bound, not an estimate.

Because labels resolve in five trading days, genuine out-of-sample evidence
accumulates continuously from the day the system goes live. Forward results are
the headline; backtest results establish a reference point.

**The gap between them is a reported result.** It is a direct measurement of how
much apparent skill in a language model's financial forecasts comes from
memorization rather than inference, measured on a multi-step agent pipeline over
primary-source filings rather than on single-shot prompts. Published work on
lookahead bias in LLM forecasts has largely examined the latter.

## Regression control

A frozen benchmark of labeled events, versioned in the repository, with
pre-registered metrics. Any change to prompts, features, models, or screening
must not regress it, and CI fails if it does.

The benchmark set is itself frozen. Changing it is a reviewed change, because a
benchmark that moves with the system it measures measures nothing.

## Known limitations

- **Sector ETFs are a crude benchmark.** They do not adjust for size, value, or
  momentum exposure, so abnormal returns retain factor loadings. A factor-model
  benchmark is a planned refinement.
- **Screening restricts the LLM arms' population.** Mitigated by the control
  sample, not eliminated.
- **Shadow portfolios model transaction costs rather than experiencing them.**
  Slippage on less liquid names is understated.
- **Survivorship.** The issuer universe is constructed from filings, so companies
  that stopped filing mid-sample are handled by exclusion at the event level
  rather than by a point-in-time universe reconstruction.
- **Single-market, single-horizon.** US equities, five-day horizon. Results
  should not be assumed to generalize beyond that.
