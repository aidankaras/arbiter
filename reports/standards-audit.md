# Standards audit

**Verdict.** The built portion of the pipeline — EDGAR ingestion, event
extraction, price resolution, and the baseline arm's evaluation — is careful
and mostly holds to the non-negotiables it states for itself: point-in-time
timestamping, per-item batch isolation, and recorded (not silent) exclusions
are implemented correctly everywhere they were checked. The one correctness
defect found is a real one: the market-holiday calendar that governs all
session arithmetic is hardcoded to two years, so trading-day logic silently
misclassifies real holidays outside that window, and the error is correlated
with exactly the calendar dimension CLAUDE.md warns about. The remaining
findings are a packaging bug that breaks the CLI on a base install, a metering
gap that already forecloses a metric the project has promised to report, a
narrow batch-coupling case between two otherwise-isolated ingestion domains,
and several documentation claims that describe work as done when the tree
shows otherwise. Nothing found corrupts the labels or the baseline arm's
published measurement itself, since no report has been committed yet for that
measurement to be checked against.

## Findings

### 1. The trading-day calendar is correct for two years and silently wrong outside them

`src/arbiter/ingestion/edgar.py:35–58` hardcodes `_MARKET_HOLIDAYS` to named
dates in 2026 and 2027 only. `is_trading_day` (`edgar.py:61–68`) is:

```python
return day.weekday() < 5 and day not in _MARKET_HOLIDAYS
```

For any date outside 2026–2027, a real market holiday that falls on a weekday
is reported as a trading day. Concretely: December 25, 2024 was a Wednesday
and the market was closed, but `_MARKET_HOLIDAYS` contains no 2024 dates, so
`is_trading_day(date(2024, 12, 25))` returns `True`. The same holds for any
weekday holiday from 2028 onward — e.g. July 4, 2028, a Tuesday — the instant
the calendar goes stale.

This function is not a peripheral utility. It is the trading calendar for the
whole system: `resolution.py`'s `sessions_after` (`resolution.py:68–83`) walks
it day-by-day to place the entry session, the horizon's closing session, and
the fetch window, and `backfill.sampled_trading_days`
(`backfill.py:87–113`) uses it to decide which days to ingest at all. A day
misclassified as tradeable shifts the counted session index for every event
whose window spans it — the exact failure class CLAUDE.md's own history
warns about ("An error correlated with the data is worse than a large one");
here the correlated dimension is the calendar itself, and it fires precisely
on the historical range a backfill for genuine out-of-sample evidence would
need to cover, since the project's own methodology anticipates measuring a
backtest period distinct from a live one. `edgar.py:151–164` compounds this
for ingestion: a real pre-2026 holiday is treated as a client failure
(`EmptyTradingDayError`) rather than an expected closure, so those days would
be recorded as failures and could contribute to tripping
`SystemicBackfillError` for reasons that have nothing to do with the pipeline.

**Fix shape:** derive the calendar from a maintained exchange-calendar source
(or extend the table to cover the actual date range in use), and fail loudly
if asked about a date outside whatever range is hardcoded, rather than
defaulting to "open."

### 2. Cost cannot currently be attributed to a packet or a prediction

CLAUDE.md's non-negotiable #7 states cost is "attributed to a packet and a
prediction." `LlmCall` (`src/arbiter/db/models.py:64–91`) has nullable
`packet_hash` and `prediction_id` columns for exactly this, with a comment
explaining why: "Cost per decision cannot be derived from a run identifier
alone, because one run scores many events." But the only function that
writes to this table, `record_run` (`src/arbiter/llm/usage.py:106–133`),
takes `payload`, `run_id`, and `agent_name` — no way to pass either column.
Every row it writes has `packet_hash=NULL` and `prediction_id=NULL`
unconditionally; `tests/integration/test_usage_recording.py` confirms this by
never setting either. `docs/operations.md` states "Cost — dollars and
tokens per decision" is a reported metric and that `llm_calls` is "the
durable record" for it; as the call site is currently shaped, that metric is
not recoverable from the table once real per-decision calls start flowing
through `record_run`, because nothing upstream of it has anywhere to put the
packet or prediction identity.

**Fix shape:** add `packet_hash` and `prediction_id` parameters to
`record_run` (or a wrapper used at the actual decision call sites) before the
LLM arms are wired to it.

### 3. One ingestion domain's systemic failure discards the other domain's completed work for that day

`ingest_day` (`src/arbiter/ingestion/pipeline.py:101–188`) extracts the
insider and red-flag domains independently and correctly isolates
per-filing failures within each. But `_quarantine` for the insider domain
runs at `pipeline.py:164–170`, and if the insider domain's parse-failure rate
trips `SystemicParseFailureError`, that exception propagates out of
`ingest_day` before `write_events(redflag_events, ...)` at
`pipeline.py:180` is ever reached — discarding red-flag events that were
extracted correctly and had nothing wrong with them. The day is recorded as
failed by the caller (`backfill.py:168–172`) and will be reprocessed on
retry, so nothing is lost permanently, but a single day's insider-domain
breakage currently costs that call's already-completed red-flag extraction,
which is the pattern CLAUDE.md's "batching couples failures" note asks every
batch to be checked against.

**Fix shape:** write each domain's events before quarantining the other, or
quarantine both domains before either raises, so one domain's systemic
failure doesn't touch the other's completed output for the same call.

### 4. Documentation states work as shipped that is not in the tree

- `reports/README.md:9` states, under a heading titled "What is here now":
  "`baseline-insider.md` — the conventional arm measured out of sample,
  written by `arbiter report`." No such file exists anywhere in `reports/`
  or in git history (`git log --all -- reports/baseline-insider.md` returns
  nothing); `reports/` contains only its own `README.md`.
- `README.md:237` lists `reports/` in the documentation table as "Weekly
  research log, generated automatically" — present tense — while
  `reports/README.md`'s own "What is planned" section says the weekly
  research log "require[s] the three language-model arms and the shadow
  portfolios, none of which exist yet." The two documents contradict each
  other about the same directory.
- `README.md:223–224` states "GitHub Actions for scheduled ingestion,
  evaluation, and reporting." The four workflows in `.github/workflows/` are
  `ci.yml` (push/PR), `claude-code-review.yml` (PR), `claude.yml` (issue
  comments), and `standards-audit.yml` (the only `schedule:` trigger in the
  repository, and it runs a documentation/code-standards audit, not
  `arbiter ingest`/`resolve`/`report`). No workflow schedules the pipeline.

### 5. `scikit-learn` is grouped as an unbuilt-arm dependency but is required by the arm that already ships

`pyproject.toml:41–48` places `scikit-learn` in the `ml` optional group,
alongside `torch`, `transformers`, `peft`, and `xgboost` — dependencies for
the not-yet-built neural arm. But `src/arbiter/cli.py:18` unconditionally
imports `arbiter.arms.evaluate`, which at `evaluate.py:22` imports
`arbiter.arms.baseline`, which at `baseline.py:35–37` does
`from sklearn.linear_model import LogisticRegression` at module scope. A
`uv sync` without `--all-extras` (the plain form of the command named in
`pyproject.toml` itself, distinct from CLAUDE.md's documented
`uv sync --all-extras`) installs a package whose console script fails on
import for the one arm the README calls already built. `structlog>=24.4`
(`pyproject.toml:20`) is a base dependency with no import anywhere in `src/`
or `tests/`.

**Fix shape:** move `scikit-learn` to the base dependency list; drop
`structlog` or start using it.

## Lower-severity: test coverage at declared thresholds

The suite is honest — every metrics and evaluation test check I traced was
tied to an independently computed expected value, and every exclusion path
checked (`test_symbol_batching.py`, `test_pipeline_quarantine.py`,
`test_insider_*`, `test_redflag_*`) is paired with a clean-input case proving
valid data is unaffected. No vacuous assertions and no restated-implementation
tests were found — the specific `x >= x` pattern CLAUDE.md records as a past
defect is now guarded by name in
`tests/unit/test_price_window_spans_the_horizon.py`.

What is missing is coverage at the boundary for several rate/threshold
constants that gate correctness-affecting behavior: `_SYSTEMIC_UNMEASURABLE_RATE`
and `_MIN_RIPE_EVENTS_TO_JUDGE` (`resolution.py:143,147`), `MIN_EVENTS_PER_DAY`
(`metrics.py:41`), and `TRAIN_FRACTION`'s default (`evaluate.py:36`) are each
exercised only far above or far below their threshold, never at the boundary
itself (e.g. 9 vs. 10 vs. 11 ripe events, or 49%/50%/51% failure). A change
from `>` to `>=`, or an off-by-one in the constant, would not be caught by the
current suite. This is a coverage gap, not a demonstrated defect — none of
these constants was shown to misbehave.

I was unable to execute `uv run pytest tests/unit -q` in this environment
(`uv` is not installed and package installation requires approval this audit
does not have); the assessment above is from static reading of every file
under `tests/unit/`, not from a live run.

## What was checked and found sound

- **Point-in-time timestamping.** `insider.py` and `redflags.py` both stamp
  `as_of` from `record.as_of`, which `edgar.py` derives via
  `acceptance_time_utc(filing.header.acceptance_datetime)` — never from
  filing date or period-of-report.
- **Price-window arithmetic.** `resolution.py`'s session counting, entry/exit
  selection, and issuer/benchmark date-pairing (guarding against a halted
  issuer silently borrowing a benchmark return from a different calendar
  span) are correct and match their own stated rationale.
- **Batch isolation.** Every batched fetch or bulk parse checked
  (`daily_bars_tolerating_gaps` splitting a rejected symbol batch,
  `resolve_day` isolating one event's `UnresolvableEventError`, `ingest_day`
  isolating one filing's parse failure) records the excluded item with a
  reason rather than dropping it or aborting the batch, with systemic-rate
  guards distinguishing "a few odd filers" from "the request itself is
  broken."
- **No silent defaults.** Missing prices, missing holdings, and missing
  tickers are all preserved as `None` with an explicit indicator feature
  rather than filled with zero; `abnormal_return` raises rather than
  returning a value when an entry price is zero.
- **Statistical design.** `information_coefficient` is computed within-day
  and averaged across days rather than pooled; discrimination, calibration,
  and the base rate are all credited once per issuer-day, not once per
  filing row; `FEATURE_NAMES` excludes the two fields the screen makes
  constant or collinear, and no other feature was found to be so.
- **Immutability.** `Prediction` and `LlmCall` rows are protected by a
  database-level `BEFORE UPDATE OR DELETE` trigger
  (`migrations/versions/0001_create_append_only_ledger.py`), not just
  application convention.
- **CI workflow hygiene.** All third-party actions are pinned to version
  tags, every workflow declares scoped `permissions:`, and no workflow
  exposes a secret to a fork's untrusted PR head.
