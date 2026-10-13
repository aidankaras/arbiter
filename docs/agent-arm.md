# The agent arm

The agent arm asks one question: does a single language-model analyst beat the
logistic-regression baseline at forecasting five-session abnormal returns after
insider trades, when both are scored on identical issuer-days by the same
machinery? This document records how the arm is built, the rules it runs under,
and the protocol for the scored run. The arm is built; **the scored run has not
been made**, and the steps under [Protocol for the scored run](#protocol-for-the-scored-run)
describe work still to do.

## Design

| Decision | Choice | Reason |
|---|---|---|
| Model | `claude-sonnet-5`, pinned | Released 2026-06-30, so its training data cannot contain the scored window's outcomes. The 5.5-generation models have a June 2026 training cutoff, which could. |
| Scored window | 2026-07-01 to 2026-09-09 | Starts after the model's release, the only provable bound on what it was trained on. June is the development set. |
| Domain | Insider (Form 4) only | Red-flag packets are not built. |
| Population | Every issuer-day in the window | The baseline's population, so the comparison is paired. |
| Unit of forecast | A forecast group: one issuer's filings on one Eastern date, split at the 09:30 open | The baseline credits one forecast per issuer per day. The split is conservative: under the label rule as implemented (#60) filings either side of the open share a window, so it costs a few extra groups and never shows a forecast a later filing. |
| Samples | Three independent samples per group, averaged | Self-consistency is the cheapest known calibration gain. |
| Company identity | Visible (ticker, name, sector ETF) | Knowledge of a company from before the model's cutoff is legitimate point-in-time information, and the arm's main potential edge. |
| Runtime | Headless Claude Code (`claude -p`) on a subscription login | Fixed cost; the `Runtime` protocol admits an API adapter. |
| Ledger | The append-only Postgres ledger, run locally by `arbiter ledger-up` | Forecasts are written before outcomes exist and cannot be edited. |

## What the analyst sees

`arms/agent/prompt.py` renders a forecast group's packets between
`<untrusted-data>` delimiters: the company, each filing's extracted fields, the
market statistics and daily bars of the earliest filing (the ones knowable
before every filing in the group). Filed text is folded to NFKC, its whitespace
collapsed and its `|` replaced, so it cannot forge a filing row, a column, or a
delimiter in any case or look-alike form. The answer is constrained by a JSON
schema: `p_up` between 0 and 1 and a rationale of at most 600 characters.

No module in `arms/agent/` may import evaluation, ingestion, features or another
arm; a test reads every module's imports.

`prompt_version()` hashes the template, the schema, the model, the effort, the
sample count, and the source of `prompt.py` and `runtime.py`. Every prediction is
stored under it, so forecasts made under different prompts or code are never
pooled. An edit to either file, comments included, is a new version.

## Isolation

`arms/agent/runtime.py` runs each sample as one `claude -p` call in an empty
temporary directory, inheriting only `PATH`, `HOME`, `LANG`, `USER` and `TMPDIR`
from the caller. Every call's own startup event is checked, and a call that fails
any check stops the run: the only tool is the one that returns the structured
answer; no MCP servers, skills, slash commands or plugins beyond the three
Claude Code loads by itself; the default output style and permission mode; the
pinned model; no API key (`apiKeySource` must be `none`); and no memory directory
with content.

Claude Code still adds a short block of its own to every call: the account's
email, the working directory and platform, the model's name and today's date. No
flag that keeps the subscription login removes it. None of it carries a price or
an outcome; the template tells the model that the date it sees is when the
archive is processed, not when the filings were made.

## Failure rules

- A sample that is invalid or refused is redrawn once. If it fails again, the
  group is recorded in `agent_exclusions` with its reason and the number of
  valid samples, and is never averaged over fewer than three.
- An error or an exhausted usage window halts the run without recording the
  group, so an outage can never become an exclusion. A usage rejection is
  treated as an outage even when the process exits cleanly.
- Every call is metered before any decision is taken on it.
- Groups already forecast or excluded under the current version are skipped, so
  a run resumes where it stopped. A group whose filings changed after some of
  them were forecast or excluded (a re-ingest that adds a late filing) is counted
  as `changed` and left alone; the comparison drops it from both arms.

## Limits and stops

Before each group the run checks that the group's worst case (three samples,
each redrawn) fits under every limit:

| Limit | Set by |
|---|---|
| Calls this invocation may make | `--max-calls` |
| Notional spend since midnight US Eastern | `DAILY_SPEND_CEILING_USD` (default 4.00) |
| Tokens this run has used | `PER_RUN_TOKEN_CEILING` (default 250,000) |
| Seven-day subscription usage, as the last call reported it | `--max-weekly-utilization` (default 0.80) |

A call that does not report the seven-day window stops the run rather than being
read as unused. A fresh invocation can run one group before it has seen the
window. The defaults are below a day of the arm; raise both ceilings in `.env`
before the pilot.

| Command | Exit code | Meaning |
|---|---|---|
| `agent-pilot`, `agent-run` | 0 | Nothing left to do within the limits given |
| | 2 | A day in the range was never built, or the pilot was given a scored day |
| | 3 | Stopped at a limit or an outage; the reason is on stderr; rerun to resume |
| `agent-compare` | 2 | No forecasts under the requested version |
| | 4 | The window is not finished; the numbers printed are partial |

## Scoring

`evaluation/comparison.py` scores both arms with the baseline's own machinery:
the walk-forward forecasts, the crediting of a day's filings to one forecast per
issuer, the daily information coefficient and the overlap-robust standard error
(five lags, the outcome horizon).

- An issuer-day enters only if the agent forecast every labelled filing in it;
  otherwise it leaves both arms.
- Every window issuer-day is counted as compared, dropped (the agent did not
  forecast all its filings) or unranked (a day left with too few issuers, or one
  the agent forecast flat). The three sum to the window's issuer-days.
- The agent line, the baseline line and the difference line cover the same days:
  those on which both arms rank.
- The baseline is trained on stored days up to the window's end only, so data
  stored later never moves it. On the store as built it scores IC +0.0289,
  t 1.31, over 49 days and 3,211 issuer-days.

The primary metric is the mean daily difference, agent minus baseline, with its
overlap-robust t-statistic.

## Protocol for the scored run

None of these steps has been taken.

1. **Check the runtime.** After any Claude Code update, and before every long
   run: `uv run pytest -q -m integration tests/integration/test_agent_runtime_live.py`
   (one real call). A changed startup event or stream shape fails here first.
2. **Pilot one June day.** `uv run arbiter ledger-up`, put the printed
   `DATABASE_URL` in `.env`, raise `DAILY_SPEND_CEILING_USD` and
   `PER_RUN_TOKEN_CEILING`, then
   `systemd-inhibit --what=sleep uv run arbiter agent-pilot 2026-06-15 --max-calls 400`.
   Record forecasts, exclusions by reason, calls, notional cost and tokens
   (`select sum(cost_usd), sum(input_tokens + output_tokens + cache_read_tokens + cache_creation_tokens) from llm_calls where run_id = '<run>'`),
   wall-clock time, and the seven-day utilization before and after. Inspect the
   sample spread (`select class_probs->'samples' from predictions where arm = 'agent'`).
   If nearly every sample sits at 0.50 to 0.55 the prompt may be changed now,
   on June data only, which changes the version.
3. **Pre-register.** Resolve #60 first. Freeze `prompt.py`, `runtime.py` and the
   template, then commit `docs/preregistration-agent-insider.md` stating: the
   model; the `prompt_version` the pilot printed; `claude --version`; the window;
   three samples at medium effort; the primary metric and its test; the
   exclusion rule; the per-invocation call cap and weekly ceiling chosen from the
   pilot; the baseline figures above; and that the result is published whichever
   way it falls. The commit's timestamp is the pre-registration.
4. **Run the window.** From a separate checkout, pointing `--ledger-dir` at the
   ledger `ledger-up` created:
   `systemd-inhibit --what=sleep uv run arbiter agent-run --max-calls N`, repeated
   until it reports no forecasts and exits 0. Exit 3 means rerun later. Measure
   progress from the ledger, not the log:
   `select count(distinct class_probs->>'issuer_day_hash') from predictions where arm = 'agent' and prompt_version = '<v>'`.
5. **Compare.** `uv run arbiter agent-compare --prompt-version <v>`. Exit 0 means
   the window is complete.
6. **Publish** `reports/agent-insider.md` with the comparison, each arm's IC,
   cost per scored issuer-day from `llm_calls`, and exclusions by reason.

## Known limitations

- Pre-market filings enter the outcome window a session later than
  `docs/methodology.md` states (#60). Both arms share the labels; the prompt
  describes the rule as implemented.
- The refusal and rate-limit stream shapes the classifier expects are inferred
  from recorded fields, not recorded themselves.
- Lesser review findings are listed in #61.
