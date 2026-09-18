# Arbiter

[![CI](https://github.com/aidankaras/arbiter/actions/workflows/ci.yml/badge.svg)](https://github.com/aidankaras/arbiter/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Event-driven equity forecasting from SEC filings, built to answer a question that
most applied-LLM systems leave unmeasured: **does an LLM agent actually beat a
conventional model when both are given exactly the same information?**

Arbiter ingests SEC filings daily and freezes each event into an immutable
point-in-time evidence packet. Four independent forecasting approaches will then
score that identical packet, each tracked as a shadow paper portfolio with full
decision provenance and per-decision cost accounting.

> **Status: the measurement pipeline and the conventional arm are built; the
> three language-model arms are not.**
>
> Working today: EDGAR ingestion for Form 4 and 8-K, event extraction for the
> insider and red-flag domains, a date-partitioned event store, market data,
> abnormal-return labels, a resumable historical backfill, the baseline
> forecasting arm, out-of-sample evaluation by information coefficient and
> calibration, an append-only ledger, and spend metering.
>
> Designed but not built: the evidence packet builder, the three language-model
> arms, the dashboard, and the paper portfolios. Sections describing those use
> the future tense; anything in the present tense refers to code in this
> repository. Paper trading only, always.

## Quickstart

```bash
git clone https://github.com/aidankaras/arbiter && cd arbiter
uv sync --all-extras
cp .env.example .env          # set SEC_USER_AGENT: "identifier your@email"
uv run arbiter ingest 2026-08-10
```

```
insider: 755
redflag: 46
rejected: 0
unpriceable: 34
```

That command fetches every Form 4 and 8-K accepted on the given day, extracts the
qualifying events, and writes them to `data/events/{domain}/{date}.parquet`.
Re-running a date replaces that date's partition, so a failed run is repeated
rather than repaired.

The last two counts are different facts and are recorded separately. `rejected`
is filings that could not be parsed, which is a defect to investigate; a day
where too many fail is refused outright rather than recorded, on the reasoning
that a format change has broken it. `unpriceable` is filings parsed correctly
whose issuer has no listed common stock — insiders at companies with only
registered debt file Form 4 like anyone else — which is an ordinary property of
the population and is excluded from that share.

Once an event's outcome window has closed, the same day is labeled:

```bash
uv run arbiter resolve 2026-08-10 insider
uv run arbiter resolve 2026-08-10 redflag
```

```
labels-insider: 751
labels-redflag: 40
```

Labeling lags ingestion. The horizon is five sessions for insider events and
twenty for red flags, and the consolidated tape will not serve a window ending
on the current session — so a day becomes measurable only after its window has
closed, and a day asked for too early yields nothing rather than a partial
measurement. Issuers with no listed common stock are recorded under
`unpriceable/` beside the labels, so a thin day stays distinguishable from a day
whose filers were unlistable.

Market data additionally requires `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`;
EDGAR ingestion needs no credentials beyond the contact string the SEC requires.

### Building a history and measuring against it

One day proves the pipeline runs. Measuring anything needs a history:

```bash
uv run arbiter backfill 2026-03-02 2026-08-14 --every 4
uv run arbiter report --domain insider
```

`backfill` ingests and labels a range of trading days, writing its progress as
it goes. Days already stored are skipped, so an interrupted run is restarted
rather than repaired.

`--every` is the lever on how long a run takes. A day costs the same regardless
of how many of its filings qualify, because each filing must be fetched to find
out and the acceptance timestamp that makes an event point-in-time is not
carried in EDGAR's bulk index. Volume varies roughly fourfold across the year —
736 Form 4 filings on one sampled day against 3,001 in early March, when annual
grants and vesting cluster — so a day takes between three and fourteen minutes.
Sampling every Nth trading day spreads observations across months at the cost of
a contiguous block, which matters because events filed on one day share a market
factor the sector benchmark only partly removes.

`report` fits the baseline arm on the earlier fraction of the stored days and
scores it on the rest, writing the result to [`reports/`](reports/). The split
is chronological, never random: a random split would place events from one day
on both sides and report as skill what is partly memory.

---

## The four arms

Each event is scored by four approaches that see byte-identical inputs:

| Arm | Approach | Sees text | Uses an LLM | Built |
|---|---|---|---|---|
| `baseline` | Logistic regression over structured filing features | No | No | Yes |
| `dl` | Fine-tuned transformer encoder with a tabular head | Yes | No | No |
| `agent` | Agent team with per-domain analysts | Yes | Yes | No |
| `arbiter` | A different model family reviewing the other three and their reasoning | Indirect | Yes | No |

`baseline` is the null hypothesis. Anything that cannot beat a logistic
regression on a handful of features has not demonstrated anything.

It is deliberately weak, and stays that way. An elaborate control would confound
the question being asked: if a gradient-boosted ensemble beat the agent, the
finding would be about model capacity rather than about reading a filing. Its
regularisation is left at the library default for the same reason — tuning it
against the evaluation data would report the best of several attempts as though
it were one measurement.

The `arbiter` arm is a fourth prediction, not a gate. It never vetoes the other
arms, because an arm whose live record has been filtered by another model is no
longer comparable to anything. All four maintain shadow portfolios; only the
arbiter's book is mirrored to the paper brokerage account.

## Why the evidence packet is the center of the design

A comparison between approaches is only meaningful if the approaches saw the same
thing. Arbiter enforces that structurally rather than by convention.

Each event produces exactly one **evidence packet**: an immutable, content-hashed
object stamped with the filing's EDGAR acceptance time. It contains the filing
text, issuer metadata, market context truncated at that timestamp, and
comparable historical cases restricted to those that had already resolved.

Downstream of the packet, nothing fetches. The agent analyst is constructed with
an empty tool list, so it is not capable of looking anything up. Every prediction
records the hash of the packet it was made from, which makes any published result
reproducible and any silent change to packet construction detectable.

This also closes the most common source of invalid results in event-study work:
retrieving a "similar past case" whose outcome had not yet occurred at the time
of the event being predicted. Comparable lookups carry the event's own timestamp
and filter on it in the query itself.

## Strategies

Three event families share one platform. Adding a fourth should require no
platform changes.

- **Earnings** — 8-K Item 2.02 plus XBRL company facts. Surprise is computed as a
  seasonal random walk against the same quarter last year, scaled by the
  volatility of trailing surprises, which avoids any dependency on paid analyst
  consensus data.
- **Insider transactions** — Form 4 ownership filings. Role, transaction code,
  size relative to existing holdings, and clustered buying. Almost entirely
  structured; little text.
- **Corporate red flags** — 8-K items covering auditor changes, non-reliance and
  restatement notices, and executive departures. Low volume, high text
  dependence.

The contrast between the second and third is the most informative comparison the
system can produce: if language models hold an advantage, it should appear where
the signal lives in prose and disappear where it lives in a table.

## Target and evaluation

The primary target is five-trading-day abnormal return against a sector ETF,
entered at the first open after the filing's acceptance timestamp. Every arm
emits both a point estimate and a class distribution, so calibration is
measurable for the non-LLM arms too.

Reported per arm and per strategy: information coefficient, directional accuracy,
reliability curves, Brier score, shadow portfolio performance with modeled costs,
cost per decision, and pairwise disagreement with conditional accuracy.

**Backtest and forward results are reported separately and labeled everywhere.**
The LLM arms were trained on data covering the backtest period, so backtested
performance overstates their ability by an unknown amount. Because labels resolve
in five trading days, genuine out-of-sample evidence accumulates continuously, and
the gap between backtest and forward performance is itself a measured result.

## Architecture

```
ingestion (no LLM) → event detection (no LLM) → evidence packet (immutable, hashed)
                                                        │
                        ┌───────────────┬───────────────┼───────────────┐
                     baseline          dl             agent          arbiter
                        └───────────────┴───────────────┼───────────────┘
                                                        │
                        predictions written before any outcome exists
                                                        │
                    ┌───────────────────┬───────────────┴──────────────┐
              shadow portfolios    t+5 label resolution          evaluation
                    └───────────────────┴──────────────┬──────────────┘
                                                        │
                                        dashboard + weekly research report
```

Language models are confined to extraction and judgment. Position sizing and
order placement are deterministic code, so no model output reaches the broker
without passing through explicit risk limits.

## Stack

Python 3.12 with `uv`. Ingested events are stored as date-partitioned Parquet;
PostgreSQL holds the append-only prediction and usage ledger, and will hold
`pgvector` embeddings for comparable retrieval once that exists. LangGraph for orchestration. FastAPI for the
API and dashboard. PyTorch with `transformers` and `peft` for the neural arm.
Docker Compose for local and deployed environments. GitHub Actions for scheduled
ingestion, evaluation, and reporting.

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Component boundaries and data flow |
| [`docs/methodology.md`](docs/methodology.md) | Leakage controls, labeling, evaluation design |
| [`docs/code-standards.md`](docs/code-standards.md) | Engineering standards for this repository |
| [`docs/data-provenance.md`](docs/data-provenance.md) | Sources, licensing terms, and what is redistributed |
| [`docs/operations.md`](docs/operations.md) | Telemetry, spend ceilings, and running migrations |
| [`docs/strategies/`](docs/strategies/) | Per-strategy event definitions and features |
| [`docs/lessons/`](docs/lessons/) | Concept notes covering the design decisions behind each subsystem |
| [`reports/`](reports/) | Weekly research log, generated automatically |

## Roadmap

- [x] Repository scaffold, CI, and engineering standards
- [x] Append-only ledger, usage metering, and spend ceilings
- [x] EDGAR ingestion: Form 4 and 8-K parsing, event extraction, Parquet store
- [x] Market data and abnormal-return labels
- [ ] XBRL facts and the earnings strategy
- [ ] Evidence packet builder with timestamp enforcement and hashing
- [ ] Baseline arm and evaluation harness
- [ ] Neural arm
- [ ] Public dashboard and shadow portfolios
- [ ] Agent arm
- [ ] Arbiter arm and disagreement analytics
- [ ] Automated weekly report and benchmark regression gate
- [ ] Distilled extraction model
- [ ] Methodology writeup

## Disclaimer

Research software. Paper trading only. Nothing here is investment advice, and no
component of this system is designed or tested for use with real capital.

## License

MIT. See [LICENSE](LICENSE).
