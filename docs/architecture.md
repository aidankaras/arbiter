# Architecture

Component boundaries, data flow, and the reasoning behind the structural choices.
For the properties these components exist to preserve, see
[`methodology.md`](methodology.md).

## Data flow

```
┌──────────────────────────────────────────────────────────────┐
│ INGESTION                                    no inference     │
│ EDGAR daily index → filing fetch → parse → normalize → store │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ EVENT DETECTION                              no inference     │
│ form type + item codes + XBRL facts → Event rows             │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ PACKET CONSTRUCTION                                           │
│ text sectioning, market context truncated at as_of,          │
│ comparable retrieval filtered by as_of, content hashing      │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
        ┌──────────┬─────────┴──────────┬──────────┐
     baseline      dl                 agent     arbiter
        └──────────┴─────────┬──────────┴──────────┘
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ PREDICTION STORE                                              │
│ four rows per screened event, each carrying a packet hash,   │
│ written before any outcome is known                          │
└────────────────────────────┬─────────────────────────────────┘
                             ▼
     ┌───────────────┬───────┴────────┬────────────────┐
  portfolios    label resolution   evaluation      cost ledger
     └───────────────┴───────┬────────┴────────────────┘
                             ▼
              dashboard  +  weekly research report
```

## Modules

Modules that exist today:

| Module | Responsibility | Depends on |
|---|---|---|
| `ingestion` | EDGAR client, filing parsers, market data, event store | `events` |
| `events` | Trigger evaluation, event normalization | `ingestion` |
| `evaluation` | Abnormal return labels | — |
| `llm` | Usage metering and spend ceilings | `db` |
| `db` | Models, session management, migrations | — |

Modules the design calls for, not yet written. They are described here so the
boundaries are settled before code arrives; none of them exists in the tree, and
the packages are created when their first module is:

| Planned module | Responsibility |
|---|---|
| `packets` | Packet construction, schema, hashing, timestamp enforcement |
| `retrieval` | Vector and relational comparable lookup, as-of filtering |
| `arms/*` | The four forecasting approaches |
| `portfolio` | Sizing, risk limits, shadow books, broker adapter |
| `reporting` | Weekly report generation |
| `api` | Dashboard data publication |

Dependencies run one direction. `packets` does not import from `arms`, and no arm
imports from another. An arm that needed to know what another arm predicted would
break the independence the comparison depends on; only `arbiter` sees other arms'
output, and it receives it as data through the prediction store rather than by
importing them.

## Storage

**Today:** ingested events are written as date-partitioned Parquet, one file per
domain per day, and PostgreSQL holds the append-only prediction and usage
ledger. Parquet suits the event store because it is written once per day, read
in full by whatever consumes it, and never updated in place.

**Planned, once comparable retrieval exists:** a single PostgreSQL instance with
the `pgvector` extension serving both relational and similarity queries.

The queries this system actually runs are hybrid: similar text, *and* the same
event type, *and* within a market-cap band, *and* resolved before a given
timestamp. Splitting text similarity into a dedicated vector database would force
the filter and the similarity search into separate systems joined in application
code, which is slower, harder to reason about, and one more service to operate.

Embeddings are computed locally with `sentence-transformers`. At this filing
volume, a paid embedding API would be a significant recurring cost, and Form 4 is
structured XML with essentially no prose worth embedding. Only narrative filings
are embedded.

## Model routing

Inference is confined to two tasks: extracting structure from filing text, and
producing a judgment from a packet. Everything else — routing, classification,
feature construction, labeling, metrics, sizing — is deterministic code, which is
where the overwhelming majority of the volume lives.

| Task | Path | Rationale |
|---|---|---|
| Routing, classification, structured fields | No inference | XML and metadata; a parser is faster, free, and exact |
| Bulk text extraction | Anthropic SDK, small model, batched and cached | High volume, tightly specified output |
| Extraction escalation | Larger model | Fires on a small fraction where confidence is low |
| Analyst judgment | Claude Agent SDK inside LangGraph nodes | Low volume, judgment-heavy, benefits from an agent loop |
| Arbiter judgment | Separate provider | Decorrelates judge errors from the arm being judged |
| Embeddings | Local | Volume makes a hosted API uneconomic |

## Orchestration

LangGraph defines the agent arm's graph: typed state, explicit edges,
checkpointing so an interrupted run resumes rather than restarting, and
interrupt points for human review.

The daily pipeline is a sequence of idempotent steps keyed by date, not a
streaming system. Filings arrive in daily batches and labels resolve on a
five-day horizon, so an event bus would add operational surface without changing
any outcome. Re-running a day produces the same result.

## Scheduling and deployment

Scheduled work runs on GitHub Actions:

| Cadence | Work | Inference cost |
|---|---|---|
| Daily | Ingest, detect, build packets, run `baseline` and `dl`, resolve labels, recompute metrics, publish dashboard data | None |
| Daily | Extraction and the two LLM arms over the screened subset | Metered |
| Weekly | Benchmark run, one proposed improvement as a pull request, research report | Metered |
| Monthly | Dependency, documentation, and repository audit | Metered |

The application is containerized and runs identically locally and in CI. The
public dashboard is published as a static build generated by the daily job, so
serving it requires no always-on host.

## Cost and safety controls

- Every model call records model, input tokens, cached tokens, output tokens, and
  computed cost, attributed to the run and agent that made it. Attribution to a
  packet and a prediction is the goal and is not yet possible: the ledger table
  carries no packet hash or prediction reference, so cost per decision cannot be
  computed until those columns and the predictions that populate them exist.
- A per-run token ceiling terminates a run that exceeds it. A prompt requesting
  efficiency is advisory; a ceiling is enforced.
- A daily spend ceiling halts the pipeline.
- Filing text is passed as delimited, untrusted data and responses are
  schema-constrained, because filings are attacker-controllable input.
- Arms emit conviction and direction. Order construction is deterministic code
  with hard position and concentration limits, so no model output reaches the
  broker unchecked.
