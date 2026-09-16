# Operations

## Telemetry

Scheduled runs export OpenTelemetry metrics and logs carrying per-request cost,
token counts, model, and the originating subagent or skill. Any OTLP-compatible
backend can receive them; the variables below target Langfuse Cloud.

```bash
export CLAUDE_CODE_ENABLE_TELEMETRY=1
export OTEL_METRICS_EXPORTER=otlp
export OTEL_LOGS_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT="https://us.cloud.langfuse.com/api/public/otel"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic ${LANGFUSE_AUTH},x-langfuse-ingestion-version=4"
```

`LANGFUSE_AUTH` is the base64 encoding of `public_key:secret_key`:

```bash
export LANGFUSE_AUTH=$(echo -n "pk-lf-...:sk-lf-..." | base64 -w 0)
```

Hosted trace retention is measured in weeks, so that backend serves interactive
debugging only. The durable record is the `llm_calls` table, which every run
writes to directly through `arbiter.llm.usage.record_run`.

## Cost attribution

A single run routinely invokes more than one model, so usage is recorded one row
per model per run rather than one row per run. Attributing a run's whole cost to
a single model name would misreport cost per decision, which is a headline
metric rather than an operational detail.

## Spend ceilings

`DAILY_SPEND_CEILING_USD` and `PER_RUN_TOKEN_CEILING` are enforced in code before
work is dispatched. Crossing either halts the run rather than reducing its scope,
because quietly doing less work would make the run's results incomparable with
earlier ones.

## Migrations

Migrations read the database URL from the validated settings object, so the
environment must supply `DATABASE_URL` and `SEC_USER_AGENT` before running:

```bash
uv run alembic upgrade head
```
