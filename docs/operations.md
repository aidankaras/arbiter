# Operations

## Telemetry

*Planned for the language-model arms; no model runs today.* Their runs will
export OpenTelemetry metrics and logs carrying per-request cost, token counts,
model, and the originating subagent or skill. Any OTLP-compatible backend can
receive them; the variables below target Langfuse Cloud.

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
debugging only. The durable record is the `llm_calls` table, written directly
through `arbiter.llm.usage.record_run`.

## Cost attribution

A single run routinely invokes more than one model, so usage is recorded one row
per model per run rather than one row per run. Attributing a run's whole cost to
a single model name would misreport cost per decision, which is a headline
metric rather than an operational detail.

## Spend ceilings

`DAILY_SPEND_CEILING_USD` and `PER_RUN_TOKEN_CEILING` are checked by the agent
arm before each forecast group is dispatched, against the worst case that group
can cost: every sample retried once, each priced at the most expensive call the
run has seen so far. The daily total resets at midnight Eastern. Crossing either
ceiling halts the run rather than reducing its scope, because quietly doing less
work would make the run's results incomparable with earlier ones.

Costs are what Claude Code reports for each call. On a subscription they are
notional, so the binding limit is usually the subscription's own usage windows:
every call reports how much of its five-hour and seven-day windows is used, and
`--max-weekly-utilization` (default 0.80) stops a run once the seven-day window
reaches it, leaving the rest of the week's allowance for other work.

## The local ledger

The ledger's guarantees are enforced by Postgres triggers and constraints, so it
needs real Postgres. On a machine without a database server, `uv sync --extra
ledger` installs an embedded one, and

```bash
uv run arbiter ledger-up
```

starts it in `data/ledger/`, applies every migration, and prints the
`DATABASE_URL` to put in `.env`. The URL is stable for that directory. The server
keeps running after the command exits but not across a reboot, which is why the
agent commands start it themselves when it is down.

## Migrations

Migrations read the database URL from the validated settings object, so the
environment must supply `DATABASE_URL` and `SEC_USER_AGENT` before running:

```bash
uv run alembic upgrade head
```
