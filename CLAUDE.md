# Working in this repository

Guidance for anyone, human or agent, contributing to Arbiter.

## What this project is

Arbiter measures whether LLM agents outperform conventional models at forecasting
short-horizon equity outcomes from SEC filings, when both are given identical
point-in-time information. Its output is a measured claim, not a product. That
framing decides most design arguments: if a change would make published results
harder to reproduce or compare, it is the wrong change regardless of how much it
improves the code.

Read [`docs/methodology.md`](docs/methodology.md) before touching anything in
`packets/`, `retrieval/`, or `evaluation/`.

## Non-negotiables

These are correctness constraints, not preferences. A change that violates one is
a bug even if every test passes.

1. **Evidence packets are immutable and content-hashed.** Every prediction stores
   the hash of the packet it came from. Never mutate a packet after construction.
2. **Nothing downstream of the packet fetches.** The agent analyst is constructed
   with an empty tool list. Keep it that way; the boundary is what makes the
   four-arm comparison valid.
3. **Every retrieval is timestamp-filtered.** Comparable lookups carry the event's
   `as_of` and exclude anything that had not resolved by then. Enforce this in the
   query, never in a prompt.
4. **Predictions are persisted before outcomes exist.** The prediction row is
   written when the forecast is made, never backfilled or amended.
5. **No model output reaches the broker.** Arms emit conviction and direction.
   Position sizing and order construction are deterministic code with hard caps.
6. **Filing text is untrusted input.** SEC filings are attacker-controllable. Always
   pass them as delimited data, never concatenated into an instruction region, and
   always constrain model responses to a schema.
7. **Every model call is metered.** Cost and token counts are recorded at the call
   site and attributed to a packet and a prediction.

## Writing for the reader

Everything committed here — code, comments, docstrings, documentation, commit
messages, and generated reports — is written for an external technical reader
encountering the project without context: a working engineer, a quantitative
researcher, or a reviewer.

Concretely:

- Docstrings state contracts, units, and failure modes. They do not restate the
  signature.
- Comments explain why a non-obvious choice was made, not what the line does.
- Documentation never narrates the development process or addresses a specific
  individual.
- Commit messages read as engineering history: what changed and why, in the
  imperative mood.

## Failure modes this codebase has actually hit

Each of these produced a plausible, passing, wrong result rather than an error.
They are recorded here, not in a document someone would have to think to open,
because by the time you suspect the problem you have already made it.

**Batching couples failures.** Work batched for efficiency needs error handling
written for the *item*, not the batch. This has cost whole days of data four
separate times: one preferred-share symbol rejecting a 75-symbol price request,
one filing without an acceptance time aborting 3,000 others, a price window one
session short deleting every Friday, one refused sector ETF voiding every event
in that sector. Whenever you batch, ask what a single bad element costs. If the
answer is "everything", add per-item isolation.

**Excluded-and-recorded is not dropped-silently.** Conflating them forces a
false choice between corrupting the dataset and crashing on any bad row.
Recording *which* item was excluded and *why* makes exclusion safe and still
lets a rate guard catch systemic breakage.

**An error correlated with the data is worse than a large one.** A window
shortfall that depends on weekday deletes Fridays; a bar misalignment that
depends on trading halts concentrates on exactly the bad news being measured.
When you find a data-handling bug, the question is not "how big" but "is it
correlated with the outcome".

**A test must not restate the implementation.** A twenty-case parameterised
test written to guard the window bug asserted that the window reached the
expression the window is computed from — it read `x >= x` and passed on the
defect. Derive a property from the requirement, never from the code. Prove a
regression test bites by replaying the old behaviour.

**Unit tests here have never caught a real defect.** Every one surfaced in a
live run. A mutation audit found 9 of 22 semantic mutations surviving the whole
suite. Run `pytest -m integration` before believing anything works.

## Engineering standards

Full detail in [`docs/code-standards.md`](docs/code-standards.md). The summary:

- `pyright` in strict mode over `src/`. Public functions are fully annotated.
- Pure functions at the core, I/O confined to the edges.
- Errors are explicit. No bare `except`, no silent fallback to a default that
  hides a failure.
- Tests assert real outcomes. Property-based tests cover risk and sizing math.
- Anything stochastic records its seed alongside its result.
- Configuration comes from the environment, validated at startup, failing fast.

## Commands

```bash
uv sync --all-extras            # install
uv run pytest                   # tests
uv run ruff check --fix .       # lint
uv run ruff format .            # format
uv run pyright                  # types
uv run pre-commit run -a        # everything CI runs
```

## Cost discipline

The system operates under a hard monthly budget, and the dominant cost is
re-sent context rather than call count.

- Order prompts cache-first: stable reference material before anything that
  varies per event. Reversed, every call silently pays full price.
- Cap agent loop turns explicitly. Each additional turn re-sends the whole
  context.
- Route anything latency-insensitive through the batch API.
- Prefer a parser to a model call. Most of this pipeline's volume is XML and
  metadata, and none of it needs inference.
