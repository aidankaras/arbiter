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
`packets/`, `arms/` or `evaluation/`, and before building comparable retrieval.

## Non-negotiables

These are correctness constraints, not preferences. A change that violates one is
a bug even if every test passes.

1. **Evidence packets are immutable and content-hashed.** Every prediction stores
   the hash of the packet it came from. Never mutate a packet after construction.
2. **Nothing downstream of the packet fetches.** The agent analyst, when built, is
   constructed with an empty tool list. Build it that way; the boundary is what makes the
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

## Verification standards

Most of these guard against one failure: a check that passes without having run.
If a check would print the same thing whether or not it happened, verify it.

- **Verify a claimed test count.** Run `uv run pytest tests/unit -q` and read the
  tail line before treating any "N passed" as done.
- **Verify a pinned Action SHA against its source**, never copy it from a plan or
  another repository: `gh api repos/<org>/<repo>/git/ref/tags/<tag> --jq .object.sha`.
- **Run mutations with `scripts/mutate.sh <file> "<old>" "<new>" "<label>"`.** It
  clears `__pycache__` first: CPython treats cached bytecode as current when the
  source's size and whole-second mtime match, so a size-preserving mutation tested
  within a second runs the previous code and reads as a surviving mutant. It also
  refuses a mutant that does not parse and verifies the file was restored.
- **Check the mechanism behind any comment that explains one.** A passing suite
  confirms the code's conclusion, not the reason a comment gives for it.
- **Review one seam outward.** Defects that survive review live between units.
  After scoping a review question, also check the caller, the level below, the
  empty input, and any neighbour sharing state.
- **Put design reasoning in `docs/`**, next to the code it constrains, not only in
  commit messages or pull requests.

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

The sharpest instance so far withdrew a published report. One transaction filed
by three joint owners was stored as three events, and joint filing is how funds,
groups and ten-percent owners file while officers file alone — so the
over-weighting tracked filer type, which is the property the study asks the data
to discriminate on. A second, older defect stored a filing's single qualifying
transaction once per line on the form, inflating the event store by a factor
averaging 2.94 and ranging from 2.34 to 4.59 across days. Both produced
plausible datasets; the second was found only because a day's packets yielded
fewer distinct hashes than packets.

**A test must not restate the implementation.** A twenty-case parameterised
test written to guard the window bug asserted that the window reached the
expression the window is computed from — it read `x >= x` and passed on the
defect. Derive a property from the requirement, never from the code. Prove a
regression test bites by replaying the old behaviour.

**Hand-built fixtures encode what a filing was assumed to contain.** Every
ingestion defect found here surfaced in a live run rather than in a test, and
that is the reason: a test that constructs a Form 4 table tests the shape its
author already had in mind, and the filings that break things have the shape
nobody anticipated — several transactions of which one clears the threshold,
one transaction reported once per joint owner, a conversion carrying no price.

`tests/unit/test_form4_contract.py` reads tables recorded from EDGAR and
committed under `tests/fixtures/form4/`, with each filing's expected event count
pinned. Add a fixture with `tests/fixtures/form4/record.py` whenever a filing
exhibits a shape the suite does not cover, and name in the test what that shape
is. A diff in a fixture is a change in the upstream contract and is reviewed as
one, never refreshed away.

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
