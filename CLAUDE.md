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

## Reporting standards for anyone dispatching or reviewing work here

Most of these describe one failure: **a check that passes without having run.** The
test for whether a rule belongs here is to ask what the check would print if it had
never happened. If that is the same thing it prints on success, it belongs, because
nothing downstream can tell the two apart.

- **A claimed test count is verified, not trusted.** Run `uv run pytest tests/unit -q`
  and read the tail line before treating any "N passed" as done. A report is the only
  evidence a reviewer has unless they rerun the suite themselves, and this project has
  already had an error message mistaken for a clean exit when it was the tail of a
  traceback.
- **A pinned Action SHA is verified against the source**, never copied from a plan or
  another repository: `gh api repos/<org>/<repo>/git/ref/tags/<tag> --jq .object.sha`.
- **A mutation test clears `__pycache__` before it runs.** CPython decides a cached
  module is current from the source's size and its modification time truncated to whole
  seconds, so a mutation that preserves the byte count and is tested within a second of
  being written executes the *previous* bytecode. The suite passes, and that outcome is
  indistinguishable from a mutation the tests genuinely failed to catch. Flipping one
  digit for another, or `<` for `>`, is size-preserving, which makes the sharpest
  mutations the ones most likely to be reported wrongly:

  ```bash
  find . -name __pycache__ -type d -prune -exec rm -rf {} +
  uv run pytest tests/unit -q -p no:cacheprovider
  ```
- **A comment explaining *why* is a second claim, and the tests do not check it.**
  A passing suite confirms the conclusion and says nothing about the reason given for
  it, so an explanation can be wrong in a file that is entirely correct — and it is
  read by the next person as though it had been verified. This has already happened
  here: a docstring justified hashing the packet in Python mode on the grounds that
  JSON mode "would turn every price into a float", which is false. JSON mode renders a
  `Decimal` to a string and loses no precision. The decision was right for a different
  reason — it renders before the canonical form can normalise, so `1.50` and `1.5`
  become two identities — and the stated reason would have taught a reader something
  untrue about the serialiser. When a comment explains a mechanism, check the
  mechanism.
- **Planning documents live outside this repository.** The spec and phase plans are kept
  under `~/.claude/references/arbiter/`, never in this tree, so that planning context
  never reaches a public reader. Design reasoning that belongs to the system goes in
  `docs/`; reasoning about why the project exists does not go in the repository at all.

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

An early mutation audit reported 9 of 22 semantic mutations surviving the suite,
and that figure is no longer cited because the mutations behind it were never
written down. It cannot be reproduced, and it was produced without clearing the
bytecode cache, which biases such a count toward *over*-stating weakness: a
size-preserving mutation that never took effect leaves the suite green and is
recorded as one the tests failed to catch. A summary that outlives its evidence
is an anecdote, whichever direction it errs in — record the mutation list with
the count, or report neither.

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
