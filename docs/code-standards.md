# Code standards

The standards this repository holds itself to, and the reasoning behind the ones
that are not self-evident. Automated checks enforce what can be automated; the
rest is review discipline.

## Types

`pyright` runs in strict mode over `src/`. Every public function is fully
annotated, including return types.

Types carry domain meaning rather than restating primitives. A function that
accepts a `Ticker` and an `AsOf` is harder to misuse than one accepting two
strings, and the compiler catches the argument-order mistake that would otherwise
surface as a silently wrong backtest. Where a value has units, the type or the
name says so: `abnormal_return_bps`, not `value`.

Domain invariants are expressed in the type where possible. An `EvidencePacket`
is frozen. A `Prediction` cannot be constructed without the hash of the packet it
came from. Making an invalid state unrepresentable removes a category of test.

## Structure

Pure functions at the core, I/O at the edges. Parsing, feature construction,
labeling, metrics, and sizing are pure: same inputs, same outputs, no clock, no
network, no database. Fetching and persistence live in thin adapter modules.

This is not stylistic. A pure core is what makes the evaluation harness able to
replay two years of history in seconds, and it is what makes a failed prediction
reproducible six months later from its packet hash alone.

Modules have one reason to change. When a file grows past roughly 400 lines it is
usually holding two responsibilities that want separating.

## Errors

Errors are explicit and specific. No bare `except`. No `except Exception` that
logs and continues. No falling back to a default value that makes a failure look
like a legitimate result.

The rule that matters most here: **a missing input is never silently treated as a
neutral one.** If price data for an event is unavailable, the event is marked
unresolvable and excluded from metrics. It does not become a zero return. Silent
substitution of defaults is how event studies acquire biases that survive every
test and invalidate every conclusion.

Exceptions carry enough context to diagnose without a reproduction: which
accession number, which timestamp, which arm.

## Tests

Tests assert real outcomes against independently-derived expectations. A test
that asserts a function returns whatever it currently returns documents a bug as
readily as it documents correct behavior.

- **Unit tests** cover pure logic with hand-computed expected values. Where a
  calculation has a closed form or a published reference, the test uses it.
- **Property-based tests** will cover the risk and sizing math once it exists,
  where the properties are clearer than any individual case: position size never
  exceeds the concentration cap, sizing is monotonic in conviction, a zero-
  conviction signal produces no position. No sizing code exists yet, so neither
  do those tests.
- **Timestamp-discipline tests** are their own category. Every retrieval path has
  a test asserting that data after the event's `as_of` is excluded. These guard
  the property the project's validity rests on.
- **Integration tests** are marked and excluded from the default run. They reach
  live SEC and market-data endpoints and an embedded database; a full day of
  ingestion takes minutes, which is why they are not run per pull request.
- **Evaluation tests** will run a frozen benchmark and fail CI on regression.
  Neither the benchmark nor that job exists yet.

Coverage is tracked but is not the target. Correct assertions over a critical
path beat broad coverage of trivial paths.

## Determinism

Every stochastic process records its seed alongside its output. Model training,
sampling, cross-validation splits, and the screening sample all log seeds to the
run record.

Any result that cannot be regenerated from committed code plus a recorded seed
plus a packet hash is not a result. This is the practical test applied before any
number goes into a report.

## Configuration

All configuration arrives through the environment, validated at startup by a
single `pydantic-settings` model. Invalid or missing configuration fails
immediately with a message naming the variable, never at the first use halfway
through a batch.

No secret is ever logged, included in a packet, or embedded in a prompt.

## Documentation in code

Docstrings state the contract: what the function guarantees, what units it works
in, what it raises, and any precondition a caller must satisfy. They do not
restate the signature.

```python
def abnormal_return(stock: PriceSeries, benchmark: PriceSeries, window: TradingWindow) -> float:
    """Return the stock's excess return over its benchmark across the window.

    The window opens at the first market open strictly after the event
    timestamp, which avoids attributing pre-event price action to the event.

    Returns a simple return difference in decimal form, not basis points.

    Raises:
        InsufficientPriceData: fewer than `window.length` sessions are
            available for either series. Callers mark the event unresolvable
            rather than substituting a default.
    """
```

Comments explain why, not what. A comment earns its place when the code is
correct for a reason a careful reader would not infer.

No commented-out code. No scaffolding left behind for a future that has not been
scheduled. Version control remembers deleted code; a repository full of dead
branches makes the live path harder to find.

## Dependencies

Pinned via `uv.lock`, which is committed. `pip-audit` runs in CI. New dependencies
need a reason that a standard-library or existing-dependency approach does not
satisfy, because each one is a supply-chain surface and a maintenance obligation.

## Commits and review

Commit messages use the imperative mood and explain why a change was made when
that is not obvious from the diff. Each commit leaves the repository in a working
state with CI green.

Every change reaches the default branch through a pull request. Branch
protection enforces four things, on administrators included:

1. The four CI checks pass.
2. The branch is current with the default branch.
3. History stays linear.
4. **Every review conversation is resolved.** An automated review runs on each
   pull request and comments inline; merging is blocked until each of those
   threads has been read and closed. Since a pull request cannot be approved by
   its own author, this is what a single-maintainer repository can enforce in
   place of a second approver, and it is the point where a person is required to
   engage with what the review found.

Changes fall into two categories, and the second is where that engagement
matters:

- **Mechanically verifiable** — dependency bumps, formatting, regenerated
  documentation, data refreshes. Automated review and passing CI are sufficient.
- **Judgment-bearing** — prompts, features, model selection, screening rules, and
  anything touching evaluation methodology. These require human review regardless
  of what any automated reviewer concluded.

The second category exists because this repository publishes measured claims. A
change to the feature pipeline that merges unreviewed makes last month's results
incomparable to this month's with no record of when the break occurred.
