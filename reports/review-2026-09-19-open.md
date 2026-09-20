# Open review findings, 2026-09-19

Three review passes ran against the evidence-packet branch. The critical and
high findings are fixed and pinned; what remains is recorded here rather than
left in a transcript, because a finding that outlives the session it was made in
is only actionable if it is written down.

Each entry states the defect, what it would corrupt, and the demonstration that
established it.

## 1. `MarketSummary` renders three different outcomes as `None`

`src/arbiter/packets/build.py`

The docstring says each statistic is `None` "when the window holds too little
history". It is also `None` for a non-positive close and for a swallowed
`InvalidOperation`. A 22-session window containing one zero close reports
`realised_volatility_21d=None` — indistinguishable from a stock with ten
sessions of history.

Worse, the two statistics disagree on the same input: `_trailing_return` guards
only the first close of its window, so a zero elsewhere yields a computed return
of `0` beside an absent volatility. A packet then asserts the stock did not move.

A zero or negative close is bad data, not thin data. It should raise and be
recorded as an exclusion, the way the pipeline now records its other two
reasons. Vendor defects cluster by symbol and by day, so this is an error
correlated with the data.

## 2. `market._decimal` has no finiteness check, while `insider._decimal` does

`src/arbiter/ingestion/market.py`

`json.loads` accepts bare `NaN` and `Infinity` literals, so a price feed can
hand `parse_bars` a non-finite close today. `insider.py` guards this explicitly
and says why; the market parser does not. With the guard added, the
`InvalidOperation` handler in `build.py` becomes unreachable and can be deleted
rather than left standing as an apparently-defensive `return None`.

## 3. `_volume_percentile` counts ties at or below, inflating illiquid names

`src/arbiter/packets/build.py`

Every session tied with the latest counts as at-or-below, including the latest
itself, so a tie pushes the rank up by the whole tie group. An illiquid issuer
trading its usual 100 shares reports a 97th-percentile session. Tie frequency is
a function of liquidity, so the bias concentrates in small caps — where insider
signals are strongest and where the comparison is most likely to be decided.

The mid-rank convention, `(below + ties / 2) / n` with the latest session
excluded from its own comparison set, returns 0.5 for a flat series and matches
the meaning the docstring already claims.

## 4. `content_hash` depends on the process-global decimal context

`src/arbiter/packets/build.py` feeding `src/arbiter/packets/hashing.py`

`MarketSummary` values come from `Decimal` division, `.ln()` and `.sqrt()`, all
of which read the thread-local context. At precision 28 and precision 20 the
same series produces different values and therefore different packet hashes.
`hashing.py` promises the digest depends on the content "and nothing else"; it
currently also depends on whatever last touched `getcontext()`.

The module already refuses a `float` and a naive `datetime` for this class of
hazard. Compute the summary inside `decimal.localcontext()` at a declared
precision and quantize each statistic before it reaches the packet — which also
stops a 28-digit trailing return being presented to a model as meaningful.

## 5. `to_session_bar`'s docstring explains the mechanism wrongly

`src/arbiter/packets/build.py`

It says a bar stamped at 05:00 UTC "belongs to the previous Eastern day for part
of the year". It never does: 05:00Z is 00:00 EST or 01:00 EDT, the same Eastern
date all year. The conversion is correct and its justification is false. The
genuinely dangerous convention is a 00:00Z stamp, which would date every bar to
the prior session — and nothing pins which convention the vendor uses.

## 6. Refused symbols are indistinguishable from symbols that did not trade

`src/arbiter/ingestion/market.py`

`daily_bars_tolerating_gaps` computes the refused-symbol list and discards it,
so the pipeline's "no price history for X" cannot tell a symbol the service
refused from one that simply did not trade. That is exactly the distinction
`_bars_isolating_rejections` exists to preserve. Pre-existing, not introduced by
the packet work.

## 7. `write_packets` leaks its temporary file on failure

`src/arbiter/packets/store.py`

`NamedTemporaryFile(delete=False)` with no cleanup, where the sibling event
store wraps the identical pattern in `try/except BaseException: unlink; raise`.
A failure mid-write leaves a `.tmp` file in the partition directory. No false
positives, since `partition_exists` matches an exact name.

## 8. Collapsed duplicate rows leave no count to compare against

`src/arbiter/events/insider.py`

The joint-owner dedup removes rows and records no rows-in versus events-out
figure. The duplication it fixes was originally found by exactly such a
comparison — a day's packets yielding fewer distinct hashes than packets. If the
identity key is ever too loose, the resulting undercount would be invisible to
the only technique that has caught this class of defect here.

---

# Test-coverage findings, 2026-09-19

A third review pass mutated the branch systematically and found nineteen
surviving mutations, each verified against a control mutation confirmed to kill
three tests. Four are now closed (the per-item isolation handler, the exclusion
record and its negative control, and the truncate-before-emptiness fix). The
rest are recorded here in the reviewer's own priority order.

A methodology note worth keeping: mutating in a *copied* tree whose `.venv` came
with it produces a table in which everything survives, because `.venv/bin/pytest`
carries an absolute shebang and runs the original repo's interpreter. Mutate in
place with `scripts/mutate.sh`, or `rm -rf .venv && uv sync` in the copy first.

## Boundaries that are not pinned

1. **The close instant inside the packet validator** (`schema.py`). The same
   boundary `closed_bars` enforces, enforced a second time, with no packet test
   whose bar closes exactly at `as_of`. Loosening `>` to `>=` changes nothing.
   The comparable branch three lines below has precisely this test.
2. **The value threshold** (`insider.py`). Tests use 101,430 and 900 against a
   50,000 floor, so nothing sits at the line — `>=` to `>` survives, and so does
   *halving the threshold*. This constant defines the studied population of
   every published comparison and `CLAUDE.md` calls it a reviewed change rather
   than a tuning knob; nothing would fail if it moved.
3. **The 21-session volatility minimum and the 63-session percentile minimum**
   (`build.py`). Only one side of each is tested. A volatility computed over 20
   returns is reported as 21-session, and the shortfall concentrates at the
   start of every issuer's history.
4. **The percentile's window anchoring** (`build.py`). Every test passes exactly
   63 bars, so `bars[-63:]` and `bars[:63]` are indistinguishable, as are
   `len(window)` and `len(bars)`. With a realistic longer series the first ranks
   the latest session against the oldest quarter.
5. **The per-event window start** (`pipeline.py`). `>= window_start` to `>`
   survives; the identity test compares two hashes that move together under it.

## Tests that cannot fail

6. **Two Eastern-conversion tests are blind.** `test_a_bar_stamped_before_the_
   eastern_day_belongs_to_the_eastern_session` uses 04:00 UTC — exactly 00:00
   EDT — so the UTC and Eastern dates agree and the conversion is never
   exercised. Every bar in `test_bar_availability.py` is stamped 05:00 UTC, the
   same Eastern date all year. Dropping `.astimezone(SEC_TIMEZONE)` from either
   `to_session_bar` or `closed_bars` leaves the suite green. These stand in for
   a whole-calendar shift.
7. **`test_packets_are_built_without_consulting_labels`** asserts a label
   directory does not exist, which the fixture guarantees. The property it names
   — a packet is built for an event whose outcome cannot be measured yet — is
   untested.
8. **Parts of the contract sweep restate the filter.** Asserting `value_usd is
   not None` and `value_usd >= THRESHOLD` over `is_candidate`'s own output
   restates its implementation.

## Identity and dedupe

9. **The dedupe key can shed `Price` or `Remaining Shares` undetected**, because
   one fixture's duplicate rows are identical in all seven fields and the
   other's differ in shares, price and value together. A test with two rows
   differing only in `Remaining Shares` would reach the part the fixtures
   cannot. The failure direction loses real observations, and correlates with
   insiders who trade more than once a day.
10. **`"Date"` being a required column is unpinned**, though the equivalent
    guard for `"Shares"` is tested.
11. **`sorted()` inside `_canonical` is redundant** — `json.dumps(sort_keys=True)`
    already sorts every level — so `test_key_order_is_not_content` passes for a
    reason other than the one its docstring gives. Not a defect; a comment.
12. **The store's atomic write is unasserted.** Replacing `Path.replace` with a
    non-atomic copy leaves the suite green, while the docstring claims an
    interrupted run leaves the previous partition intact.
