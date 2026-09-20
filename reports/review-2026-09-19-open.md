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
