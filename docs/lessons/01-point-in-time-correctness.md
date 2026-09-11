# 01 — Point-in-time correctness

**Subsystems:** `packets`, `retrieval`
**Status:** design settled, implementation pending

## The concept

A forecasting system is only measuring forecasting if, at the moment it makes a
prediction, it has access to nothing that was unavailable at that moment in the
real world. Violating this is called lookahead bias, and its defining property is
that it inflates measured performance while producing no error, no failing test,
and no visible symptom.

The reason it deserves a dedicated note is that the obvious precautions are not
the ones that catch it. Using timestamped source data is necessary and nowhere
near sufficient.

## Where it actually enters this system

Four distinct paths, in rough order of how easily each is missed.

**Comparable retrieval.** The system retrieves similar historical events along
with how they resolved, so an approach can reason from precedent. Retrieving a
case that had not yet resolved at the time of the event being predicted inserts
a future outcome into a past decision. This is the subtlest path because the
retrieval itself looks correct: the comparable event is genuinely older than the
event being predicted. It is the comparable's *resolution*, not its occurrence,
that must precede the prediction.

**Entry timing.** Filings frequently arrive after the market close. Measuring
return from the same session's close would credit the strategy with a full
session of price movement that preceded the information becoming public.

**Restated fundamentals.** Financial data vendors and XBRL company facts reflect
restatements. Querying a company's reported earnings for Q2 2024 today may return
a figure that was revised in 2025. The figure available on the announcement date
is the only one a forecaster could have used.

**Model training data.** A language model trained on data covering the backtest
period may recall how a specific event resolved rather than inferring it. This
one cannot be engineered away, because the knowledge is already in the weights.

## The decision

Enforce the first three structurally, and measure the fourth rather than
attempting to remove it.

Every event carries an `as_of` timestamp taken from the filing's EDGAR
acceptance datetime, which is when the information became public. That timestamp
propagates into the evidence packet and into every query the packet's
construction makes. Comparable retrieval filters on resolution time in the query
itself. Return windows open at the first market open strictly after `as_of`.

For the fourth, backtest and forward results are reported separately and always
labeled. Because labels resolve in five trading days, out-of-sample evidence
accumulates continuously, and the gap between backtested and forward performance
becomes a measurement of memorization advantage rather than a contaminant.

## Alternatives considered

**Trusting the prompt.** Instructing an analyst not to use knowledge it should
not have. Rejected: it is unverifiable, it fails silently, and it does nothing
about retrieval, which is where the leak actually originates.

**A global as-of filter applied at the session or connection level.** Attractive
because it is one place to get right. Rejected because the system legitimately
needs unfiltered access in other contexts, including label resolution and
evaluation, and a filter that must be selectively disabled is a filter that will
be disabled in the wrong place.

**Entity redaction and date shifting in the packet, to defeat model
memorization.** This was seriously considered as the project's centerpiece before
the event universe changed. It is a real technique and the published literature
uses it. Rejected here because five-day label resolution makes genuine forward
testing available immediately, and forward testing is strictly better evidence
than any decontamination proxy. Redaction remains available as a secondary check
if the forward sample stays small for longer than expected.

## How it is implemented

The `as_of` timestamp is a required field on the packet, not an optional
parameter with a default. Retrieval functions take it as a positional argument,
so omitting it is a type error rather than a silently permissive query.

Every retrieval path carries a test that inserts a comparable resolving after the
event timestamp and asserts it is excluded. These tests are treated as
correctness tests for the project's central claim, not as coverage.

Packets are content-hashed and every prediction stores the hash, so if packet
construction changes, results produced before and after are distinguishable
rather than silently mixed.

## Questions a reviewer would ask

**"How do you know your backtest isn't contaminated?"** For the non-LLM arms,
because every input is filtered on `as_of` in the query and the filter is tested.
For the LLM arms, it is contaminated, that is stated rather than claimed
otherwise, and the size of the effect is measured by the gap against forward
results.

**"Why acceptance time rather than filing date?"** Filing date is a calendar day
and does not distinguish a filing that landed at 9:00 from one that landed at
19:00. The distinction determines which session can be traded.

**"What happens when price data is missing for an event?"** The event is marked
unresolvable and excluded from metrics. It is never assigned a zero return,
because substituting a neutral value for a missing one biases results toward
whatever the substitute is.

**"Your comparables include past predictions and outcomes. Isn't that circular?"**
Only if a comparable's outcome postdates the event being predicted, which the
as-of filter prevents. Retrieving genuinely resolved history is what a human
analyst does, and it is the mechanism that lets an approach adapt without
retraining.
