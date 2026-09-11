# Corporate red flags

## Trigger

8-K filings carrying:

| Item | Event |
|---|---|
| 4.01 | Changes in the registrant's certifying accountant |
| 4.02 | Non-reliance on previously issued financial statements |
| 5.02 | Departure or election of directors and principal officers |

Low thousands of events per year, the smallest universe of the three strategies.

## Why this strategy exists

The signal here lives in prose rather than in fields. Whether an auditor was
dismissed or resigned, whether a departure was characterized as a resignation for
personal reasons or carried a disagreement disclosure, and the scope a
restatement covers are all expressed in narrative text with no standard
structure.

This makes it the natural test of whether language models hold a genuine
advantage. If they do, it should appear here and be absent in the insider
strategy, where the same information is tabular. The contrast between the two is
the most informative comparison the system can produce, and it is more
interesting than either strategy's standalone performance.

## Features

Largely extracted rather than computed:

- Auditor change direction, whether a disagreement was disclosed, and the
  relative standing of the outgoing and incoming firms
- Restatement scope, the periods affected, and the line items involved
- Departure role, stated circumstances, notice period, and whether a
  disagreement disclosure accompanies it
- Time since the issuer's last red-flag event of any kind

## Sample-size constraint

With low thousands of events annually, this strategy will have the widest
confidence intervals of the three, and per-item-code breakdowns will be thinner
still. Results are reported with intervals rather than point estimates, and no
claim is made from an item code with insufficient resolved events.

This is a known limitation rather than a defect. It is the price of studying the
event type where text actually matters.
