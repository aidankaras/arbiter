"""Rendering a measurement as something a reader can check.

A result is reported with the things needed to judge it, not just its headline:
how many observations it rests on, over how many days, what the split was, and
how far the estimate sits from zero relative to its own uncertainty. A single
number without those is not a finding.

The report is written as Markdown and committed, so that what was claimed at a
given commit stays inspectable — a result recomputed later against more data is
a new report, not an edit to the old one.
"""

from __future__ import annotations

import math
from datetime import date

from arbiter.arms.evaluate import Evaluation

#: Width of the bar drawn beside each calibration band, in characters. A drawn
#: comparison is read at a glance where two columns of decimals are not.
_BAR_WIDTH = 24


def _bar(value: float, width: int = _BAR_WIDTH) -> str:
    """Draw a proportion as a fixed-width bar."""
    filled = max(0, min(width, round(value * width)))
    return "█" * filled + "·" * (width - filled)


def _signed(value: float, places: int = 4) -> str:
    """Format a number with its sign always shown, so zero is never ambiguous."""
    if math.isnan(value):
        return "n/a"
    return f"{value:+.{places}f}"


def _verdict(evaluation: Evaluation) -> str:
    """State plainly whether the measurement supports a claim of skill.

    Written as a sentence rather than left to the reader, because a table of
    statistics invites the most flattering reading of itself.
    """
    ic = evaluation.ic
    if math.isnan(ic.t_statistic):
        return (
            "**No conclusion.** The information coefficient has no spread to test "
            "against, so nothing here distinguishes the arm from chance."
        )
    if not ic.is_significant:
        return (
            f"**No measurable skill.** The mean information coefficient is "
            f"{ic.mean:+.4f} against a standard error of {ic.standard_error:.4f} "
            f"(t = {ic.t_statistic:+.2f}), which is indistinguishable from zero at "
            f"the conventional threshold. With {ic.days} scored days this is the "
            "expected outcome whether or not an edge exists; it is a statement "
            "about the sample, not evidence that no signal is there."
        )
    direction = "positive" if ic.mean > 0 else "negative"
    return (
        f"**A {direction} information coefficient of {ic.mean:+.4f}** (standard "
        f"error {ic.standard_error:.4f}, t = {ic.t_statistic:+.2f}) across "
        f"{ic.days} days. This clears the conventional two-standard-error "
        "threshold, which is a weak bar: it is one test on one sample, and the "
        "estimate should be expected to shrink as more days are added."
    )


def render_baseline_report(evaluation: Evaluation, generated_on: date) -> str:
    """Render an evaluation as a Markdown report.

    `generated_on` is passed rather than read from the clock so the same
    evaluation renders identically whenever it is regenerated, which is what
    makes a committed report diffable.
    """
    ic = evaluation.ic
    lines: list[str] = [
        "# Baseline arm: insider transactions",
        "",
        "Logistic regression on filing characteristics, forecasting the sign of the",
        "five-session abnormal return. This is the control the language-model arms",
        "are measured against, not a proposed strategy.",
        "",
        f"Generated {generated_on.isoformat()} from the event store.",
        "",
        "## Result",
        "",
        _verdict(evaluation),
        "",
        "## What this rests on",
        "",
        "| | |",
        "|---|---|",
        f"| Fitted on | {len(evaluation.train_days)} days, "
        f"{evaluation.train_events:,} events |",
        f"| Scored on | {len(evaluation.test_days)} days, {evaluation.test_events:,} events |",
        # These two counts differ whenever a day carried too few distinct
        # issuers to rank. Reporting only one of them would leave the day count
        # in the verdict contradicting the day count in this table.
        f"| Days contributing a rank correlation | {ic.days} of {len(evaluation.test_days)} |",
        f"| Scored observations | {evaluation.test_issuer_days:,} issuer-days |",
        f"| Fitting period | {evaluation.train_days[0]} to {evaluation.train_days[-1]} |",
        f"| Scoring period | {evaluation.test_days[0]} to {evaluation.test_days[-1]} |",
        f"| Base rate | {evaluation.base_rate:.1%} of events had a positive abnormal return |",
        f"| Brier skill | {_signed(evaluation.brier_skill)} against forecasting "
        "the base rate |",
        "",
        "The split is chronological: every scored day falls after every fitted day.",
        "Discrimination is credited once per issuer per day, because several insiders",
        "at one company on one day resolve to a single outcome.",
        "",
        "## Information coefficient by day",
        "",
        "| Day | Rank correlation |",
        "|---|---|",
    ]

    for day, value in sorted(evaluation.daily_ic.items()):
        lines.append(f"| {day.isoformat()} | {_signed(value)} |")

    lines += [
        f"| **Mean** | **{_signed(ic.mean)}** |",
        f"| Standard error | {ic.standard_error:.4f} |",
        f"| t | {_signed(ic.t_statistic, places=2)} |",
        "",
        "## Calibration",
        "",
        "Whether a stated probability means what it says. A well-calibrated forecast",
        "has the realised column tracking the forecast column down the table.",
        "",
        "| Forecast band | Mean forecast | Realised | Events | |",
        "|---|---|---|---|---|",
    ]

    for band in evaluation.calibration:
        lines.append(
            f"| {band.lower:.1f} to {band.upper:.1f} | {band.forecast:.3f} | "
            f"{band.realised:.3f} | {band.count:,} | `{_bar(band.realised)}` |"
        )

    lines += [
        "",
        "## Fitted weights",
        "",
        "On the standardised scale, so magnitudes are comparable across features.",
        "A weight is what the model leaned on, not evidence that the characteristic",
        "predicts anything on its own.",
        "",
        "| Feature | Weight |",
        "|---|---|",
    ]

    for name, weight in sorted(evaluation.coefficients.items(), key=lambda item: -abs(item[1])):
        lines.append(f"| `{name}` | {_signed(weight)} |")

    lines += [
        "",
        "## Reading this honestly",
        "",
        "- An information coefficient measures ranking, not profit. Nothing here",
        "  accounts for transaction costs, market impact, or borrow availability.",
        "- The scored sample is one contiguous period. A result from one regime is",
        "  not evidence about another.",
        "- Insider-trading returns documented in the literature have decayed",
        "  substantially since publication, so a weak or absent signal here is the",
        "  expected finding rather than a surprising one.",
        "",
    ]
    return "\n".join(lines)
