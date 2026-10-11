"""Rendering a measurement as something a reader can check.

A result is reported with the things needed to judge it, not just its headline:
how many observations it rests on, over how many days, how it was fitted, and
how far the estimate sits from zero relative to its own uncertainty. A single
number without those is not a finding.

The report is written as Markdown and committed, so that what was claimed at a
given commit stays inspectable — a result recomputed later against more data is
a new report, not an edit to the old one.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import date

from arbiter.arms.evaluate import Evaluation
from arbiter.evaluation.insider_type import TypeBreakdown
from arbiter.evaluation.metrics import MIN_EVENTS_PER_DAY, ICSummary
from arbiter.evaluation.size import SIZE_SESSIONS, BucketIC, SizeBreakdown

#: Width of the bar drawn beside each calibration band, in characters. A drawn
#: comparison is read at a glance where two columns of decimals are not.
_BAR_WIDTH = 24

#: A count-weighted residual below this is too small to report as a shift.
_MATERIAL_RESIDUAL = 0.005


def _bar(value: float, width: int = _BAR_WIDTH) -> str:
    """Draw a proportion as a fixed-width bar."""
    filled = max(0, min(width, round(value * width)))
    return "█" * filled + "·" * (width - filled)


def _signed(value: float, places: int = 4) -> str:
    """Format a number with its sign always shown, so zero is never ambiguous."""
    if math.isnan(value):
        return "n/a"
    return f"{value:+.{places}f}"


#: Brier skill this close to zero is reported as no difference from the base rate.
#: No standard error is computed for it, and a sign on a value this small would
#: read as a finding — "worse than a constant" under a strong discrimination
#: result — when it is the noise floor of a few hundred observations.
_NEGLIGIBLE_SKILL = 0.005


def _calibration_verdict(evaluation: Evaluation) -> str:
    """State what the forecasts' stated probabilities were worth.

    Reported beside discrimination rather than left in a table, because the two
    fail independently and the reader who stops at "no measurable skill" would
    otherwise take a forecast that is actively worse than a constant as merely
    uninformative.
    """
    skill = evaluation.brier_skill
    if math.isnan(skill):
        return ""
    fitted = (
        f"the up-rate each refit had observed, {evaluation.fitted_base_rate:.1%} on average"
    )
    if abs(skill) < _NEGLIGIBLE_SKILL:
        return (
            f" The stated probabilities are worth no more than forecasting {fitted} "
            f"for every issuer-day: a Brier skill of {skill:+.4f} is indistinguishable "
            "from it." + _hindsight_skill(evaluation) + _bias_direction(evaluation)
        )
    if skill > 0:
        return (
            f" The stated probabilities are worth slightly more than forecasting "
            f"{fitted} (Brier skill {skill:+.4f}), which is a weaker claim than "
            "discrimination and should be read as one." + _hindsight_skill(evaluation)
        )
    return (
        f" Calibration is worse than that: a Brier skill of {skill:+.4f} means "
        f"the stated probabilities were less useful than forecasting {fitted} "
        "for every issuer-day." + _hindsight_skill(evaluation) + _bias_direction(evaluation)
    )


def _hindsight_skill(evaluation: Evaluation) -> str:
    """State the skill against the scored period's own rate, named as hindsight.

    The rate observed by each refit is the comparison a forecaster could have
    made; the scored period's rate is stricter because it was not knowable when the
    forecasts were made. Both are stated so neither reading is left to infer.
    """
    skill = evaluation.brier_skill_scored_rate
    if math.isnan(skill):
        return ""
    return (
        f" Against the scored period's own rate of {evaluation.base_rate:.1%}, which "
        f"was not knowable in advance, the Brier skill is {skill:+.4f}."
    )


def _bias_direction(evaluation: Evaluation) -> str:
    """Describe which way the forecasts miss, measured rather than assumed.

    A negative Brier skill says only that the probabilities were worse than a
    constant; it carries no direction. Taking one from that sign alone stated
    the opposite of what the first published report showed.

    Both the size and the consistency of the miss are read from the reliability
    bands: the count-weighted residual says how far, and the per-band signs say
    whether the bands agree. Calling them contradictory from a small aggregate
    alone would repeat the original error one level down, because bands that all
    run slightly the same way also average to a small number.
    """
    counted = [band for band in evaluation.calibration if band.count]
    if not counted:
        return ""

    pooled = sum(band.count for band in counted)
    residuals = [(band.forecast - band.realised, band.count) for band in counted]
    weighted = sum(residual * count for residual, count in residuals) / pooled

    high = sum(1 for residual, _ in residuals if residual > 0)
    low = sum(1 for residual, _ in residuals if residual < 0)
    agree = not (high and low)

    if abs(weighted) >= _MATERIAL_RESIDUAL:
        direction = "high" if weighted > 0 else "low"
        # The direction is count-weighted and the tallies are not, so a few
        # heavy bands can outvote the majority; the opposing count is whichever
        # tally runs against the direction, not the smaller one.
        opposed = low if weighted > 0 else high
        consistency = (
            "in every populated band"
            if agree
            else (f"on balance, with {opposed} of {len(residuals)} bands running the other way")
        )
        return (
            f" The forecasts run {direction} by {abs(weighted):.3f} across the "
            f"reliability bands below, weighted by the issuer-days in each, {consistency}."
        )

    if agree:
        direction = "high" if weighted > 0 else "low"
        return (
            f" The forecasts run slightly {direction}, by {abs(weighted):.3f} — "
            "small, but in the same direction in every populated band below."
        )

    return (
        f" The bands below disagree in direction: {high} run high and {low} run "
        "low, so the miss is dispersion rather than a shift."
    )


def _overlap_note(ic: ICSummary) -> str:
    """How the standard error allowed for overlapping outcome windows, if it did."""
    if ic.overlap_lags == 0:
        return ""
    return (
        f", the larger of the error treating days as independent and one allowing "
        f"for each day's outcome window overlapping the next {ic.overlap_lags}"
    )


def _verdict(evaluation: Evaluation) -> str:
    """State plainly whether the measurement supports a claim of skill.

    Written as a sentence rather than left to the reader, because a table of
    statistics invites the most flattering reading of itself. Discrimination and
    calibration are both stated: a forecast can rank well while being badly
    calibrated, or rank at chance while being honest about its uncertainty, and
    reporting only the first lets the second failure pass unremarked.
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
            f"{ic.mean:+.4f} against a standard error of {ic.standard_error:.4f}"
            f"{_overlap_note(ic)} (t = {ic.t_statistic:+.2f}), which is "
            "indistinguishable from zero at "
            f"the conventional threshold. With {ic.days} scored days this is the "
            "expected outcome whether or not an edge exists; it is a statement "
            "about the sample, not evidence that no signal is there."
            + _calibration_verdict(evaluation)
        )
    direction = "positive" if ic.mean > 0 else "negative"
    return (
        f"**A {direction} information coefficient of {ic.mean:+.4f}** (standard "
        f"error {ic.standard_error:.4f}{_overlap_note(ic)}, t = {ic.t_statistic:+.2f}) across "
        f"{ic.days} days. This clears the conventional two-standard-error "
        "threshold, which is a weak bar: it is one test on one sample, and the "
        "estimate should be expected to shrink as more days are added."
        + _calibration_verdict(evaluation)
    )


def _unlabelled(evaluation: Evaluation) -> str:
    """Say how many scored-day events had no outcome, when any did."""
    if not evaluation.test_events_unlabelled:
        return ""
    return (
        f"; {evaluation.test_events_unlabelled:,} more could not be priced and are "
        "recorded as exclusions"
    )


def _group_row(
    label: str, ic: ICSummary | None, days: int, issuer_days: int, leading: Sequence[str] = ()
) -> str:
    """One group's row; a group that never reached the threshold is blank, not zero.

    `leading` are the cells between the label and the statistics.
    """
    cells = [label, *leading]
    if ic is None:
        cells += ["n/a", "n/a", "n/a"]
    else:
        cells += [
            _signed(ic.mean),
            f"{ic.standard_error:.4f}",
            _signed(ic.t_statistic, places=2),
        ]
    cells += [str(days), f"{issuer_days:,}"]
    return "| " + " | ".join(cells) + " |"


def _size_row(bucket: BucketIC) -> str:
    """One size bucket's row."""
    median = "n/a" if bucket.median_size is None else f"{bucket.median_size / 1e6:,.2f}"
    return _group_row(bucket.label, bucket.ic, bucket.days, bucket.issuer_days, [median])


def _crossing_alone(ics: Iterable[ICSummary | None], headline_significant: bool) -> bool:
    """Whether some group reaches |t| of 2 while the headline does not."""
    return any(ic is not None and ic.is_significant for ic in ics) and not headline_significant


def _size_caveat(size: SizeBreakdown, headline_significant: bool) -> list[str]:
    """What the table cannot show, sharper when one row crosses |t| of 2 alone."""
    lines = [
        "These are three subsets of the headline. The table shows where any signal would",
        "sit; it does not test whether the buckets differ from one another, and a",
        "difference between two rows is not evidence that size matters.",
    ]
    if _crossing_alone((b.ic for b in size.buckets), headline_significant):
        lines += [
            "A row reaching |t| of 2 while the headline does not is one of three subsets",
            "crossing that bar, which is not unusual; the buckets were not tested against",
            "each other, and such a row is not a finding.",
        ]
    return lines


def _size_section(
    size: SizeBreakdown | None, omitted: str | None, headline_significant: bool
) -> list[str]:
    """The by-size section, or the one line saying why it is absent."""
    lines = ["## By issuer size", ""]
    if size is None:
        return [*lines, omitted or "Not computed.", ""]
    in_rows = sum(bucket.issuer_days for bucket in size.buckets)
    lines += [
        "Rank correlation within terciles of issuer size, drawn afresh each day. Size is",
        f"the median daily dollar volume over the issuer's last {SIZE_SESSIONS} sessions in",
        "its evidence packet, so it uses only bars that had closed before the filing.",
        f"A bucket-day counts only with at least {MIN_EVENTS_PER_DAY} issuers in that bucket,",
        "the headline's minimum applied to each third, so a day needs about three times as",
        "many sized issuers as the headline asks. The size column is the median over the",
        "issuer-days in each row, in millions of dollars of daily volume.",
        "",
        "| Bucket | Median daily volume, $M | Mean | SE | t | Days | Issuer-days |",
        "|---|---|---|---|---|---|---|",
        *(_size_row(bucket) for bucket in size.buckets),
        "",
        f"Of {size.scored_issuer_days:,} scored issuer-days, {in_rows:,} are in the rows;",
        f"{size.dropped:,} had a size but fell in a bucket-day below the minimum, and",
        f"{size.unbucketed:,} had no size: no packet, or fewer than {SIZE_SESSIONS} bars.",
        "",
        *_size_caveat(size, headline_significant),
        "",
    ]
    return lines


def _history_lines(breakdown: TypeBreakdown) -> list[str]:
    """Say which quarters of history were loaded, when known."""
    if not breakdown.history_first:
        return []
    return [
        f"The history is Forms 4 and 4/A from the {breakdown.history_first} to the "
        f"{breakdown.history_last}",
        f"data sets; trades filed after {breakdown.history_last} are absent, and an event",
        "needs three prior calendar years of it to be classified.",
    ]


def _skipped_lines(breakdown: TypeBreakdown) -> list[str]:
    """Report rows excluded from the history, only when there were any."""
    parts: list[str] = []
    if breakdown.unparseable_rows:
        parts.append(
            f"{breakdown.unparseable_rows:,} history rows had an unparseable date or CIK"
        )
    if breakdown.future_dated_trades:
        parts.append(
            f"{breakdown.future_dated_trades:,} history trades were dated after their filing"
        )
    if not parts:
        return []
    return [f"Excluded from the history: {' and '.join(parts)}."]


def _type_caveat(breakdown: TypeBreakdown, headline_significant: bool) -> list[str]:
    """What the table cannot show, sharper when one row crosses |t| of 2 alone."""
    lines = [
        "These are two subsets of the headline. The table shows where any signal would",
        "sit; it does not test whether the rows differ from one another, and a",
        "difference between them is not evidence that insider type matters.",
    ]
    if _crossing_alone((r.ic for r in breakdown.rows), headline_significant):
        lines += [
            "A row reaching |t| of 2 while the headline does not is one of two subsets",
            "crossing that bar; the rows were not tested against each other, and such a",
            "row is not a finding.",
        ]
    return lines


def _type_section(
    breakdown: TypeBreakdown | None, omitted: str | None, headline_significant: bool
) -> list[str]:
    """The by-insider-type section, or the one line saying why it is absent."""
    lines = ["## By insider type", ""]
    if breakdown is None:
        return [*lines, omitted or "Not computed.", ""]
    in_rows = sum(row.issuer_days for row in breakdown.rows)
    lines += [
        "Rank correlation within routine and opportunistic insiders, after Cohen, Malloy",
        'and Pomorski (2012), "Decoding Inside Information", Journal of Finance. As there,',
        "the trader is classified, not the trade: for an event in year Y an insider is",
        "routine if there is a calendar month in which they made open-market trades in each",
        "of the three prior years, opportunistic if they traded in each of those years but in",
        "no common month, and unclassified otherwise. An event with several owners is",
        "classified only if they agree, and an issuer-day only if its classified events agree.",
        *_history_lines(breakdown),
        "",
        "Departures from the paper: the insider is the owner at one issuer, not the person.",
        "History counts open-market purchases and sales in any non-derivative security,",
        "including trades under 10b5-1 plans, which the scored events exclude. The event's",
        "year is that of its acceptance, while history years come from transaction dates",
        "because events store no transaction date, so a trade made in late December and",
        "accepted in January can sit a year early.",
        "",
        f"A type-day counts only with at least {MIN_EVENTS_PER_DAY} issuers of that type.",
        "",
        "| Type | Mean | SE | t | Days | Issuer-days |",
        "|---|---|---|---|---|---|",
        *(
            _group_row(r.label.capitalize(), r.ic, r.days, r.issuer_days)
            for r in breakdown.rows
        ),
        "",
        f"Of {breakdown.scored_issuer_days:,} scored issuer-days, {in_rows:,} are in the rows;",
        f"{breakdown.unclassified:,} were unclassified, {breakdown.mixed:,} were mixed (their",
        "events disagreed), and",
        f"{breakdown.dropped:,} were classified but fell in a type-day below the minimum.",
        *_skipped_lines(breakdown),
        "",
        *_type_caveat(breakdown, headline_significant),
        "",
    ]
    return lines


def render_baseline_report(
    evaluation: Evaluation,
    generated_on: date,
    size: SizeBreakdown | None = None,
    size_omitted: str | None = None,
    insider_type: TypeBreakdown | None = None,
    insider_type_omitted: str | None = None,
) -> str:
    """Render an evaluation as a Markdown report.

    `generated_on` is passed rather than read from the clock so the same
    evaluation renders identically whenever it is regenerated, which is what
    makes a committed report diffable. `size` is the breakdown by issuer size;
    without it the section says why, in `size_omitted`. `insider_type` and
    `insider_type_omitted` do the same for the routine/opportunistic section.
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
        f"| Initial fitting window | {len(evaluation.initial_days)} days, "
        f"{evaluation.initial_days[0]} to {evaluation.initial_days[-1]}; never scored |",
        f"| Refits | {len(evaluation.folds)}, one per block of up to "
        f"{max(len(fold.scored_days) for fold in evaluation.folds)} scored days |",
        f"| Scored on | {len(evaluation.test_days)} days, "
        f"{evaluation.test_events:,} labelled events{_unlabelled(evaluation)} |",
        # These two counts differ whenever a day carried too few distinct
        # issuers to rank. Reporting only one of them would leave the day count
        # in the verdict contradicting the day count in this table.
        f"| Days contributing a rank correlation | {ic.days} of {len(evaluation.test_days)} |",
        f"| Scored observations | {evaluation.test_issuer_days:,} issuer-days |",
        f"| Scoring period | {evaluation.test_days[0]} to {evaluation.test_days[-1]} |",
        f"| First refit | fitted on {len(evaluation.folds[0].fit_days)} days, "
        f"{evaluation.folds[0].train_events:,} events: only the initial-window events "
        "whose outcomes had closed |",
        f"| Final refit | fitted on {len(evaluation.folds[-1].fit_days)} days, "
        f"{evaluation.train_events:,} events |",
        f"| Up-rate, refits | {evaluation.fitted_base_rate:.1%} of fitted issuer-days had a "
        "positive abnormal return, averaged over scored issuer-days by the refit that "
        "scored them |",
        f"| Up-rate, scoring period | {evaluation.base_rate:.1%} of issuer-days |",
        f"| Brier skill | {_signed(evaluation.brier_skill)} against forecasting "
        "the up-rate of the refit that made each forecast |",
        f"| Brier skill, hindsight reference | {_signed(evaluation.brier_skill_scored_rate)} "
        "against forecasting the scoring period's own rate, not knowable in advance |",
        "",
        "Evaluation is walk-forward: each block of scored days is forecast by a model",
        "fitted only on events whose outcomes had closed before the block's first day.",
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
        f"| Standard error | {ic.standard_error:.4f}{_overlap_note(ic)} |",
        f"| t | {_signed(ic.t_statistic, places=2)} |",
        "",
        *_size_section(size, size_omitted, ic.is_significant),
        *_type_section(insider_type, insider_type_omitted, ic.is_significant),
        "## Calibration",
        "",
        "Whether a stated probability means what it says. A well-calibrated forecast",
        "has the realised column tracking the forecast column down the table.",
        f"Pooled over all {evaluation.pooled_issuer_days:,} scored issuer-days, including",
        "any on days with too few issuers to rank.",
        "",
        "| Forecast band | Mean forecast | Realised | Issuer-days | |",
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
        "From the final refit, which saw the most data; earlier refits' weights are",
        "not shown. On the standardised scale, so magnitudes are comparable across",
        "features.",
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
