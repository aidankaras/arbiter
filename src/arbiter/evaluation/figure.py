"""Drawing a measurement as a figure a reader can take in at a glance.

The figure is rendered from the same `Evaluation` as the Markdown report and
written beside it, so it can never show a different result from the one the
report states. It is plain SVG built without a plotting library: the output is
text, diffable in a commit, and identical every time it is regenerated.

It carries its own light background rather than inheriting the page's, because
an image embedded in a README is shown on both light and dark themes and cannot
read which one it is on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from xml.sax.saxutils import escape

from arbiter.arms.evaluate import Evaluation

_WIDTH = 880
_HEIGHT = 340

_SURFACE = "#fcfcfb"
_BORDER = "#e1e0d9"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_SERIES = "#2a78d6"
_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

#: Bars are capped well short of their slot, so the gaps between days read as air.
_BAR_WIDTH = 18
#: The rounded radius at a bar's data end; the end at the baseline stays square.
_BAR_RADIUS = 4


def _number(value: float) -> str:
    """Format a coordinate compactly and identically on every platform."""
    rounded = round(value, 1)
    return f"{rounded:g}" if rounded != int(rounded) else str(int(rounded))


def _text(
    x: float,
    y: float,
    content: str,
    *,
    size: int = 11,
    fill: str = _INK_MUTED,
    anchor: str = "start",
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{_number(x)}" y="{_number(y)}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{escape(content)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, stroke: str, width: float = 1) -> str:
    return (
        f'<line x1="{_number(x1)}" y1="{_number(y1)}" x2="{_number(x2)}" y2="{_number(y2)}" '
        f'stroke="{stroke}" stroke-width="{_number(width)}"/>'
    )


def _ticks(low: float, high: float, step: float) -> list[float]:
    """Return the multiples of `step` from `low` to `high`, inclusive."""
    first = math.ceil(round(low / step, 9))
    last = math.floor(round(high / step, 9))
    return [round(index * step, 10) for index in range(first, last + 1)]


def _floor_tenth(value: float) -> float:
    """Round down to a tenth, treating float noise such as 0.6000000001 as 0.6."""
    return math.floor(round(value * 10, 9)) / 10


def _ceil_tenth(value: float) -> float:
    """Round up to a tenth, treating float noise such as 0.6000000001 as 0.6."""
    return math.ceil(round(value * 10, 9)) / 10


def _bar(x: float, zero: float, top: float) -> str:
    """Draw a bar from the baseline to `top`, rounded only at its data end."""
    height = abs(zero - top)
    radius = min(_BAR_RADIUS, height / 2, _BAR_WIDTH / 2)
    left, right = x - _BAR_WIDTH / 2, x + _BAR_WIDTH / 2
    # The data end is above the baseline for a positive value and below it for
    # a negative one; the arc sweeps toward the data end either way.
    direction = -1 if top < zero else 1
    end = zero + direction * height
    shoulder = end - direction * radius
    sweep = 1 if direction < 0 else 0
    arc = f"A{_number(radius)},{_number(radius)} 0 0 {sweep}"
    path = (
        f"M{_number(left)},{_number(zero)} V{_number(shoulder)} "
        f"{arc} {_number(left + radius)},{_number(end)} H{_number(right - radius)} "
        f"{arc} {_number(right)},{_number(shoulder)} V{_number(zero)} Z"
    )
    return f'<path d="{path}" fill="{_SERIES}"/>'


def _ic_panel(evaluation: Evaluation, left: float, top: float, width: float) -> list[str]:
    """Daily rank correlations as bars, with the mean and its ±2 SE band."""
    plot_top, plot_bottom = top + 40, top + 250
    plot_left, plot_right = left + 44, left + width - 92
    ic = evaluation.ic
    days = sorted(evaluation.daily_ic)
    values = [evaluation.daily_ic[day] for day in days]

    band = (ic.mean - 2 * ic.standard_error, ic.mean + 2 * ic.standard_error)
    low = _floor_tenth(min(0.0, *values, band[0]))
    high = _ceil_tenth(max(0.0, *values, band[1]))

    def y(value: float) -> float:
        return plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)

    parts = [
        _text(
            left,
            top + 14,
            "Information coefficient by scored day",
            size=13,
            fill=_INK,
            weight="600",
        ),
        _text(
            left,
            top + 30,
            f"Rank correlation across each day's issuers, {len(days)} days",
            size=11,
            fill=_INK_SECONDARY,
        ),
    ]
    for tick in _ticks(low, high, 0.1):
        parts.append(_line(plot_left, y(tick), plot_right, y(tick), _GRID))
        parts.append(
            _text(plot_left - 8, y(tick) + 4, f"{tick:+.1f}" if tick else "0", anchor="end")
        )

    band_top, band_height = y(band[1]), y(band[0]) - y(band[1])
    parts.append(
        f'<rect x="{_number(plot_left)}" y="{_number(band_top)}" '
        f'width="{_number(plot_right - plot_left)}" height="{_number(band_height)}" '
        f'fill="{_INK_SECONDARY}" fill-opacity="0.1"/>'
    )

    slot = (plot_right - plot_left) / len(days)
    for index, (day, value) in enumerate(zip(days, values, strict=True)):
        x = plot_left + slot * (index + 0.5)
        parts.append(
            f"<g><title>{day.isoformat()}: {value:+.4f}</title>{_bar(x, y(0), y(value))}</g>"
        )
        # Every other day is labelled: eleven dates do not fit at this width,
        # and the first, middle and last are enough to place the period.
        if index % 2 == 0:
            parts.append(_text(x, plot_bottom + 18, f"{day:%b} {day.day}", anchor="middle"))

    parts.append(_line(plot_left, y(0), plot_right, y(0), _BASELINE))
    parts.append(_line(plot_left, y(ic.mean), plot_right, y(ic.mean), _INK, 1.5))
    parts.append(
        _text(plot_right + 8, y(ic.mean) + 4, f"mean {ic.mean:+.3f}", fill=_INK, weight="600")
    )
    parts.append(_text(plot_right + 8, y(band[1]) + 4, "+2 SE", fill=_INK_SECONDARY))
    parts.append(_text(plot_right + 8, y(band[0]) + 4, "-2 SE", fill=_INK_SECONDARY))
    return parts


def _calibration_panel(evaluation: Evaluation, left: float, top: float) -> list[str]:
    """Mean forecast against realised share per band, beside the diagonal."""
    bands = [band for band in evaluation.calibration if band.count]
    size = 210
    plot_left, plot_top = left + 44, top + 40
    plot_right, plot_bottom = plot_left + size, plot_top + size

    low = _floor_tenth(min(band.lower for band in bands))
    high = _ceil_tenth(max(band.upper for band in bands))

    def x(value: float) -> float:
        return plot_left + (value - low) / (high - low) * size

    def y(value: float) -> float:
        return plot_bottom - (value - low) / (high - low) * size

    parts = [
        _text(left, top + 14, "Calibration by forecast band", size=13, fill=_INK, weight="600"),
        _text(
            left,
            top + 30,
            f"{evaluation.pooled_issuer_days:,} scored issuer-days",
            size=11,
            fill=_INK_SECONDARY,
        ),
    ]
    for tick in _ticks(low, high, 0.1):
        parts.append(_line(x(tick), plot_top, x(tick), plot_bottom, _GRID))
        parts.append(_line(plot_left, y(tick), plot_right, y(tick), _GRID))
        parts.append(_text(x(tick), plot_bottom + 18, f"{tick:.1f}", anchor="middle"))
        parts.append(_text(plot_left - 8, y(tick) + 4, f"{tick:.1f}", anchor="end"))
    parts.append(_line(x(low), y(low), x(high), y(high), _BASELINE, 1.5))
    parts.append(
        _text(plot_left + size / 2, plot_bottom + 36, "Mean forecast", anchor="middle")
    )
    parts.append(
        f'<text x="{_number(left + 6)}" y="{_number(plot_top + size / 2)}" font-size="11" '
        f'fill="{_INK_MUTED}" text-anchor="middle" '
        f'transform="rotate(-90 {_number(left + 6)} {_number(plot_top + size / 2)})">'
        "Share that rose</text>"
    )

    largest = max(band.count for band in bands)
    for band in bands:
        # Area, not radius, tracks the count, so a band with four times the
        # issuer-days does not look sixteen times as heavy.
        radius = 4 + 6 * math.sqrt(band.count / largest)
        cx, cy = x(band.forecast), y(band.realised)
        parts.append(
            f"<g><title>{band.lower:.1f} to {band.upper:.1f}: forecast {band.forecast:.3f}, "
            f"realised {band.realised:.3f}, {band.count:,} issuer-days</title>"
            f'<circle cx="{_number(cx)}" cy="{_number(cy)}" r="{_number(radius)}" '
            f'fill="{_SERIES}" stroke="{_SURFACE}" stroke-width="2"/></g>'
        )
        parts.append(_text(cx + radius + 4, cy + 4, f"n={band.count:,}", fill=_INK_SECONDARY))
    return parts


def render_baseline_figure(evaluation: Evaluation) -> str:
    """Render the daily information coefficient and calibration as one SVG."""
    body: Sequence[str] = [
        *_ic_panel(evaluation, left=24, top=20, width=560),
        *_calibration_panel(evaluation, left=604, top=20),
    ]
    ic = evaluation.ic
    summary = (
        f"Mean information coefficient {ic.mean:+.4f} (t = {ic.t_statistic:+.2f}) over "
        f"{ic.days} scored days; calibration across {evaluation.pooled_issuer_days:,} "
        "issuer-days."
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" height="{_HEIGHT}" '
            f'viewBox="0 0 {_WIDTH} {_HEIGHT}" font-family=\'{_FONT}\' role="img">',
            "<title>Baseline arm: insider transactions</title>",
            f"<desc>{escape(summary)}</desc>",
            f'<rect x="0.5" y="0.5" width="{_WIDTH - 1}" height="{_HEIGHT - 1}" rx="8" '
            f'fill="{_SURFACE}" stroke="{_BORDER}"/>',
            *body,
            "</svg>",
            "",
        ]
    )
