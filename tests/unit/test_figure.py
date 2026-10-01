"""The figure must show the evaluation it was rendered from, and nothing else.

Each test parses the SVG rather than matching its text, so a change in
formatting cannot pass for a change in content or hide one.
"""

import math
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import replace
from datetime import date

from arbiter.arms.evaluate import Evaluation
from arbiter.evaluation.figure import render_baseline_figure
from arbiter.evaluation.metrics import CalibrationBin, ICSummary

SVG = "{http://www.w3.org/2000/svg}"

DAILY = {
    date(2026, 6, 12): 0.11,
    date(2026, 6, 18): -0.12,
    date(2026, 6, 25): 0.08,
    date(2026, 7, 1): 0.40,
}


def _evaluation() -> Evaluation:
    return Evaluation(
        ic=ICSummary(
            mean=0.12, standard_error=0.105, t_statistic=1.12, days=4, observations=200
        ),
        daily_ic=DAILY,
        calibration=[
            CalibrationBin(lower=0.3, upper=0.4, forecast=0.37, realised=0.38, count=61),
            CalibrationBin(lower=0.4, upper=0.5, forecast=0.44, realised=0.42, count=445),
            CalibrationBin(lower=0.5, upper=0.6, forecast=0.0, realised=0.0, count=0),
        ],
        brier_skill=0.002,
        brier_skill_scored_rate=-0.001,
        fitted_base_rate=0.45,
        base_rate=0.43,
        train_days=[date(2026, 3, 2), date(2026, 6, 1)],
        test_days=sorted(DAILY),
        embargoed_days=[],
        train_events=3000,
        test_events=1300,
        test_events_unlabelled=0,
        test_issuer_days=200,
        pooled_issuer_days=506,
        coefficients={"is_purchase": 0.2},
    )


def _parse(svg: str) -> ElementTree.Element:
    return ElementTree.fromstring(svg)


def _bars(root: ElementTree.Element) -> dict[str, str]:
    """Map each bar's tooltip to its path data."""
    bars: dict[str, str] = {}
    for group in root.iter(f"{SVG}g"):
        title, path = group.find(f"{SVG}title"), group.find(f"{SVG}path")
        if title is not None and path is not None and title.text:
            bars[title.text] = path.attrib["d"]
    return bars


def test_the_figure_is_well_formed_svg():
    root = _parse(render_baseline_figure(_evaluation()))

    assert root.tag == f"{SVG}svg"


def test_every_scored_day_is_drawn_as_one_bar():
    bars = _bars(_parse(render_baseline_figure(_evaluation())))

    assert sorted(bars) == sorted(
        f"{day.isoformat()}: {value:+.4f}" for day, value in DAILY.items()
    )


def test_a_negative_coefficient_is_drawn_below_the_baseline_and_a_positive_one_above():
    """SVG y grows downward: a bar rising from the baseline moves to a smaller y."""
    bars = _bars(_parse(render_baseline_figure(_evaluation())))

    for label, path in bars.items():
        start, first_edge = (
            float(value) for value in re.findall(r"M[\d.]+,([\d.]+) V([\d.]+)", path)[0]
        )
        rises = first_edge < start
        assert rises == (float(label.split(": ")[1]) > 0), label


def test_a_taller_coefficient_draws_a_taller_bar():
    bars = _bars(_parse(render_baseline_figure(_evaluation())))

    def height(day: date) -> float:
        path = bars[f"{day.isoformat()}: {DAILY[day]:+.4f}"]
        start, edge = (
            float(value) for value in re.findall(r"M[\d.]+,([\d.]+) V([\d.]+)", path)[0]
        )
        return abs(start - edge)

    assert height(date(2026, 7, 1)) > height(date(2026, 6, 12)) > height(date(2026, 6, 25))


def test_only_bands_holding_forecasts_are_plotted():
    root = _parse(render_baseline_figure(_evaluation()))

    circles = list(root.iter(f"{SVG}circle"))

    assert len(circles) == 2


def test_the_mean_and_the_pooled_count_are_stated():
    svg = render_baseline_figure(_evaluation())

    assert "mean +0.120" in svg
    assert "506 scored issuer-days" in svg


def test_the_same_evaluation_renders_identically():
    assert render_baseline_figure(_evaluation()) == render_baseline_figure(_evaluation())


def test_a_different_result_renders_differently():
    """The premise of the test above: the output depends on the evaluation."""
    moved = replace(_evaluation(), daily_ic={**DAILY, date(2026, 6, 25): 0.3})

    assert render_baseline_figure(moved) != render_baseline_figure(_evaluation())


def test_a_day_with_an_undefined_coefficient_is_left_out_rather_than_crashing():
    undefined = replace(_evaluation(), daily_ic={**DAILY, date(2026, 7, 8): math.nan})

    bars = _bars(_parse(render_baseline_figure(undefined)))

    assert len(bars) == len(DAILY)


def test_coefficients_that_are_all_zero_still_render():
    """The case the report calls "No conclusion" must not crash the figure."""
    flat = replace(
        _evaluation(),
        ic=ICSummary(
            mean=0.0, standard_error=0.0, t_statistic=math.nan, days=4, observations=200
        ),
        daily_ic=dict.fromkeys(DAILY, 0.0),
    )

    assert _parse(render_baseline_figure(flat)).tag == f"{SVG}svg"


def test_a_band_realising_outside_its_forecast_range_stays_on_the_canvas():
    edge = replace(
        _evaluation(),
        calibration=[
            CalibrationBin(lower=0.4, upper=0.5, forecast=0.45, realised=0.44, count=400),
            CalibrationBin(lower=0.6, upper=0.7, forecast=0.62, realised=1.0, count=2),
        ],
    )
    root = _parse(render_baseline_figure(edge))
    height = float(root.attrib["height"])

    for circle in root.iter(f"{SVG}circle"):
        radius = float(circle.attrib["r"])
        assert radius <= float(circle.attrib["cy"]) <= height - radius
