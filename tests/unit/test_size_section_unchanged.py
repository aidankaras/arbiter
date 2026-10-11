"""The by-size section renders byte-for-byte as it did before the per-group refactor.

The goldens under `tests/fixtures/report/` were rendered by the size-only
implementation, before the arithmetic and row formatting were shared with the
insider-type breakdown. A diff here is a change in what a published report says.
"""

from datetime import date
from pathlib import Path

import pytest

from arbiter.arms.evaluate import IssuerScore
from arbiter.evaluation.report import render_baseline_report
from arbiter.evaluation.size import SizeBreakdown, ic_by_size
from tests.unit.test_report import GENERATED
from tests.unit.test_report import _evaluation as report_evaluation
from tests.unit.test_size_buckets import _breakdown, _scores_and_sizes

GOLDEN = Path(__file__).parent.parent / "fixtures" / "report"


def _section(text: str) -> str:
    # Trailing blank lines are not compared: the repository's end-of-file hook
    # normalises the goldens to a single final newline.
    lines = text.splitlines()
    start, end = lines.index("## By issuer size"), lines.index("## By insider type")
    return "\n".join(lines[start:end]).rstrip("\n") + "\n"


def _populated() -> SizeBreakdown:
    scores, sizes = _scores_and_sizes(days=4)
    day: date = next(iter(scores))
    scores[day] = [s for s in scores[day] if sizes[(day, s.cik)] != 30e6]
    for rows in scores.values():
        rows.extend(IssuerScore(cik, 0.5, 0.0) for cik in (1, 2, 3))
    return ic_by_size(scores, sizes, overlap_lags=1)


@pytest.mark.parametrize(
    ("golden", "evaluation", "size", "omitted"),
    [
        ("size_section_populated.md", (0.01, 0.02, 0.5), "populated", None),
        ("size_section_crossing.md", (0.01, 0.02, 0.5), "crossing", None),
        ("size_section_significant.md", (0.05, 0.01, 5.0), "crossing", None),
        ("size_section_omitted.md", (0.01, 0.02, 0.5), None, "Omitted: no packets."),
    ],
)
def test_the_size_section_matches_the_pre_refactor_render(
    golden: str,
    evaluation: tuple[float, float, float],
    size: str | None,
    omitted: str | None,
):
    breakdown = {"populated": _populated, "crossing": lambda: _breakdown(2.5), None: None}[size]
    text = render_baseline_report(
        report_evaluation(*evaluation),
        GENERATED,
        size=breakdown() if breakdown else None,
        size_omitted=omitted,
    )
    assert _section(text) == (GOLDEN / golden).read_text()
