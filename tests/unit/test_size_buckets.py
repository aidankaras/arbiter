"""The information coefficient by issuer size.

Expectations are derived from the requirement, not from the code: a size is a
median of dollar volume over the latest 20 sessions that a hand can compute, a
bucket is a third of the day's sized issuers, and a forecast built to rank
perfectly in one third and backwards in another must score +1 and -1 there
while the pooled coefficient sits strictly between.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.arms.evaluate import IssuerScore, evaluate_baseline
from arbiter.evaluation.metrics import MIN_EVENTS_PER_DAY, information_coefficient
from arbiter.evaluation.report import render_baseline_report
from arbiter.evaluation.size import (
    SIZE_SESSIONS,
    assign_buckets,
    ic_by_size,
    load_sizes,
    median_dollar_volume,
    packet_sizes,
)
from arbiter.packets.schema import EvidencePacket, InsiderExtract, IssuerIdentity, SessionBar
from arbiter.packets.store import write_packets
from tests.unit.test_baseline_evaluation import _informative_days
from tests.unit.test_report import GENERATED
from tests.unit.test_report import _evaluation as report_evaluation

#: A permutation of 0..9 whose rank correlation with 0..9 is -0.006: no ranking.
_UNRELATED = (0, 1, 8, 9, 7, 6, 5, 4, 3, 2)


def _sessions(count: int, last: date = date(2026, 8, 31)) -> list[date]:
    """The `count` weekdays ending on `last`, oldest first."""
    found: list[date] = []
    day = last
    while len(found) < count:
        if day.weekday() < 5:
            found.append(day)
        day -= timedelta(days=1)
    return found[::-1]


def _bar(session: date, close: str, volume: str) -> SessionBar:
    price = Decimal(close)
    return SessionBar(
        session=session, open=price, high=price, low=price, close=price, volume=Decimal(volume)
    )


def _bars(
    volumes: list[int], close: str = "2", last: date = date(2026, 8, 31)
) -> tuple[SessionBar, ...]:
    return tuple(
        _bar(session, close, str(volume))
        for session, volume in zip(_sessions(len(volumes), last), volumes, strict=True)
    )


def _packet(
    cik: int, event_id: str, as_of: datetime, bars: tuple[SessionBar, ...]
) -> EvidencePacket:
    return EvidencePacket(
        event_id=event_id,
        domain="insider",
        as_of=as_of,
        issuer=IssuerIdentity(cik=cik, ticker="T", company="Co", sector_etf="XLK"),
        extracted=InsiderExtract(
            insider_name="A",
            position="Officer",
            transaction_code="P",
            shares=Decimal("1"),
            price=Decimal("1"),
            value_usd=Decimal("1"),
        ),
        bars=bars,
    )


# 2026-09-01 14:00 UTC is 10:00 Eastern: the 09-01 session has begun, not closed.
MORNING = datetime(2026, 9, 1, 14, 0, tzinfo=UTC)
EVENING = datetime(2026, 9, 1, 22, 0, tzinfo=UTC)


def test_size_is_the_median_dollar_volume_of_the_last_twenty_sessions():
    # Five early sessions of enormous volume, then volumes 1000..20000 at a
    # close of 2. Dollar volumes 2000..40000: the median of twenty is the mean
    # of the 10th and 11th, (20000 + 22000) / 2 = 21000. Counting the early
    # sessions, or taking volume without the price, gives a different number.
    volumes = [10**9] * 5 + [1000 * k for k in range(1, 21)]
    assert median_dollar_volume(_bars(volumes)) == 21000.0


def test_fewer_than_twenty_sessions_give_no_size_and_twenty_give_one():
    assert median_dollar_volume(_bars([1000] * (SIZE_SESSIONS - 1))) is None
    assert median_dollar_volume(_bars([1000] * SIZE_SESSIONS)) == 2000.0
    assert median_dollar_volume(()) is None


def test_events_of_one_issuer_filed_together_share_one_size():
    bars = _bars([1000 * k for k in range(1, 26)])
    packets = [_packet(7, f"a{n}", MORNING, bars) for n in range(3)]
    assert len({median_dollar_volume(p.bars) for p in packets}) == 1
    assert packet_sizes(packets) == {7: median_dollar_volume(bars)}


def test_events_straddling_the_close_take_the_size_of_the_earliest_acceptance():
    # The evening filing sees the 09-01 session as well, so its bars (and its
    # size) differ from the morning one's. The earliest acceptance decides.
    base = [1000 * k for k in range(1, 26)]
    morning = _packet(7, "m", MORNING, _bars(base, last=date(2026, 8, 31)))
    evening = _packet(7, "e", EVENING, _bars([*base[1:], 10**7], last=date(2026, 9, 1)))
    assert median_dollar_volume(morning.bars) != median_dollar_volume(evening.bars)
    assert packet_sizes([evening, morning]) == {7: median_dollar_volume(morning.bars)}


def test_an_issuer_without_enough_bars_is_absent_from_the_sizes():
    short = _packet(1, "s", MORNING, _bars([1000] * 5))
    full = _packet(2, "f", MORNING, _bars([1000] * 30))
    assert set(packet_sizes([short, full])) == {2}


def test_sizes_load_through_the_store_and_missing_partitions_are_counted(tmp_path: Path):
    built, absent = date(2026, 9, 1), date(2026, 9, 2)
    write_packets([_packet(2, "f", MORNING, _bars([1000] * 30))], tmp_path, "insider", built)
    lookup, partitions = load_sizes(tmp_path, "insider", [built, absent])
    assert lookup == {(built, 2): 2000.0}
    assert partitions == 1
    assert load_sizes(tmp_path / "none", "insider", [built]) == ({}, 0)


def test_terciles_are_rank_based_with_ties_broken_by_cik():
    # Ten issuers: ranks 0..9 go to bucket 3r // 10, so 4 / 3 / 3. Issuers 5 and
    # 4 tie on size, and the lower CIK takes the lower rank.
    sizes = {cik: float(cik) for cik in range(10, 0, -1)}
    sizes[5] = sizes[4]
    buckets = assign_buckets(sizes)
    assert sorted(buckets.values()) == [0] * 4 + [1] * 3 + [2] * 3
    assert buckets[4] <= buckets[5]
    # Equal sizes at a boundary still split deterministically.
    flat = dict.fromkeys(range(1, 7), 5.0)
    assert [assign_buckets(flat)[cik] for cik in range(1, 7)] == [0, 0, 1, 1, 2, 2]


def _scores_and_sizes(
    days: int = 12,
) -> tuple[dict[date, list[IssuerScore]], dict[tuple[date, int], float]]:
    """Thirty issuers a day, size rising with CIK.

    Small (CIKs 1-10): returns rank with the forecast. Large (21-30): reversed.
    Middle: returns unrelated to the forecast. On odd days two small-bucket
    returns swap, so days differ and an error across days exists.
    """
    scores: dict[date, list[IssuerScore]] = {}
    sizes: dict[tuple[date, int], float] = {}
    for offset in range(days):
        day = date(2026, 8, 3) + timedelta(days=offset)
        small = [cik / 100 for cik in range(1, 11)]
        if offset % 2:
            small[0], small[1] = small[1], small[0]
        rows: list[IssuerScore] = []
        for cik in range(1, 31):
            forecast = cik / 100
            if cik <= 10:
                realised = small[cik - 1]
            elif cik <= 20:
                realised = _UNRELATED[cik - 11] / 100
            else:
                realised = -forecast
            rows.append(IssuerScore(cik, forecast, realised))
            sizes[(day, cik)] = cik * 1e6
        scores[day] = rows
    return scores, sizes


def test_a_forecast_that_ranks_only_the_small_third_scores_only_there():
    scores, sizes = _scores_and_sizes()
    result = ic_by_size(scores, sizes)
    small, middle, large = result.buckets

    # Premise: the pooled coefficient is neither +1 nor -1, so a bucket result
    # of +-1 cannot come from the whole sample ranking one way.
    pooled = [
        information_coefficient([s.forecast for s in day], [s.abnormal_return for s in day])
        for day in scores.values()
    ]
    assert all(abs(value) < 0.9 for value in pooled)

    assert small.ic is not None and large.ic is not None and middle.ic is not None
    assert small.ic.mean > 0.98
    assert large.ic.mean == pytest.approx(-1.0)
    assert abs(middle.ic.mean) < 0.05
    assert (small.days, small.issuer_days) == (12, 120)
    assert result.unbucketed == 0
    assert result.scored_issuer_days == 360
    # Ranges come from the size of each bucket's members, in dollars.
    assert (small.lower, small.upper) == (1e6, 10e6)
    assert (large.lower, large.upper) == (21e6, 30e6)


def test_buckets_are_formed_within_each_day_not_across_the_sample():
    scores, sizes = _scores_and_sizes()
    # Market-wide volume doubles on every other day; the ordering within a day
    # is unchanged, so no issuer may change bucket.
    doubled = {
        (day, cik): size * (2 if day.toordinal() % 2 else 1)
        for (day, cik), size in sizes.items()
    }
    for day in scores:
        assert assign_buckets(
            {cik: s for (d, cik), s in sizes.items() if d == day}
        ) == assign_buckets({cik: s for (d, cik), s in doubled.items() if d == day})
    base, drifted = ic_by_size(scores, sizes), ic_by_size(scores, doubled)
    for left, right in zip(base.buckets, drifted.buckets, strict=True):
        assert left.ic == right.ic
        assert left.issuer_days == right.issuer_days
    # Premise: ranking the whole sample at once would have moved issuers, so the
    # equality above is not satisfied by a drift too small to matter.
    pooled = assign_buckets(doubled)
    assert any(
        pooled[(day, cik)]
        != assign_buckets({c: s for (d, c), s in doubled.items() if d == day})[cik]
        for day, cik in doubled
    )


def test_a_bucket_day_below_the_threshold_contributes_nothing():
    # Days with 10 per bucket reach the threshold; days with 9 do not. The thin
    # days must add neither a coefficient nor their issuer-days to a bucket.
    assert MIN_EVENTS_PER_DAY == 10
    scores, sizes = _scores_and_sizes(days=6)
    thin_days = list(scores)[:3]
    for day in thin_days:
        keep = {c for c in range(1, 31) if c % 10 != 0}  # drops 10, 20, 30
        scores[day] = [s for s in scores[day] if s.cik in keep]
    result = ic_by_size(scores, sizes)
    for bucket in result.buckets:
        assert bucket.days == 3
        assert bucket.issuer_days == 30


def test_issuer_days_without_a_size_are_counted_and_left_out():
    scores, sizes = _scores_and_sizes(days=4)
    for rows in scores.values():
        rows.extend(IssuerScore(cik, 0.5, 0.0) for cik in (31, 32, 33))
    result = ic_by_size(scores, sizes)
    assert result.unbucketed == 12
    assert result.scored_issuer_days == 4 * 33
    # Thirty sized issuers a day still make three buckets of ten.
    assert [bucket.issuer_days for bucket in result.buckets] == [40, 40, 40]


def test_the_report_omits_the_section_with_a_reason_when_no_packets_exist(tmp_path: Path):
    lookup, partitions = load_sizes(tmp_path, "insider", [date(2026, 8, 3)])
    assert (lookup, partitions) == ({}, 0)
    text = render_baseline_report(
        report_evaluation(0.01, 0.02, 0.5),
        GENERATED,
        size=None,
        size_omitted="Omitted: no packets.",
    )
    assert "## By issuer size" in text
    assert "Omitted: no packets." in text
    assert "| Small |" not in text


def test_the_section_reports_each_bucket_and_the_unsized_count():
    scores, sizes = _scores_and_sizes(days=4)
    for rows in scores.values():
        rows.extend(IssuerScore(cik, 0.5, 0.0) for cik in (31, 32, 33))
    text = render_baseline_report(
        report_evaluation(0.01, 0.02, 0.5),
        GENERATED,
        size=ic_by_size(scores, sizes, overlap_lags=1),
    )
    assert "12 of 132 scored issuer-days have no size" in text
    for label in ("Small", "Middle", "Large"):
        assert f"| {label} |" in text
    assert "| Small | 1.00 to 10.00 |" in text
    assert text.index("## Information coefficient by day") < text.index("## By issuer size")
    assert text.index("## By issuer size") < text.index("## Calibration")


def test_adding_the_size_section_changes_no_other_line():
    scores, sizes = _scores_and_sizes(days=4)
    without = render_baseline_report(
        report_evaluation(0.01, 0.02, 0.5), GENERATED, size_omitted="x"
    ).splitlines()
    full = render_baseline_report(
        report_evaluation(0.01, 0.02, 0.5), GENERATED, size=ic_by_size(scores, sizes)
    ).splitlines()
    start, end = without.index("## By issuer size"), without.index("## Calibration")
    fstart, fend = full.index("## By issuer size"), full.index("## Calibration")
    assert without[:start] == full[:fstart]
    assert without[end:] == full[fend:]


def test_the_exposed_scores_reproduce_the_headline_coefficients():
    evaluation = evaluate_baseline(_informative_days(), overlap_lags=0)
    assert set(evaluation.issuer_scores) == set(evaluation.test_days)
    for day, value in evaluation.daily_ic.items():
        rows = evaluation.issuer_scores[day]
        assert value == information_coefficient(
            [row.forecast for row in rows], [row.abnormal_return for row in rows]
        )
    assert (
        sum(len(rows) for rows in evaluation.issuer_scores.values())
        == evaluation.pooled_issuer_days
    )
