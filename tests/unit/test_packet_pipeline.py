"""Packets are built for every stored event, not for the measurable ones.

The property under test is the one that is easiest to break by accident and
impossible to see afterwards: a packet describes what was knowable at a filing,
and whether that filing's outcome has resolved is a fact about the calendar.
Building only resolvable events would make the packet set depend on when the
pass ran, and would repeat a defect this project has already had, where a day's
features were fitted over only its labelled rows.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from arbiter.ingestion.edgar import is_trading_day
from arbiter.ingestion.market import Bar
from arbiter.packets.pipeline import UnsupportedPacketDomainError, build_day
from arbiter.packets.store import partition_exists, read_packets

DAY = date(2026, 7, 13)


def _row(accession: str, ticker: str, hour: int = 20) -> dict[str, object]:
    return {
        "accession_no": accession,
        "as_of": datetime(2026, 7, 13, hour, 47, tzinfo=UTC),
        "cik": 764180,
        "ticker": ticker,
        "issuer": "Altria Group Inc.",
        "insider_name": "Jane Roe",
        "position": "Officer",
        "transaction_code": "P",
        "shares": Decimal("1000"),
        "price": Decimal("50.00"),
        "value_usd": Decimal("50000.00"),
        "remaining_shares": Decimal("4000"),
        "is_10b5_1": False,
    }


def _bars(count: int = 5) -> list[Bar]:
    return [
        Bar(
            timestamp=datetime(2026, 7, 13, 4, 0, tzinfo=UTC) - timedelta(days=offset + 1),
            open=Decimal("50"),
            high=Decimal("51"),
            low=Decimal("49"),
            close=Decimal("50"),
            volume=Decimal("1000"),
        )
        for offset in range(count)
    ]


def _store(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    from pydantic import BaseModel, ConfigDict

    from arbiter.ingestion.store import write_events

    class _Event(BaseModel):
        model_config = ConfigDict(frozen=True)

        accession_no: str
        as_of: datetime
        cik: int
        ticker: str
        issuer: str
        insider_name: str
        position: str
        transaction_code: str
        shares: Decimal
        # Nullable exactly as `InsiderEvent` is. A stricter stub cannot store the
        # rows production actually holds, so the isolation path below would be
        # untestable and the divergence would read as the code being safe.
        price: Decimal | None
        value_usd: Decimal | None
        remaining_shares: Decimal | None
        is_10b5_1: bool

    root = tmp_path / "events"
    write_events([_Event(**row) for row in rows], root, "insider", DAY)  # pyright: ignore[reportArgumentType]
    return root


def _fetch(series: dict[str, list[Bar]]):
    calls: list[tuple[list[str], date, date]] = []

    def fetch(symbols, start, end):  # pyright: ignore[reportMissingParameterType]
        calls.append((list(symbols), start, end))
        return {symbol: series.get(symbol, []) for symbol in symbols}

    return fetch, calls


def test_a_packet_is_built_for_every_event_with_a_price_series(tmp_path: Path):
    root = _store(tmp_path, [_row("a", "MO"), _row("b", "MO")])
    fetch, _ = _fetch({"MO": _bars()})

    written, excluded = build_day(
        DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP"
    )

    assert written == 2
    assert excluded == []
    assert len(read_packets(tmp_path / "packets", "insider", DAY)) == 2


def test_packets_are_built_without_consulting_labels(tmp_path: Path):
    """The pass reads events only; no label store exists in this tree at all.

    If building ever came to depend on resolvability, this would fail rather
    than quietly produce a smaller packet set.
    """
    root = _store(tmp_path, [_row("a", "MO")])
    fetch, _ = _fetch({"MO": _bars()})

    written, _ = build_day(
        DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP"
    )

    assert written == 1
    assert not (root / "labels-insider").exists()


def test_every_symbol_is_fetched_in_one_request(tmp_path: Path):
    """A request per issuer would cost hundreds against a per-minute limit."""
    root = _store(tmp_path, [_row("a", "MO"), _row("b", "KO"), _row("c", "PEP")])
    fetch, calls = _fetch({"MO": _bars(), "KO": _bars(), "PEP": _bars()})

    build_day(DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP")

    assert len(calls) == 1
    assert calls[0][0] == ["KO", "MO", "PEP"]


def test_the_window_reaches_back_for_the_history_a_packet_describes(tmp_path: Path):
    """Counted in sessions from the market calendar, not against the constant.

    An earlier version asserted `(DAY - start).days >= HISTORY_DAYS`, which
    restates the expression the window is built from and reads `x >= x`. It
    passed with the history shortened to twelve days.
    """
    root = _store(tmp_path, [_row("a", "MO")])
    fetch, calls = _fetch({"MO": _bars()})

    build_day(DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP")

    _, start, end = calls[0]
    sessions = sum(
        1
        for offset in range((DAY - start).days)
        if is_trading_day(start + timedelta(days=offset))
    )

    assert sessions >= 63, (
        f"{sessions} sessions between {start} and {DAY}; a packet's volume "
        "percentile ranks over 63"
    )
    assert end <= DAY, "a packet reads no bar after its own event"


def test_an_event_with_no_price_history_is_recorded_rather_than_dropped(tmp_path: Path):
    root = _store(tmp_path, [_row("a", "MO"), _row("b", "KO")])
    fetch, _ = _fetch({"MO": _bars()})  # KO returns nothing

    written, excluded = build_day(
        DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP"
    )

    assert written == 1
    assert [record["accession_no"] for record in excluded] == ["b"]
    assert "no price history" in excluded[0]["reason"]


def test_one_unpriceable_issuer_does_not_cost_the_others(tmp_path: Path):
    """The batching lesson, applied here: a bad element costs only itself."""
    root = _store(
        tmp_path, [_row(str(index), "MO") for index in range(5)] + [_row("bad", "KO")]
    )
    fetch, _ = _fetch({"MO": _bars()})

    written, excluded = build_day(
        DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP"
    )

    assert written == 5
    assert len(excluded) == 1


def test_a_day_with_no_usable_events_still_writes_a_partition(tmp_path: Path):
    """A day built and found empty must differ from a day never built."""
    root = _store(tmp_path, [_row("a", "KO")])
    fetch, _ = _fetch({})

    written, excluded = build_day(
        DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP"
    )

    assert written == 0
    assert len(excluded) == 1
    assert partition_exists(tmp_path / "packets", "insider", DAY)


def test_no_request_is_made_when_no_event_has_a_usable_symbol(tmp_path: Path):
    """Nothing to ask about is not a reason to ask."""
    root = _store(tmp_path, [_row("a", "NOT A TICKER")])
    fetch, calls = _fetch({})

    written, excluded = build_day(
        DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP"
    )

    assert calls == []
    assert written == 0
    assert "no tradeable symbol" in excluded[0]["reason"]


def test_the_benchmark_recorded_is_the_one_the_issuer_maps_to(tmp_path: Path):
    root = _store(tmp_path, [_row("a", "MO")])
    fetch, _ = _fetch({"MO": _bars()})

    build_day(DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLE")

    (packet,) = read_packets(tmp_path / "packets", "insider", DAY)
    assert packet.issuer.sector_etf == "XLE"


def test_a_domain_with_no_extract_model_is_refused(tmp_path: Path):
    root = _store(tmp_path, [_row("a", "MO")])
    fetch, _ = _fetch({"MO": _bars()})

    with pytest.raises(UnsupportedPacketDomainError, match="redflag"):
        build_day(
            DAY,
            root,
            tmp_path / "packets",
            fetch_bars=fetch,
            benchmark_for=lambda _: "XLP",
            domain="redflag",
        )


def test_a_packet_holds_no_bar_that_had_not_closed(tmp_path: Path):
    """The end-to-end form of the guard, through real storage.

    The filing lands at 10:00 Eastern, mid-session, so the day's own bar is
    fetched and must not survive into the packet.
    """
    root = _store(tmp_path, [_row("a", "MO", hour=14)])  # 10:00 Eastern
    same_session = Bar(
        timestamp=datetime(2026, 7, 13, 4, 0, tzinfo=UTC),
        open=Decimal("50"),
        high=Decimal("51"),
        low=Decimal("49"),
        close=Decimal("50"),
        volume=Decimal("1000"),
    )
    fetch, _ = _fetch({"MO": [*_bars(), same_session]})

    build_day(DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP")

    (packet,) = read_packets(tmp_path / "packets", "insider", DAY)
    assert DAY not in [bar.session for bar in packet.bars]


def test_a_packets_identity_does_not_depend_on_the_other_events_that_day(tmp_path: Path):
    """A hash must name the evidence, not the day's roster.

    The window is planned across the day because requests are the constrained
    resource, but a packet cut from that shared span takes its oldest bar from
    whichever event happened to be earliest. Its hash then moves when an
    unrelated filing is added or removed — so re-ingesting a day would silently
    re-hash every other packet in it, and every prediction citing the old hashes
    would become untraceable with no error anywhere.
    """
    late = _row("late", "MO", hour=20)
    early = _row("early", "MO", hour=14)
    bars = [
        Bar(
            timestamp=datetime(2026, 7, 13, 4, 0, tzinfo=UTC) - timedelta(days=offset),
            open=Decimal("50"),
            high=Decimal("51"),
            low=Decimal("49"),
            close=Decimal("50"),
            volume=Decimal("1000"),
        )
        for offset in range(1, 200)
    ]

    alone_root = _store(tmp_path / "alone", [late])
    fetch_alone, _ = _fetch({"MO": bars})
    build_day(
        DAY, alone_root, tmp_path / "p1", fetch_bars=fetch_alone, benchmark_for=lambda _: "XLP"
    )
    (alone,) = read_packets(tmp_path / "p1", "insider", DAY)

    together_root = _store(tmp_path / "together", [early, late])
    fetch_together, _ = _fetch({"MO": bars})
    build_day(
        DAY,
        together_root,
        tmp_path / "p2",
        fetch_bars=fetch_together,
        benchmark_for=lambda _: "XLP",
    )
    together = next(
        packet
        for packet in read_packets(tmp_path / "p2", "insider", DAY)
        if packet.event_id == "late"
    )

    assert together.content_hash == alone.content_hash


def test_the_price_window_is_bounded_by_the_market_session_not_the_utc_date(tmp_path: Path):
    """An evening filing's UTC date is the next day, and the feed refuses it.

    EDGAR gives a Form 4 accepted at 21:00 Eastern that day's filing date, but
    its UTC instant falls on the next calendar day. Asking the consolidated feed
    for a window ending then is asking for the current session, which it
    refuses — so a day containing an evening filing, which is most days, would
    fail to build at all.
    """
    # 02:00 UTC on the 13th is 22:00 Eastern on the 12th: one session, two
    # calendar dates, and only the Eastern one is a market session.
    root = _store(tmp_path, [_row("evening", "MO", hour=2)])
    fetch, calls = _fetch({"MO": _bars()})

    build_day(DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP")

    _, _, end = calls[0]
    assert end == date(2026, 7, 12), (
        f"window ends {end}; the event's market session is 2026-07-12 and asking "
        "the consolidated feed for the 13th is asking for the current session"
    )


def test_one_malformed_row_costs_one_event_and_not_the_day(tmp_path: Path):
    """The batching rule, at the row level.

    Written because adding a bare `raise` to the isolation handler left the
    whole suite passing: the fix was indistinguishable from its absence. Before
    it, an unparseable price escaped `build_day` before `write_packets` ran, so
    one bad row discarded every other packet on the day and left no partition at
    all.
    """
    good = [_row(f"ok-{index}", "MO") for index in range(3)]
    bad = {**_row("bad", "MO"), "value_usd": None}  # nullable in store, unusable here
    root = _store(tmp_path, [*good, bad])
    fetch, _ = _fetch({"MO": _bars()})

    written, excluded = build_day(
        DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP"
    )

    assert written == 3
    assert [record["accession_no"] for record in excluded] == ["bad"]
    assert len(read_packets(tmp_path / "packets", "insider", DAY)) == 3


def test_the_days_exclusions_are_written_beside_its_packets(tmp_path: Path):
    """A record nothing reads is a record that does not exist."""
    import json

    root = _store(tmp_path, [_row("kept", "MO"), _row("dropped", "KO")])
    fetch, _ = _fetch({"MO": _bars()})

    build_day(DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP")

    path = (
        tmp_path / "packets" / "unpriceable" / "packets" / "insider" / f"{DAY.isoformat()}.json"
    )
    assert path.exists()
    recorded = json.loads(path.read_text())
    assert [entry["accession_no"] for entry in recorded] == ["dropped"]


def test_a_day_excluding_nothing_still_writes_its_exclusion_record(tmp_path: Path):
    """The negative control, without which absent and empty are one state."""
    import json

    root = _store(tmp_path, [_row("kept", "MO")])
    fetch, _ = _fetch({"MO": _bars()})

    build_day(DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP")

    path = (
        tmp_path / "packets" / "unpriceable" / "packets" / "insider" / f"{DAY.isoformat()}.json"
    )
    assert path.exists(), "a day that excluded nothing must say so"
    assert json.loads(path.read_text()) == []


def test_a_series_lying_entirely_after_its_event_is_excluded_not_emptied(tmp_path: Path):
    """The truncate-before-emptiness fix, replayed.

    A non-empty series that all postdates the event passes a test on the
    vendor's rows and truncates to nothing, writing a hashed, citable packet
    carrying no market evidence at all.
    """
    root = _store(tmp_path, [_row("mid-session", "MO", hour=14)])
    after = [
        Bar(
            timestamp=datetime(2026, 7, 13, 4, 0, tzinfo=UTC) + timedelta(days=offset),
            open=Decimal("50"),
            high=Decimal("51"),
            low=Decimal("49"),
            close=Decimal("50"),
            volume=Decimal("1000"),
        )
        for offset in range(0, 5)
    ]
    fetch, _ = _fetch({"MO": after})

    written, excluded = build_day(
        DAY, root, tmp_path / "packets", fetch_bars=fetch, benchmark_for=lambda _: "XLP"
    )

    assert written == 0
    assert "before the event" in excluded[0]["reason"]
    assert read_packets(tmp_path / "packets", "insider", DAY) == []
