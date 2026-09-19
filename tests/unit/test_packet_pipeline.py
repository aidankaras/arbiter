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
        price: Decimal
        value_usd: Decimal
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
