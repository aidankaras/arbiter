"""Command-line entry point.

Thin wrapper over the library. Every subcommand is a small adapter around a
function that is independently testable without the CLI.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

import typer
from pydantic import ValidationError

from arbiter import __version__
from arbiter.arms.evaluate import (
    INITIAL_FRACTION,
    REFIT_EVERY,
    DayOfEvents,
    LabelCoverageError,
    evaluate_baseline,
    load_days,
    stored_days,
)
from arbiter.arms.features import UnsupportedDomainError
from arbiter.config import Settings, get_settings
from arbiter.evaluation.figure import render_baseline_figure
from arbiter.evaluation.insider_type import (
    available_archives,
    ic_by_insider_type,
    load_history,
    load_issuer_day_classes,
)
from arbiter.evaluation.metrics import InsufficientObservationsError
from arbiter.evaluation.report import render_baseline_report
from arbiter.evaluation.resolution import HORIZONS, WindowsStillOpenError, resolve_stored_day
from arbiter.evaluation.size import ic_by_size, load_sizes
from arbiter.ingestion.backfill import backfill as run_backfill
from arbiter.ingestion.backfill import day_is_stored, ingest_and_resolve, sampled_trading_days
from arbiter.ingestion.market import Bar, current_session_date, daily_bars_tolerating_gaps
from arbiter.ingestion.pipeline import ingest_day
from arbiter.ingestion.sectors import benchmark_for_issuer, issuer_ticker

app = typer.Typer(
    name="arbiter",
    help="Event-driven equity forecasting from SEC filings.",
    no_args_is_help=True,
    add_completion=False,
)


def _settings_or_exit() -> Settings:
    """Return validated settings, or exit with the variable that needs setting.

    A stack trace is the wrong first impression for a missing environment
    variable, and it buries the one line that says which variable it was.
    """
    try:
        return get_settings()
    except ValidationError as exc:
        typer.echo("Configuration is incomplete:", err=True)
        for error in exc.errors():
            variable = str(error["loc"][0]).upper()
            typer.echo(f"  {variable}: {error['msg']}", err=True)
        typer.echo("\nCopy .env.example to .env and fill in the values above.", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def check() -> None:
    """Validate configuration and report what is reachable.

    Run this after editing the environment. It exits non-zero on invalid
    configuration so it can gate a scheduled run.
    """
    settings = _settings_or_exit()
    typer.echo(f"environment:  {settings.environment}")
    typer.echo(f"broker:       {settings.alpaca_base_url}")
    typer.echo(f"daily ceiling: ${settings.daily_spend_ceiling_usd}")
    typer.echo("configuration valid")


@app.command()
def ingest(
    day: str,
    root: str = "data/events",
    min_value_usd: str = "50000",
) -> None:
    """Ingest one day of filings into the event store.

    The day is a calendar date in ISO form. Re-running a day replaces that day's
    partitions, so a failed run is repeated rather than repaired.
    """
    counts = ingest_day(date.fromisoformat(day), Path(root), Decimal(min_value_usd))
    for domain, count in counts.items():
        typer.echo(f"{domain}: {count}")


@app.command()
def resolve(
    day: str,
    domain: str,
    root: str = "data/events",
) -> None:
    """Label one stored day of a domain, once its outcome window has closed.

    Runs a day at a time against the consolidated tape, which will not serve a
    window ending on the current session — so the most recent day that can be
    labeled is always at least one session behind ingestion. A day holding any
    event whose window is still open is refused, writing nothing, rather than
    stored as a partial measurement.

    Events whose issuer cannot be priced are recorded under `unpriceable/`
    beside the labels rather than dropped.
    """
    if domain not in HORIZONS:
        # Caught here because the next step reads that domain's partition, and a
        # misspelled domain would surface as "the day was never processed" —
        # pointing at the ingestion run rather than at the typo.
        known = ", ".join(sorted(HORIZONS))
        typer.echo(f"unknown domain '{domain}'; expected one of: {known}", err=True)
        raise typer.Exit(code=2)

    today = current_session_date()

    def fetch(symbols: Sequence[str], start: date, end: date) -> dict[str, list[Bar]]:
        return daily_bars_tolerating_gaps(list(symbols), start, end, today)

    try:
        written = resolve_stored_day(
            day=date.fromisoformat(day),
            domain=domain,
            root=Path(root),
            fetch_bars=fetch,
            benchmark_for=benchmark_for_issuer,
            today=today,
            ticker_for=issuer_ticker,
        )
    except WindowsStillOpenError as exc:
        typer.echo(f"not labelled yet: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"labels-{domain}: {written}")


@app.command(name="build-packets")
def build_packets(
    day: str,
    root: str = "data/events",
    packet_root: str = "data/packets",
) -> None:
    """Build one stored day's evidence packets.

    A packet is built for every stored event, not only for events whose outcome
    has resolved. A packet records what was knowable at the filing, and whether
    its label exists yet is a fact about the calendar — so this pass reads the
    event store and never the label store, and its output does not depend on
    when it was run.

    Events with no tradeable symbol, or whose issuer returned no price history,
    are reported here rather than silently absent from the day.
    """
    from arbiter.packets.pipeline import UnsupportedPacketDomainError, build_day

    today = current_session_date()

    def fetch(symbols: Sequence[str], start: date, end: date) -> dict[str, list[Bar]]:
        return daily_bars_tolerating_gaps(list(symbols), start, end, today)

    try:
        written, excluded = build_day(
            date.fromisoformat(day),
            Path(root),
            Path(packet_root),
            fetch_bars=fetch,
            benchmark_for=benchmark_for_issuer,
        )
    except UnsupportedPacketDomainError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error

    typer.echo(f"packets: {written}")
    if excluded:
        typer.echo(f"excluded: {len(excluded)}")
        for record in excluded[:10]:
            typer.echo(f"  {record['accession_no']}: {record['reason']}")
        if len(excluded) > 10:
            typer.echo(f"  ... and {len(excluded) - 10} more")


def _days_needed(initial_fraction: float) -> int:
    """Return the fewest populated days that leave an initial fit and two scored days."""
    needed = 3
    while needed - math.ceil(needed * initial_fraction) < 2:
        needed += 1
    return needed


@app.command()
def report(
    domain: str = "insider",
    root: str = "data/events",
    out: str = "reports",
    initial_fraction: float = INITIAL_FRACTION,
    refit_every: int = REFIT_EVERY,
    packet_root: str = "data/packets",
    insider_history: str = "data/raw/form345",
) -> None:
    """Fit the baseline arm on the stored history and write its measurement.

    Reads every day the store holds labels for, fits on the first
    `--initial-fraction` of them, then scores the rest walk-forward, refitting
    every `--refit-every` scored days. The report is written to `reports/` so that what
    was claimed at a given commit stays inspectable; recomputing against more
    data produces a new report rather than an edit to the old one.

    The information coefficient by issuer size is read from the evidence packets under
    `--packet-root`; with no packet partition for any scored day that section is
    left out, with a line saying so.

    The breakdown by insider type (routine versus opportunistic) classifies each event
    from the SEC Insider Transactions Data Sets in `--insider-history`; with no
    `*_form345.zip` there, that section is left out, with a line saying so. The first
    run streams the zips and caches the result beside them.
    """
    if not 0 < initial_fraction < 1:
        typer.echo(
            f"--initial-fraction must be strictly between 0 and 1, not {initial_fraction}",
            err=True,
        )
        raise typer.Exit(code=2)
    if refit_every < 1:
        typer.echo(f"--refit-every must be at least 1, not {refit_every}", err=True)
        raise typer.Exit(code=2)

    store = Path(root)
    days = stored_days(store, domain)
    if not days:
        typer.echo(f"no labelled {domain} days under {root}; run `arbiter backfill`", err=True)
        raise typer.Exit(code=1)

    loaded: list[DayOfEvents] = []
    try:
        # `load_days` is what raises: features are built as each day is read,
        # so the call has to sit inside the handler rather than before it.
        loaded = load_days(store, domain, days)
        evaluation = evaluate_baseline(
            loaded,
            initial_fraction=initial_fraction,
            refit_every=refit_every,
            # A window opens at the next session and closes HORIZONS[domain]
            # sessions after that one, so its exit session is the entry session
            # of the day HORIZONS[domain] later: it overlaps that many neighbours.
            overlap_lags=HORIZONS[domain],
        )
    except UnsupportedDomainError as exc:
        typer.echo(f"the baseline arm cannot score '{domain}': {exc}", err=True)
        raise typer.Exit(code=2) from None
    except LabelCoverageError as exc:
        typer.echo(f"refusing to score a day whose labels are stale: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except InsufficientObservationsError as exc:
        populated = sum(1 for day in loaded if len(day) > 0)
        typer.echo(
            f"cannot evaluate: {exc}. {populated} populated days are stored; at "
            f"--initial-fraction {initial_fraction} at least "
            f"{_days_needed(initial_fraction)} are needed. Collect more days with "
            "`arbiter backfill`, or change --initial-fraction: a larger window "
            "leaves fewer days to score, a smaller one leaves fewer to fit on.",
            err=True,
        )
        raise typer.Exit(code=1) from None

    # Both are rendered before either is written, so a failure in one cannot
    # leave a new report beside a figure from an earlier run.
    sizes, partitions = load_sizes(Path(packet_root), domain, evaluation.test_days)
    if partitions == 0:
        size_breakdown = None
        size_omitted = (
            f"Omitted: no {domain} packet partition under {packet_root} covers a scored day, "
            "so issuer size is unknown."
        )
    else:
        size_breakdown = ic_by_size(
            evaluation.issuer_scores, sizes, overlap_lags=HORIZONS[domain]
        )
        size_omitted = None
    type_breakdown = None
    type_omitted: str | None = None
    if domain != "insider":
        type_omitted = (
            f"Omitted: insider type is defined for insider transactions, not '{domain}'."
        )
    elif not available_archives(Path(insider_history)):
        type_omitted = (
            f"Omitted: no *_form345.zip under {insider_history}, so insiders' trading "
            "history is unknown."
        )
    else:
        history = load_history(Path(insider_history))
        type_breakdown = ic_by_insider_type(
            evaluation.issuer_scores,
            load_issuer_day_classes(store, domain, evaluation.test_days, history.trades),
            overlap_lags=HORIZONS[domain],
            history=history,
        )
    report_text = render_baseline_report(
        evaluation,
        generated_on=current_session_date(),
        size=size_breakdown,
        size_omitted=size_omitted,
        insider_type=type_breakdown,
        insider_type_omitted=type_omitted,
    )
    figure = render_baseline_figure(evaluation)
    destination = Path(out) / f"baseline-{domain}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report_text, encoding="utf-8")
    destination.with_suffix(".svg").write_text(figure, encoding="utf-8")

    typer.echo(
        f"{destination}: IC {evaluation.ic.mean:+.4f} "
        f"(se {evaluation.ic.standard_error:.4f}, t {evaluation.ic.t_statistic:+.2f}) "
        f"over {evaluation.ic.days} days, {evaluation.test_issuer_days:,} issuer-days"
    )


@app.command(name="ledger-up")
def ledger_up(ledger_dir: str = "data/ledger") -> None:
    """Start the local ledger, apply migrations, and print the URL to export.

    Run once per boot. The server keeps running after this command exits, and
    its URL does not change for a given directory, so it can live in `.env`.
    """
    import os

    from arbiter.db.local import ensure_local_ledger, upgrade_to_head
    from arbiter.db.session import get_engine

    url = ensure_local_ledger(Path(ledger_dir))
    os.environ["DATABASE_URL"] = url
    get_settings.cache_clear()
    get_engine.cache_clear()
    upgrade_to_head()
    typer.echo(f'DATABASE_URL="{url}"')


@app.command()
def backfill(
    start: str,
    end: str,
    every: int = 1,
    root: str = "data/events",
    min_value_usd: str = "50000",
    with_packets: bool = False,
    packet_root: str = "data/packets",
) -> None:
    """Build a history by ingesting and labeling a range of trading days.

    A day costs the same regardless of how many of its filings qualify, because
    each must be fetched to find out, so `--every` is the lever on how long a
    run takes. Sampling every Nth trading day spreads observations across months
    at the cost of a contiguous block, which matters because events filed on one
    day share a market factor that the sector benchmark only partly removes.

    Days already stored are skipped, so an interrupted run is simply restarted.
    With `--with-packets`, evidence packets are built for each day as it lands
    and a day counts as stored only once its packets exist.
    """
    days = sampled_trading_days(date.fromisoformat(start), date.fromisoformat(end), every)
    ingest_step, resolve_step = ingest_and_resolve(Path(root), Decimal(min_value_usd))

    typer.echo(f"{len(days)} trading days from {days[0]} to {days[-1]}" if days else "no days")

    from arbiter.ingestion.backfill import day_is_packed
    from arbiter.packets.pipeline import build_day as build_packet_day

    today = current_session_date()

    def fetch(
        symbols: Sequence[str], window_start: date, window_end: date
    ) -> dict[str, list[Bar]]:
        return daily_bars_tolerating_gaps(list(symbols), window_start, window_end, today)

    def pack(day: date) -> int:
        written, excluded = build_packet_day(
            day,
            Path(root),
            Path(packet_root),
            fetch_bars=fetch,
            benchmark_for=benchmark_for_issuer,
        )
        # `build_day` persists the exclusions beside the packets; this line only
        # surfaces them in the run's own output. Discarding the second element
        # here once meant a backfill reported a count and nothing else, leaving
        # the difference between the event store and the packet store
        # unexplained for every day it wrote.
        if excluded:
            typer.echo(f"  {day.isoformat()}: {len(excluded)} events excluded from packets")
        return written

    def stored_without_packets(day: date) -> bool:
        return day_is_stored(Path(root), day)

    def stored_with_packets(day: date) -> bool:
        # The resume rule tightens with the work: a day whose packet build
        # failed still has its events and labels, and the looser rule would call
        # it finished and skip it forever.
        return day_is_packed(Path(root), Path(packet_root), day)

    pack_step = pack if with_packets else None
    stored = stored_with_packets if with_packets else stored_without_packets

    summary = run_backfill(
        days,
        ingest=ingest_step,
        resolve=resolve_step,
        is_stored=stored,
        pack=pack_step,
        on_progress=typer.echo,
    )

    typer.echo(
        f"completed: {len(summary.completed)}  skipped: {len(summary.skipped)}  "
        f"failed: {len(summary.failed)}  pending: {len(summary.pending)}  "
        f"events: {summary.events}  labels: {summary.labels}"
        + (f"  packets: {summary.packets}" if with_packets else "")
    )
    for day, last_close in sorted(summary.pending.items()):
        typer.echo(
            f"  {day.isoformat()}: windows open until {last_close.isoformat()}; "
            "its packets wait with it; rerun after it",
            err=True,
        )
    for day, cause in sorted(summary.failed.items()):
        typer.echo(f"  {day.isoformat()}: {cause}", err=True)
