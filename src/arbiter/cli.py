"""Command-line entry point.

Thin wrapper over the library. Every subcommand is a small adapter around a
function that is independently testable without the CLI.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

import typer
from pydantic import ValidationError

from arbiter import __version__
from arbiter.arms.evaluate import (
    TRAIN_FRACTION,
    evaluate_baseline,
    load_days,
    stored_days,
)
from arbiter.arms.features import UnsupportedDomainError
from arbiter.config import Settings, get_settings
from arbiter.evaluation.report import render_baseline_report
from arbiter.evaluation.resolution import HORIZONS, resolve_stored_day
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
    labeled is always at least one session behind ingestion, and a day whose
    window has not closed yields nothing rather than a partial measurement.

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

    written = resolve_stored_day(
        day=date.fromisoformat(day),
        domain=domain,
        root=Path(root),
        fetch_bars=fetch,
        benchmark_for=benchmark_for_issuer,
        today=today,
        ticker_for=issuer_ticker,
    )
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


@app.command()
def report(
    domain: str = "insider",
    root: str = "data/events",
    out: str = "reports",
    train_fraction: float = TRAIN_FRACTION,
) -> None:
    """Fit the baseline arm on the stored history and write its measurement.

    Reads every day the store holds labels for, fits on the earlier fraction of
    them, and scores the rest. The report is written to `reports/` so that what
    was claimed at a given commit stays inspectable; recomputing against more
    data produces a new report rather than an edit to the old one.
    """
    store = Path(root)
    days = stored_days(store, domain)
    if not days:
        typer.echo(f"no labelled {domain} days under {root}; run `arbiter backfill`", err=True)
        raise typer.Exit(code=1)

    try:
        # `load_days` is what raises: features are built as each day is read,
        # so the call has to sit inside the handler rather than before it.
        loaded = load_days(store, domain, days)
        evaluation = evaluate_baseline(loaded, train_fraction)
    except UnsupportedDomainError as exc:
        typer.echo(f"the baseline arm cannot score '{domain}': {exc}", err=True)
        raise typer.Exit(code=2) from None

    destination = Path(out) / f"baseline-{domain}.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_baseline_report(evaluation, generated_on=current_session_date()),
        encoding="utf-8",
    )

    typer.echo(
        f"{destination}: IC {evaluation.ic.mean:+.4f} "
        f"(se {evaluation.ic.standard_error:.4f}, t {evaluation.ic.t_statistic:+.2f}) "
        f"over {evaluation.ic.days} days, {evaluation.test_issuer_days:,} issuer-days"
    )


@app.command()
def backfill(
    start: str,
    end: str,
    every: int = 1,
    root: str = "data/events",
    min_value_usd: str = "50000",
) -> None:
    """Build a history by ingesting and labeling a range of trading days.

    A day costs the same regardless of how many of its filings qualify, because
    each must be fetched to find out, so `--every` is the lever on how long a
    run takes. Sampling every Nth trading day spreads observations across months
    at the cost of a contiguous block, which matters because events filed on one
    day share a market factor that the sector benchmark only partly removes.

    Days already stored are skipped, so an interrupted run is simply restarted.
    """
    days = sampled_trading_days(date.fromisoformat(start), date.fromisoformat(end), every)
    ingest_step, resolve_step = ingest_and_resolve(Path(root), Decimal(min_value_usd))

    typer.echo(f"{len(days)} trading days from {days[0]} to {days[-1]}" if days else "no days")

    summary = run_backfill(
        days,
        ingest=ingest_step,
        resolve=resolve_step,
        is_stored=lambda day: day_is_stored(Path(root), day),
        on_progress=typer.echo,
    )

    typer.echo(
        f"completed: {len(summary.completed)}  skipped: {len(summary.skipped)}  "
        f"failed: {len(summary.failed)}  events: {summary.events}  labels: {summary.labels}"
    )
    for day, cause in sorted(summary.failed.items()):
        typer.echo(f"  {day.isoformat()}: {cause}", err=True)
