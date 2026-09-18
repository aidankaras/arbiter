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
from arbiter.config import Settings, get_settings
from arbiter.evaluation.resolution import HORIZONS, resolve_stored_day
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
