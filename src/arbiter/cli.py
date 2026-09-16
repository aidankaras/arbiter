"""Command-line entry point.

Thin wrapper over the library. Every subcommand is a small adapter around a
function that is independently testable without the CLI.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import typer

from arbiter import __version__
from arbiter.config import get_settings
from arbiter.ingestion.pipeline import ingest_day

app = typer.Typer(
    name="arbiter",
    help="Event-driven equity forecasting from SEC filings.",
    no_args_is_help=True,
    add_completion=False,
)


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
    settings = get_settings()
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
