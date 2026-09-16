"""Command-line entry point.

Thin wrapper over the library. Every subcommand is a small adapter around a
function that is independently testable without the CLI.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import typer
from pydantic import ValidationError

from arbiter import __version__
from arbiter.config import Settings, get_settings
from arbiter.ingestion.pipeline import ingest_day

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
