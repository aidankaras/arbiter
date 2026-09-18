"""The resolve subcommand's argument handling.

The command itself reaches the network and is covered by the live suites. What
is tested here is the part that must fail before any of that happens: a domain
the project does not measure has to be refused at the boundary, because the next
step reads that domain's partition and would report a missing day instead.
"""

from pathlib import Path

from typer.testing import CliRunner

from arbiter.cli import app
from arbiter.evaluation.resolution import HORIZONS

runner = CliRunner()


def test_an_unknown_domain_is_refused_before_anything_is_read(tmp_path: Path):
    result = runner.invoke(app, ["resolve", "2026-08-03", "redflags", "--root", str(tmp_path)])

    assert result.exit_code == 2
    assert "unknown domain 'redflags'" in result.output
    assert not list(tmp_path.iterdir()), "a rejected domain must not create anything on disk"


def test_the_refusal_names_the_domains_that_do_exist():
    """A typo is the likely cause, so the message has to be actionable."""
    result = runner.invoke(app, ["resolve", "2026-08-03", "earnings"])

    for domain in HORIZONS:
        assert domain in result.output
