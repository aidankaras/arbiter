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


def test_a_training_fraction_outside_the_unit_interval_is_refused_before_reading(
    tmp_path: Path,
):
    """The same empty store is reported as empty when the fraction is valid.

    That pairing shows the refusal comes from the fraction, not from the store.
    """
    refused = runner.invoke(
        app, ["report", "--root", str(tmp_path), "--train-fraction", "-0.2"]
    )
    accepted = runner.invoke(
        app, ["report", "--root", str(tmp_path), "--train-fraction", "0.6"]
    )

    assert refused.exit_code == 2
    assert "strictly between 0 and 1" in refused.output
    assert accepted.exit_code == 1
    assert "no labelled" in accepted.output
