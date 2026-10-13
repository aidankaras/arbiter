import pytest

from arbiter.arms.agent.ledger import PgLedger
from tests import agent_ledger_contract as contract

pytestmark = pytest.mark.integration


def test_written_events_read_back_as_predicted(migrated_database: str) -> None:
    contract.check_written_events_read_back_as_predicted(PgLedger(), "pg-read-back")


def test_a_forecast_with_a_duplicate_writes_nothing(migrated_database: str) -> None:
    contract.check_a_forecast_with_a_duplicate_writes_nothing(PgLedger(), "pg-duplicate")


def test_exclusions_are_per_forecast_group(migrated_database: str) -> None:
    contract.check_exclusions_are_per_forecast_group(PgLedger(), "pg-exclusions")


def test_usage_sums_every_token_and_cost(migrated_database: str) -> None:
    contract.check_usage_sums_every_token_and_cost(PgLedger(), "pg-usage-run")


def test_forecasts_and_exclusions_read_back_by_version(migrated_database: str) -> None:
    contract.check_forecasts_and_exclusions_read_back_by_version(
        PgLedger(), "pg-read-by-version"
    )
