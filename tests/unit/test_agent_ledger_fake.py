from tests import agent_ledger_contract as contract
from tests.agent_fakes import FakeLedger


def test_written_events_read_back_as_predicted() -> None:
    contract.check_written_events_read_back_as_predicted(FakeLedger(), "v1")


def test_a_forecast_with_a_duplicate_writes_nothing() -> None:
    contract.check_a_forecast_with_a_duplicate_writes_nothing(FakeLedger(), "v1")


def test_exclusions_are_per_forecast_group() -> None:
    contract.check_exclusions_are_per_forecast_group(FakeLedger(), "v1")


def test_usage_sums_every_token_and_cost() -> None:
    contract.check_usage_sums_every_token_and_cost(FakeLedger(), "run-1")


def test_forecasts_and_exclusions_read_back_by_version() -> None:
    contract.check_forecasts_and_exclusions_read_back_by_version(FakeLedger(), "v1")
