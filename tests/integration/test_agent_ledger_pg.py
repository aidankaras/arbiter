from decimal import Decimal

import pytest
from sqlalchemy import select

from arbiter.arms.agent.ledger import PgLedger
from arbiter.db.models import LlmCall, Prediction
from arbiter.db.session import session_scope
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


def test_an_exclusion_is_written_once_per_version(migrated_database: str) -> None:
    contract.check_an_exclusion_is_written_once_per_version(PgLedger(), "pg-exclude-once")


def test_a_forecast_and_its_calls_join_on_the_evidence_they_name(
    migrated_database: str,
) -> None:
    # Cost per decision is a join: each call carries the forecast group's hash,
    # and each prediction carries its packet's hash and its group's hash.
    ledger, forecast = PgLedger(), contract.row("j-1")
    assert forecast.packet_hash != forecast.issuer_day_hash  # the swap must be visible
    ledger.write_forecast("claude-sonnet-5", "pg-join", [forecast])
    ledger.record_call(contract.payload(), "pg-join-run", forecast.issuer_day_hash)
    with session_scope() as session:
        stored = session.execute(
            select(Prediction).where(Prediction.prompt_version == "pg-join")
        ).scalar_one()
        assert stored.packet_hash == forecast.packet_hash
        assert stored.class_probs["issuer_day_hash"] == forecast.issuer_day_hash
        joined = (
            session.execute(
                select(LlmCall).where(
                    LlmCall.packet_hash == stored.class_probs["issuer_day_hash"]
                )
            )
            .scalars()
            .all()
        )
    assert len(joined) == len(contract.payload()["modelUsage"])


def test_another_arms_predictions_are_invisible_to_the_agent(migrated_database: str) -> None:
    with session_scope() as session:
        session.add(
            Prediction(
                event_id="arm-1",
                arm="baseline",
                model_id="logistic",
                prompt_version="pg-arms",
                packet_hash=contract.row("arm-1").packet_hash,
                horizon_days=5,
                class_probs={"up": 0.6},
                point_estimate=Decimal("0.6"),
            )
        )
    ledger = PgLedger()
    assert ledger.predicted("pg-arms", ["arm-1"]) == set()
    assert ledger.forecasts("pg-arms") == {}


def test_excluded_events_are_found_by_filing(migrated_database: str) -> None:
    contract.check_excluded_events_are_found_by_filing(PgLedger(), "pg-excluded-events")
