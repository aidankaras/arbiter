"""Runtime configuration, validated once at startup.

All configuration arrives through the environment. Validation happens eagerly so
a missing or malformed value fails immediately with the offending variable named,
rather than surfacing partway through a batch that has already spent money.
"""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum
from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The SEC requires a descriptive User-Agent carrying contact information on every
# request. Requests without one are throttled or refused outright, so this is
# validated at startup rather than discovered on the first fetch.
_SEC_USER_AGENT_PATTERN = re.compile(r"^\S+\s+\S+@\S+\.\S+$")


class Environment(StrEnum):
    """Deployment environment, which gates destructive and live-trading paths."""

    DEVELOPMENT = "development"
    CI = "ci"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Validated application configuration.

    Constructed once per process through `get_settings`. Secrets are held as
    `SecretStr` so that an accidental log or repr of this object cannot leak
    them; call `.get_secret_value()` explicitly at the point of use.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        frozen=True,
    )

    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"

    sec_user_agent: str = Field(
        description="Contact string sent to EDGAR, formatted as 'name email@host'.",
    )
    database_url: str = Field(
        default="postgresql+psycopg://arbiter:arbiter@localhost:5432/arbiter",
        description="SQLAlchemy URL for the Postgres instance holding all state.",
    )

    anthropic_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"

    alpaca_api_key: SecretStr | None = None
    alpaca_secret_key: SecretStr | None = None
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Broker endpoint. Must remain a paper endpoint.",
    )

    daily_spend_ceiling_usd: Annotated[Decimal, Field(gt=0)] = Decimal("4.00")
    per_run_token_ceiling: Annotated[int, Field(gt=0)] = 250_000

    @field_validator("sec_user_agent")
    @classmethod
    def _validate_sec_user_agent(cls, value: str) -> str:
        """Reject a User-Agent EDGAR would throttle.

        Raises:
            ValueError: the value does not contain a name and an email address.
        """
        if not _SEC_USER_AGENT_PATTERN.match(value.strip()):
            msg = (
                "SEC_USER_AGENT must be a name followed by a contact email, "
                "for example 'jane-doe jane@example.com'. EDGAR throttles or "
                "refuses requests without one."
            )
            raise ValueError(msg)
        return value.strip()

    @field_validator("alpaca_base_url")
    @classmethod
    def _reject_live_broker_endpoint(cls, value: str) -> str:
        """Refuse any broker endpoint that is not the paper-trading endpoint.

        This project trades no real capital. Enforcing that here means a
        misconfigured environment fails at startup rather than at order
        submission.

        Raises:
            ValueError: the URL does not point at a paper-trading endpoint.
        """
        if "paper-api" not in value:
            msg = (
                f"Refusing non-paper broker endpoint {value!r}. This system is "
                "designed and tested for paper trading only."
            )
            raise ValueError(msg)
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, constructing them on first call.

    Cached so that validation runs once and every caller observes the same
    configuration for the life of the process.
    """
    return Settings()  # pyright: ignore[reportCallIssue]  # values come from env
