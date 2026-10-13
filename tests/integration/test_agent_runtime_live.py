"""One real call through the runtime, as the run will make it. Uses the subscription."""

import pytest

from arbiter.arms.agent.prompt import system_prompt
from arbiter.arms.agent.runtime import ClaudeCodeRuntime, Status

pytestmark = pytest.mark.integration

MESSAGE = "\n".join(
    [
        "<untrusted-data>",
        "company: Example Corp",
        "ticker: EXMP",
        "sector_etf: XLK",
        "",
        "filings:",
        "- accepted 2026-06-15T20:00:00+00:00 | insider Jane Roe"
        " | position Chief Financial Officer | code P | shares 10000 | price 12"
        " | value_usd 120000 | remaining_shares 50000 | 10b5-1 False",
        "",
        "bars (session,open,high,low,close,volume):",
        "2026-06-15,12,12.4,11.8,12.1,800000",
        "</untrusted-data>",
    ]
)


def test_a_real_call_runs_isolated_and_returns_a_probability() -> None:
    sample = ClaudeCodeRuntime().sample(system_prompt(), MESSAGE)
    assert sample.status is Status.OK, sample.detail
    assert sample.p_up is not None and 0.0 <= sample.p_up <= 1.0
    assert "seven_day" in sample.utilization
