"""Tests for universe screening and the live configuration guards."""

from __future__ import annotations

import pytest

from kronos_trader.config import ACCOUNT_CASH, LiveConfig
from kronos_trader.pricing import OptionQuote
from kronos_trader.risk import RiskLimits
from kronos_trader.universe import ScreenCriteria, affordable_quotes


def q(strike, ask, *, right="call", oi=1000, dte=21, spread=0.02) -> OptionQuote:
    return OptionQuote(
        symbol="F", strike=strike, expiry_years=dte / 365.0, right=right,
        bid=max(0.0, ask - spread), ask=ask, open_interest=oi,
    )


SPOT = 12.0
CRIT = ScreenCriteria(budget=75.0)


# --- affordability ------------------------------------------------------------


def test_contract_over_budget_is_dropped():
    assert affordable_quotes([q(12.5, 0.90)], SPOT, CRIT) == []  # $90 > $75
    assert len(affordable_quotes([q(12.5, 0.70)], SPOT, CRIT)) == 1  # $70 ok


def test_far_otm_lottery_ticket_is_dropped():
    """The screen that stops a small account drifting into junk strikes."""
    assert affordable_quotes([q(20.0, 0.05)], SPOT, CRIT) == []  # 67% OTM
    assert affordable_quotes([q(13.0, 0.40)], SPOT, CRIT)        # ~8% OTM, fine


def test_expiry_window_enforced():
    assert affordable_quotes([q(12.5, 0.40, dte=2)], SPOT, CRIT) == []
    assert affordable_quotes([q(12.5, 0.40, dte=200)], SPOT, CRIT) == []
    assert affordable_quotes([q(12.5, 0.40, dte=21)], SPOT, CRIT)


def test_illiquid_and_wide_contracts_dropped():
    assert affordable_quotes([q(12.5, 0.40, oi=5)], SPOT, CRIT) == []
    assert affordable_quotes([q(12.5, 0.40, spread=0.30)], SPOT, CRIT) == []


def test_puts_use_the_mirrored_moneyness_rule():
    # A put is OTM below spot.
    assert affordable_quotes([q(4.0, 0.05, right="put")], SPOT, CRIT) == []
    assert affordable_quotes([q(11.2, 0.40, right="put")], SPOT, CRIT)


def test_zero_ask_contract_is_unbuyable():
    assert affordable_quotes([q(12.5, 0.0)], SPOT, CRIT) == []


# --- live config guards -------------------------------------------------------


def test_default_live_config_fits_the_account():
    cfg = LiveConfig()
    assert cfg.limits.max_risk_per_trade == 75.0
    assert cfg.limits.max_risk_per_trade <= ACCOUNT_CASH
    assert cfg.screen.budget <= cfg.limits.max_risk_per_trade


def test_risk_cap_above_account_cash_is_rejected():
    """A $100 cap on an $80 account can never fill; refuse it at construction."""
    with pytest.raises(ValueError, match="exceeds account cash"):
        LiveConfig(limits=RiskLimits(max_risk_per_trade=100.0))


def test_screen_budget_cannot_exceed_risk_cap():
    with pytest.raises(ValueError, match="screen budget must not exceed"):
        LiveConfig(screen=ScreenCriteria(budget=76.0))


def test_seeds_are_the_agreed_starters():
    assert LiveConfig().seeds == ("F", "SOFI", "INTC", "T", "PFE")
    assert LiveConfig().screen_own_universe is True
