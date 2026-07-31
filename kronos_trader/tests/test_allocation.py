"""Tests for the target-allocation engine."""

from __future__ import annotations

import pytest

from kronos_trader.allocation import (
    AllocationError,
    AllocationPolicy,
    Holding,
    PortfolioState,
    plan_investment,
)

PRICES = {"VOO": 500.0, "VXUS": 60.0, "BND": 72.0}
POLICY = AllocationPolicy(targets={"VOO": 0.6, "VXUS": 0.25, "BND": 0.15})


def test_weights_must_sum_to_one():
    with pytest.raises(AllocationError, match="must sum to 1.0"):
        AllocationPolicy(targets={"VOO": 0.6, "BND": 0.3})


def test_rejects_nonpositive_target():
    with pytest.raises(AllocationError, match="must be positive"):
        AllocationPolicy(targets={"VOO": 1.2, "BND": -0.2})


def test_empty_account_buys_in_target_proportions():
    state = PortfolioState(cash=1000.0)
    plan = plan_investment(state, POLICY, PRICES)
    by = {t.symbol: t.amount for t in plan.trades}
    assert all(t.side == "buy" for t in plan.trades)
    assert by["VOO"] == pytest.approx(600.0, abs=0.01)
    assert by["VXUS"] == pytest.approx(250.0, abs=0.01)
    assert by["BND"] == pytest.approx(150.0, abs=0.01)
    assert plan.total_buys == pytest.approx(1000.0, abs=0.02)


def test_contribution_goes_to_most_underweight_first():
    """The core routing rule: new money corrects drift without selling."""
    state = PortfolioState(
        holdings={
            "VOO": Holding("VOO", 1.8, 500.0),   # $900, way over
            "VXUS": Holding("VXUS", 0.5, 60.0),  # $30, way under
        },
        cash=100.0,
    )
    plan = plan_investment(state, POLICY, PRICES)
    buys = [t for t in plan.trades if t.side == "buy"]
    assert buys, "should deploy the cash"
    # VXUS and BND are the underweights; VOO must not be bought.
    assert "VOO" not in [t.symbol for t in buys]


def test_no_selling_merely_because_a_holding_is_over_target():
    """Small drift must not trigger a sale. Trading costs money."""
    investable = 1000.0
    state = PortfolioState(
        holdings={
            "VOO": Holding("VOO", 1.24, 500.0),  # 62%, 2pp over -> inside band
            "VXUS": Holding("VXUS", 4.0, 60.0),
            "BND": Holding("BND", 2.0, 72.0),
        },
        cash=0.0,
    )
    plan = plan_investment(state, POLICY, PRICES)
    assert [t for t in plan.trades if t.side == "sell"] == []


def test_sell_only_on_hard_band_breach():
    state = PortfolioState(
        holdings={
            "VOO": Holding("VOO", 3.0, 500.0),   # $1500 of $1700 -> 88%
            "VXUS": Holding("VXUS", 2.0, 60.0),
            "BND": Holding("BND", 1.0, 72.0),
        },
        cash=0.0,
    )
    plan = plan_investment(state, POLICY, PRICES)
    sells = [t for t in plan.trades if t.side == "sell"]
    assert len(sells) == 1 and sells[0].symbol == "VOO"
    assert "over the 5% band" in sells[0].reason


def test_price_movement_alone_never_produces_a_sell():
    """A crash must not generate sell orders -- only band breaches do."""
    state = PortfolioState(
        holdings={
            "VOO": Holding("VOO", 1.2, 250.0),   # price halved
            "VXUS": Holding("VXUS", 4.2, 30.0),
            "BND": Holding("BND", 2.1, 36.0),
        },
        cash=0.0,
    )
    crashed = {"VOO": 250.0, "VXUS": 30.0, "BND": 36.0}
    plan = plan_investment(state, POLICY, crashed)
    assert [t for t in plan.trades if t.side == "sell"] == []


def test_dust_below_minimum_is_not_traded():
    state = PortfolioState(cash=3.0)
    plan = plan_investment(state, POLICY, PRICES)
    assert plan.trades == []
    assert any("below the" in n for n in plan.notes)


def test_cash_buffer_is_respected():
    state = PortfolioState(cash=100.0)
    policy = AllocationPolicy(targets={"VOO": 1.0}, cash_buffer=40.0)
    plan = plan_investment(state, policy, PRICES)
    assert plan.total_buys == pytest.approx(60.0, abs=0.01)


def test_unmanaged_positions_are_never_traded():
    """Legacy holdings the user opened stay untouched."""
    state = PortfolioState(
        holdings={
            "VOO": Holding("VOO", 1.0, 500.0),
            "SOFI": Holding("SOFI", 1.88, 17.0),
            "F": Holding("F", 0.2, 15.5),
        },
        cash=100.0,
        unmanaged={"SOFI", "F"},
    )
    plan = plan_investment(state, POLICY, PRICES)
    traded = {t.symbol for t in plan.trades}
    assert "SOFI" not in traded and "F" not in traded
    assert any("never traded" in n for n in plan.notes)


def test_missing_price_aborts_rather_than_guessing():
    state = PortfolioState(cash=1000.0)
    plan = plan_investment(state, POLICY, {"VOO": 500.0, "VXUS": 60.0})
    assert plan.trades == []
    assert any("no usable price" in n for n in plan.notes)


def test_zero_price_treated_as_missing():
    state = PortfolioState(cash=1000.0)
    plan = plan_investment(state, POLICY, {"VOO": 500.0, "VXUS": 60.0, "BND": 0.0})
    assert plan.trades == []


def test_plan_is_deterministic():
    state = PortfolioState(holdings={"VOO": Holding("VOO", 0.5, 500.0)}, cash=250.0)
    a = plan_investment(state, POLICY, PRICES)
    b = plan_investment(state, POLICY, PRICES)
    assert [(t.symbol, t.side, t.amount) for t in a.trades] == \
           [(t.symbol, t.side, t.amount) for t in b.trades]


def test_buys_never_exceed_available_cash():
    state = PortfolioState(cash=50.0)
    plan = plan_investment(state, POLICY, PRICES)
    assert plan.total_buys <= 50.0 + 1e-6
