"""Tests for the two-sleeve policy builder.

The satellite sleeve is driven by a watchlist someone edits in a phone app, so
these tests are mostly about what happens when that list contains something
unexpected. Every case here is a way real money could be sent somewhere it
should not be.
"""

from __future__ import annotations

import pytest

from kronos_trader.allocation import (
    Holding,
    PortfolioState,
    plan_investment,
)
from kronos_trader.invest_config import (
    CORE_TARGETS,
    CORE_WEIGHT,
    SATELLITE_WEIGHT,
    UNMANAGED,
    build_policy,
)


def test_core_targets_sum_to_core_weight():
    assert sum(CORE_TARGETS.values()) == pytest.approx(CORE_WEIGHT, abs=1e-9)


def test_empty_watchlist_scales_core_to_one():
    policy, notes = build_policy([])
    assert sum(policy.targets.values()) == pytest.approx(1.0, abs=1e-9)
    assert set(policy.targets) == set(CORE_TARGETS)
    # Relative proportions inside the core are preserved.
    assert policy.targets["VOO"] == pytest.approx(
        CORE_TARGETS["VOO"] / CORE_WEIGHT, abs=1e-9
    )
    assert any("no usable satellites" in n for n in notes)


def test_satellites_split_the_sleeve_evenly():
    policy, _ = build_policy(["PLTR", "MU", "XOM", "CVX"])
    assert sum(policy.targets.values()) == pytest.approx(1.0, abs=1e-9)
    for sym in ("PLTR", "MU", "XOM", "CVX"):
        assert policy.targets[sym] == pytest.approx(SATELLITE_WEIGHT / 4, abs=1e-9)
    # Core weights are untouched when satellites are present.
    assert policy.targets["VOO"] == pytest.approx(CORE_TARGETS["VOO"], abs=1e-9)


def test_unmanaged_symbol_on_the_watchlist_is_never_given_a_target():
    """Dropping F on the watchlist must not make the bot buy the owner's stock."""
    victim = sorted(UNMANAGED)[0]
    policy, notes = build_policy(["PLTR", victim])
    assert victim not in policy.targets
    assert policy.targets["PLTR"] == pytest.approx(SATELLITE_WEIGHT, abs=1e-9)
    assert any(victim in n and "unmanaged" in n for n in notes)


def test_core_symbol_on_the_watchlist_is_not_double_counted():
    policy, notes = build_policy(["VOO", "PLTR"])
    assert policy.targets["VOO"] == pytest.approx(CORE_TARGETS["VOO"], abs=1e-9)
    assert policy.targets["PLTR"] == pytest.approx(SATELLITE_WEIGHT, abs=1e-9)
    assert sum(policy.targets.values()) == pytest.approx(1.0, abs=1e-9)
    assert any("VOO" in n and "core" in n for n in notes)


def test_unpriced_satellites_are_excluded_not_fatal():
    policy, notes = build_policy(["PLTR", "DELISTED"], priced={"PLTR"})
    assert "DELISTED" not in policy.targets
    assert policy.targets["PLTR"] == pytest.approx(SATELLITE_WEIGHT, abs=1e-9)
    assert any("DELISTED" in n and "price" in n for n in notes)


def test_duplicates_and_casing_are_normalised():
    policy, _ = build_policy(["pltr", "PLTR", " pltr ", "mu"])
    assert policy.targets["PLTR"] == pytest.approx(SATELLITE_WEIGHT / 2, abs=1e-9)
    assert policy.targets["MU"] == pytest.approx(SATELLITE_WEIGHT / 2, abs=1e-9)


@pytest.mark.parametrize("n", [1, 7, 23, 101])
def test_weights_sum_to_one_for_any_list_length(n):
    policy, _ = build_policy([f"S{i}" for i in range(n)])
    assert sum(policy.targets.values()) == pytest.approx(1.0, abs=1e-9)


def test_buys_never_exceed_deployable_cash_across_many_holdings():
    """The rounding bug: per-order rounding used to sum above the cash on hand.

    With 29 targets the overshoot reached eight cents, which is enough to fail
    the caller's buys_within_cash check and halt an otherwise correct run.
    """
    symbols = [f"S{i}" for i in range(23)]
    policy, _ = build_policy(symbols)
    prices = {s: 10.0 for s in policy.targets}
    state = PortfolioState(holdings={}, cash=2000.0)

    plan = plan_investment(state, policy, prices)

    deployable = 2000.0 - policy.cash_buffer
    assert plan.total_buys <= deployable + 1e-9
    # And it should still deploy nearly all of it, not bail out early.
    assert plan.total_buys > deployable - 1.0


def test_amounts_are_whole_cents():
    policy, _ = build_policy(["A", "B", "C"])
    prices = {s: 7.77 for s in policy.targets}
    state = PortfolioState(holdings={"VOO": Holding("VOO", 1.0, 7.77)}, cash=933.13)

    plan = plan_investment(state, policy, prices)

    for t in plan.trades:
        assert round(t.amount, 2) == t.amount, f"{t.symbol} amount {t.amount!r}"
