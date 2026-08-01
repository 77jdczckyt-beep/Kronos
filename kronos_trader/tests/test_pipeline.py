"""Tests for the forecast -> signal -> risk pipeline.

These run without model weights by using `BootstrapForecaster`, which is also
the honest baseline the real model has to beat.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from kronos_trader.forecast import BootstrapForecaster, PathEnsemble
from kronos_trader.pricing import OptionQuote, bs_price, implied_vol
from kronos_trader.risk import (
    ExitRules,
    Position,
    RiskBook,
    RiskLimits,
    RiskViolation,
)
from kronos_trader.signals import SignalConfig, best_signal, evaluate_contract

ALI_CSV = "finetune_csv/data/HK_ali_09988_kline_5min_all.csv"


@pytest.fixture(scope="module")
def ali() -> pd.DataFrame:
    df = pd.read_csv(ALI_CSV)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.set_index("timestamps")


@pytest.fixture(scope="module")
def ensemble(ali: pd.DataFrame) -> PathEnsemble:
    hist = ali.iloc[-600:-100]
    horizon = ali.index[-100:-20]
    return BootstrapForecaster(seed=7).forecast("09988", hist, horizon, n_paths=512)


# --- the central design claim -------------------------------------------------


def test_averaging_paths_destroys_dispersion(ali: pd.DataFrame):
    """Why we never use KronosPredictor.predict for signals.

    `predict` averages its sampled paths before returning them. Averaging
    preserves whatever drift the paths share, but it cancels the independent
    noise -- so the dispersion of an averaged forecast shrinks by ~sqrt(N)
    while the central tendency survives.

    That is precisely the wrong trade-off for options. A call is priced on the
    *spread* of terminal outcomes, not on the centre. Collapsing the ensemble
    keeps the part that is nearly worthless for pricing and throws away the
    part that determines the premium.

    Here we measure it directly: draw many independent ensembles of size N and
    compare the spread of their mean-path terminals against the spread of
    individual path terminals.
    """
    hist = ali.iloc[-600:-100]
    horizon = ali.index[-100:-20]
    n = 32

    single_path_terminals = []
    averaged_terminals = []
    for seed in range(40):
        ens = BootstrapForecaster(seed=seed).forecast("09988", hist, horizon, n_paths=n)
        single_path_terminals.append(ens.closes[0, -1])
        averaged_terminals.append(ens.closes.mean(axis=0)[-1])

    spread_single = float(np.std(single_path_terminals, ddof=1))
    spread_averaged = float(np.std(averaged_terminals, ddof=1))

    assert spread_averaged < spread_single, (
        "averaging must reduce dispersion"
    )
    # The sqrt(N) law predicts a reduction near sqrt(32) ~= 5.7x. Allow a wide
    # band -- block bootstrapping induces mild correlation between paths -- but
    # the effect must be unmistakably large.
    reduction = spread_single / spread_averaged
    assert reduction > 2.5, (
        f"expected a large dispersion collapse, got {reduction:.1f}x "
        f"(single {spread_single:.3f}, averaged {spread_averaged:.3f})"
    )


def test_ensemble_rejects_degenerate_input(ali: pd.DataFrame):
    with pytest.raises(ValueError, match="not an ensemble"):
        PathEnsemble(
            symbol="X",
            spot=100.0,
            closes=np.array([[100.0, 101.0]]),
            highs=np.array([[100.0, 101.0]]),
            lows=np.array([[100.0, 101.0]]),
            timestamps=pd.DatetimeIndex(ali.index[:2]),
        )


def test_forecaster_refuses_single_path(ali: pd.DataFrame):
    hist = ali.iloc[-300:-100]
    horizon = ali.index[-100:-50]
    with pytest.raises(ValueError, match="n_paths must be >= 2"):
        BootstrapForecaster(seed=1).forecast("09988", hist, horizon, n_paths=1)


# --- probabilities ------------------------------------------------------------


def test_probabilities_are_coherent(ensemble: PathEnsemble):
    spot = ensemble.spot
    assert 0.0 <= ensemble.prob_above(spot) <= 1.0
    # Touching a level is always at least as likely as ending beyond it.
    up = spot * 1.02
    assert ensemble.prob_touch_above(up) >= ensemble.prob_above(up)
    down = spot * 0.98
    assert ensemble.prob_touch_below(down) >= ensemble.prob_below(down)
    # Far strikes are monotonically less likely.
    assert ensemble.prob_above(spot * 1.10) <= ensemble.prob_above(spot * 1.02)


# --- pricing ------------------------------------------------------------------


def test_bs_call_put_parity():
    spot, strike, t, vol = 100.0, 105.0, 0.25, 0.30
    call = bs_price(spot, strike, t, vol, "call")
    put = bs_price(spot, strike, t, vol, "put")
    assert call - put == pytest.approx(spot - strike, abs=1e-6)


def test_implied_vol_roundtrip():
    spot, strike, t, vol = 100.0, 103.0, 0.1, 0.42
    price = bs_price(spot, strike, t, vol, "call")
    assert implied_vol(price, spot, strike, t, "call") == pytest.approx(vol, abs=1e-4)


def test_implied_vol_returns_nan_outside_arbitrage_band():
    # A call cannot be worth more than the underlying.
    assert np.isnan(implied_vol(150.0, 100.0, 90.0, 0.2, "call"))


def test_spread_frac_handles_zero_mid():
    q = OptionQuote("X", 100.0, 0.1, "call", bid=0.0, ask=0.0)
    assert q.spread_frac == float("inf")


def test_crossed_quote_rejected():
    with pytest.raises(ValueError, match="crossed quote"):
        OptionQuote("X", 100.0, 0.1, "call", bid=2.0, ask=1.0)


# --- signals ------------------------------------------------------------------


def test_overpriced_contract_is_rejected(ensemble: PathEnsemble):
    """A contract priced far above modelled value must not qualify."""
    q = OptionQuote(
        "09988", strike=ensemble.spot * 1.01, expiry_years=0.05,
        right="call", bid=45.0, ask=46.0, open_interest=5000,
    )
    sig = evaluate_contract(ensemble, q)
    assert not sig.accepted
    assert "edge ratio" in sig.reason or "P(ITM)" in sig.reason


def test_wide_spread_is_rejected(ensemble: PathEnsemble):
    q = OptionQuote(
        "09988", strike=ensemble.spot * 1.01, expiry_years=0.05,
        right="call", bid=1.0, ask=3.0, open_interest=5000,
    )
    sig = evaluate_contract(ensemble, q)
    assert not sig.accepted
    assert "spread" in sig.reason


def test_illiquid_contract_is_rejected(ensemble: PathEnsemble):
    q = OptionQuote(
        "09988", strike=ensemble.spot * 1.01, expiry_years=0.05,
        right="call", bid=1.00, ask=1.05, open_interest=3,
    )
    sig = evaluate_contract(ensemble, q)
    assert not sig.accepted
    assert "open interest" in sig.reason


def test_fairly_priced_option_is_rejected(ensemble: PathEnsemble):
    """The key economic test.

    Price a contract at exactly the model's own expected payoff. It carries
    zero edge, so it must be refused -- otherwise the bot would happily churn
    fair-value trades and bleed out through spread and commission.
    """
    strike = ensemble.spot * 1.01
    fair = float(np.mean(np.maximum(ensemble.terminal - strike, 0.0)))
    q = OptionQuote(
        "09988", strike=strike, expiry_years=0.05, right="call",
        bid=fair * 0.99, ask=fair, open_interest=5000,
    )
    sig = evaluate_contract(ensemble, q)
    assert not sig.accepted, f"fair-value contract must be rejected, got: {sig.reason}"


def test_high_conviction_directional_signal_is_accepted(ensemble: PathEnsemble):
    """Regression test for the filter bug the oracle experiment exposed.

    An earlier version capped `prob_itm` at 70% and required forecast vol to
    exceed implied vol. Together those vetoed every high-conviction directional
    forecast -- a perfect oracle, which produces P(ITM) near 100% and no vol
    premium at all, was rejected on 399 of 400 decisions.

    The economically correct test is modelled value against market price. A
    cheap contract the model is confident about must qualify.
    """
    strike = ensemble.spot * 1.005
    # Confident forecast: shift every path decisively above the strike.
    shifted = ensemble.closes * (strike * 1.03 / ensemble.terminal.mean())
    confident = PathEnsemble(
        symbol=ensemble.symbol,
        spot=ensemble.spot,
        closes=shifted,
        highs=shifted,
        lows=shifted,
        timestamps=ensemble.timestamps,
    )
    fair = float(np.mean(np.maximum(confident.terminal - strike, 0.0)))
    # Market offers it at half the model's expected payoff -> 2x edge.
    q = OptionQuote(
        "09988", strike=strike, expiry_years=0.05, right="call",
        bid=fair * 0.49, ask=fair * 0.50, open_interest=5000,
    )
    sig = evaluate_contract(confident, q)
    assert sig.accepted, f"high-conviction cheap contract was rejected: {sig.reason}"
    assert sig.edge_ratio > 1.5


def test_deep_itm_contract_is_rejected_on_moneyness(ensemble: PathEnsemble):
    """Moneyness is judged on the contract, not on model confidence."""
    strike = ensemble.spot * 0.90  # 10% ITM call
    fair = float(np.mean(np.maximum(ensemble.terminal - strike, 0.0)))
    q = OptionQuote(
        "09988", strike=strike, expiry_years=0.05, right="call",
        bid=fair * 0.49, ask=fair * 0.50, open_interest=5000,
    )
    sig = evaluate_contract(ensemble, q)
    assert not sig.accepted
    assert "in the money" in sig.reason


def test_best_signal_returns_none_when_nothing_qualifies(ensemble: PathEnsemble):
    quotes = [
        OptionQuote("09988", ensemble.spot * m, 0.05, "call", bid=20.0, ask=21.0, open_interest=1000)
        for m in (1.01, 1.05, 1.10)
    ]
    best, scored = best_signal(ensemble, quotes)
    assert best is None
    assert len(scored) == 3
    assert all(not s.accepted for s in scored)


# --- risk ---------------------------------------------------------------------


def _pos(symbol="AAA", price=1.0, contracts=1, days=30) -> Position:
    return Position(
        symbol=symbol, strike=100.0, right="call",
        expiry=date(2026, 1, 1) + timedelta(days=days),
        contracts=contracts, entry_price=price,
        opened_at=datetime(2026, 1, 1, 10, 0),
    )


def test_per_trade_cap_enforced():
    book = RiskBook(limits=RiskLimits(max_risk_per_trade=150.0))
    now = datetime(2026, 1, 1, 10, 0)
    with pytest.raises(RiskViolation, match="per-trade cap"):
        book.check_entry("AAA", premium_per_share=2.00, contracts=1, now=now)  # $200


def test_daily_entry_cap_enforced():
    book = RiskBook(limits=RiskLimits(max_entries_per_day=2, max_risk_per_day=1000.0,
                                      max_open_risk=1000.0, max_open_positions=5,
                                      one_per_underlying=False))
    now = datetime(2026, 1, 1, 10, 0)
    book.record_entry(_pos("AAA", 0.5), now)
    book.record_entry(_pos("BBB", 0.5), now)
    with pytest.raises(RiskViolation, match="already opened 2"):
        book.check_entry("CCC", 0.5, 1, now)


def test_one_position_per_underlying():
    book = RiskBook(limits=RiskLimits(max_open_risk=1000.0, max_risk_per_day=1000.0))
    now = datetime(2026, 1, 1, 10, 0)
    book.record_entry(_pos("AAA", 0.5), now)
    with pytest.raises(RiskViolation, match="already hold a position in AAA"):
        book.check_entry("AAA", 0.5, 1, now)


def test_daily_counters_reset_next_day():
    book = RiskBook(limits=RiskLimits(max_entries_per_day=1, max_open_risk=1000.0,
                                      max_risk_per_day=1000.0, one_per_underlying=False))
    day1 = datetime(2026, 1, 1, 10, 0)
    book.record_entry(_pos("AAA", 0.5), day1)
    with pytest.raises(RiskViolation):
        book.check_entry("BBB", 0.5, 1, day1)
    day2 = datetime(2026, 1, 2, 10, 0)
    book.check_entry("BBB", 0.5, 1, day2)  # must not raise


def test_max_contracts_respects_every_cap():
    book = RiskBook(limits=RiskLimits(max_risk_per_trade=150.0, max_open_risk=150.0,
                                      max_risk_per_day=150.0))
    assert book.max_contracts(1.50) == 1   # $150 exactly
    assert book.max_contracts(0.50) == 3   # $50 each
    assert book.max_contracts(2.00) == 0   # $200 > cap
    assert book.max_contracts(0.0) == 0


def test_exits_are_never_blocked_by_caps():
    """Daily caps must not trap a position that needs closing."""
    book = RiskBook(limits=RiskLimits(max_entries_per_day=1, max_open_risk=1000.0,
                                      max_risk_per_day=1000.0))
    now = datetime(2026, 1, 1, 10, 0)
    p = _pos("AAA", 1.0)
    book.record_entry(p, now)
    due = book.due_exits({"AAA": 0.4}, now, ExitRules())  # -60%
    assert due and "stop loss" in due[0][1]
    book.record_exit(p)
    assert book.positions == []


def test_exit_rules_trigger_correctly():
    rules = ExitRules(take_profit=0.6, stop_loss=0.5, min_days_to_expiry=2, max_hold_days=10)
    p = _pos("AAA", price=1.0, days=30)
    now = datetime(2026, 1, 2, 10, 0)
    assert p.exit_reason(1.7, now, rules).startswith("take profit")
    assert p.exit_reason(0.4, now, rules).startswith("stop loss")
    assert p.exit_reason(1.0, now, rules) is None
    # Near expiry forces a close even at breakeven.
    near = Position("AAA", 100.0, "call", date(2026, 1, 3), 1, 1.0, datetime(2026, 1, 1, 10, 0))
    assert "expiry" in near.exit_reason(1.0, now, rules)
    # Stale position closed on time alone.
    stale = datetime(2026, 1, 15, 10, 0)
    assert "held" in p.exit_reason(1.0, stale, rules)
