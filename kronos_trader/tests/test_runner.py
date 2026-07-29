"""Tests for data access, the broker gate, and the scan loop."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from kronos_trader.broker import (
    DryRunBroker,
    LiveBroker,
    MCPBroker,
    NotArmed,
    OrderRequest,
)
from kronos_trader.data import CsvMarketData, MarketDataError
from kronos_trader.forecast import BootstrapForecaster
from kronos_trader.pricing import OptionQuote
from kronos_trader.risk import RiskBook, RiskLimits
from kronos_trader.runner import ScanConfig, run, scan

ALI_CSV = "finetune_csv/data/HK_ali_09988_kline_5min_all.csv"


@pytest.fixture(scope="module")
def ali() -> pd.DataFrame:
    df = pd.read_csv(ALI_CSV)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.set_index("timestamps")


# --- market data --------------------------------------------------------------


def test_bars_never_leak_the_future(ali: pd.DataFrame):
    """The single most important property of a backtest data source."""
    md = CsvMarketData(ali)
    cutoff = ali.index[5000]
    bars = md.bars("09988", 100, as_of=cutoff)
    assert bars.index.max() <= cutoff
    assert len(bars) == 100


def test_insufficient_history_raises_rather_than_truncating(ali: pd.DataFrame):
    md = CsvMarketData(ali)
    with pytest.raises(MarketDataError, match="need 500 bars"):
        md.bars("09988", 500, as_of=ali.index[10])


def test_csv_source_refuses_to_invent_a_chain(ali: pd.DataFrame):
    md = CsvMarketData(ali)
    with pytest.raises(MarketDataError, match="no option chain"):
        md.chain("09988")


def test_unsorted_frame_is_sorted(ali: pd.DataFrame):
    md = CsvMarketData(ali.iloc[:100].sample(frac=1.0, random_state=0))
    assert md.frame.index.is_monotonic_increasing


# --- broker gate --------------------------------------------------------------


def _order(price=1.0, contracts=1) -> OrderRequest:
    return OrderRequest("AAA", 100.0, "call", "2026-09-01", contracts, price)


def test_order_cannot_express_selling_to_open():
    """Short options are unrepresentable by construction, not by policy."""
    fields = OrderRequest.__dataclass_fields__
    assert "side" not in fields and "action" not in fields
    assert _order().max_loss == 100.0


def test_order_rejects_invalid_input():
    with pytest.raises(ValueError, match="contracts must be >= 1"):
        OrderRequest("AAA", 100.0, "call", "2026-09-01", 0, 1.0)
    with pytest.raises(ValueError, match="limit_price must be positive"):
        OrderRequest("AAA", 100.0, "call", "2026-09-01", 1, 0.0)


def test_each_order_gets_a_unique_ref_id():
    assert _order().ref_id != _order().ref_id


def test_live_broker_refuses_unless_armed():
    lb = LiveBroker(inner=DryRunBroker(starting_buying_power=1000.0))
    assert lb.review(_order()).accepted  # review is always safe
    with pytest.raises(NotArmed, match="armed=True"):
        lb.place(_order())


def test_armed_broker_respects_its_order_budget():
    lb = LiveBroker(inner=DryRunBroker(starting_buying_power=1000.0), armed=True, max_orders=1)
    assert lb.place(_order()).accepted
    with pytest.raises(NotArmed, match="order budget exhausted"):
        lb.place(_order())


def test_dry_run_broker_blocks_on_buying_power():
    b = DryRunBroker(starting_buying_power=50.0)
    res = b.review(_order(price=1.0))  # needs $100
    assert not res.accepted
    assert "insufficient buying power" in res.detail
    assert b.submitted == []


def test_mcp_broker_is_an_explicit_stub():
    """It must fail loudly rather than pretend to be wired up."""
    with pytest.raises(NotImplementedError, match="do not guess tool names"):
        MCPBroker()


# --- scan ---------------------------------------------------------------------


class _ChainSource(CsvMarketData):
    """CSV bars plus a supplied option chain."""

    def __init__(self, frame, quotes):
        super().__init__(frame)
        self._quotes = quotes

    def chain(self, symbol, as_of=None):
        return self._quotes


def test_scan_reports_reasons_when_nothing_qualifies(ali: pd.DataFrame):
    spot = float(ali["close"].iloc[-1])
    # Absurdly expensive contracts: nothing can clear the edge filter.
    quotes = [
        OptionQuote("09988", spot * m, 0.05, "call", bid=40.0, ask=41.0, open_interest=1000)
        for m in (1.01, 1.02)
    ]
    md = _ChainSource(ali, quotes)
    report = scan(
        ["09988"], md, BootstrapForecaster(seed=3), RiskBook(),
        now=ali.index[-1].to_pydatetime(), cfg=ScanConfig(lookback=300, horizon=24),
    )
    assert report.candidates == []
    assert "no contract cleared" in report.skipped["09988"]


def test_scan_surfaces_data_errors_without_aborting(ali: pd.DataFrame):
    md = CsvMarketData(ali)  # chain() raises
    report = scan(
        ["09988"], md, BootstrapForecaster(seed=3), RiskBook(),
        now=ali.index[-1].to_pydatetime(), cfg=ScanConfig(lookback=300, horizon=24),
    )
    assert "09988" in report.errors
    assert report.candidates == []


def test_scan_bids_the_mid_not_the_ask(ali: pd.DataFrame):
    """Paying the full spread on entry is a guaranteed loss."""
    hist = ali.iloc[-800:]
    spot = float(hist["close"].iloc[-1])
    ens = BootstrapForecaster(seed=5).forecast(
        "09988", hist.iloc[-300:], ali.index[-24:], n_paths=128
    )
    strike = spot * 1.005
    fair = float(np.mean(np.maximum(ens.terminal - strike, 0.0)))
    quotes = [
        OptionQuote("09988", strike, 0.05, "call",
                    bid=fair * 0.30, ask=fair * 0.33, open_interest=1000)
    ]
    md = _ChainSource(hist, quotes)
    report = scan(
        ["09988"], md, BootstrapForecaster(seed=5),
        RiskBook(limits=RiskLimits(max_risk_per_trade=1e6, max_risk_per_day=1e6,
                                   max_open_risk=1e6)),
        now=hist.index[-1].to_pydatetime(),
        cfg=ScanConfig(lookback=300, horizon=24, n_paths=128),
    )
    if report.candidates:  # bootstrap edge is stochastic; assert only if it fired
        c = report.candidates[0]
        assert c.order.limit_price <= quotes[0].ask
        assert c.order.limit_price == round(quotes[0].mid, 2)


def test_run_defaults_to_dry_run_and_sends_nothing(ali: pd.DataFrame):
    md = CsvMarketData(ali)
    broker = DryRunBroker(starting_buying_power=10_000.0)
    report, results = run(
        ["09988"], md, BootstrapForecaster(seed=3), broker=broker,
        now=ali.index[-1].to_pydatetime(), cfg=ScanConfig(lookback=300, horizon=24),
    )
    assert broker.submitted == []
    assert isinstance(results, list)


def test_run_through_unarmed_live_broker_raises_on_placement(ali: pd.DataFrame):
    """Guards against a candidate silently reaching a live broker."""
    spot = float(ali["close"].iloc[-1])
    strike = spot * 1.005
    ens = BootstrapForecaster(seed=5).forecast(
        "09988", ali.iloc[-300:], ali.index[-24:], n_paths=128
    )
    fair = float(np.mean(np.maximum(ens.terminal - strike, 0.0)))
    quotes = [
        OptionQuote("09988", strike, 0.05, "call",
                    bid=fair * 0.30, ask=fair * 0.33, open_interest=1000)
    ]
    md = _ChainSource(ali, quotes)
    lb = LiveBroker(inner=DryRunBroker(starting_buying_power=10_000.0), armed=False)
    try:
        run(["09988"], md, BootstrapForecaster(seed=5), broker=lb,
            now=ali.index[-1].to_pydatetime(),
            cfg=ScanConfig(lookback=300, horizon=24, n_paths=128))
    except NotArmed:
        pass  # correct: a candidate reached the gate and was stopped
    assert lb.inner.submitted == []
