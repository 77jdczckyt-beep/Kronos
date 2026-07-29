"""Walk-forward backtest for the long-option strategy.

Read this before trusting any number it prints.

There is no historical options data in this repository, so the harness
*simulates* an options market: it prices contracts with Black-Scholes using a
volatility drawn from the underlying's own trailing realised volatility, marked
up by a volatility risk premium. That markup is the single most important
parameter in the whole file. It represents the empirically robust fact that
implied volatility trades above subsequent realised volatility -- option
sellers are paid for bearing variance risk, and option buyers pay for it. It is
the headwind every long-premium strategy runs into.

The consequence: a forecaster with no genuine predictive skill *must* lose
money here, at roughly the rate of the premium plus spread. If a strategy shows
a profit in this harness, the first hypothesis is a bug in the harness, not an
edge. That is what makes the result worth anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .forecast import Forecaster
from .pricing import OptionQuote, bs_price
from .risk import ExitRules, Position, RiskBook, RiskLimits, RiskViolation
from .signals import SignalConfig, best_signal


@dataclass
class BacktestConfig:
    """Parameters of the simulated market and the walk-forward loop.

    Attributes:
        lookback: Bars of history fed to the forecaster.
        horizon: Bars ahead the forecast covers; also the option's life.
        step: Bars between successive decision points.
        n_paths: Ensemble size per decision.
        vol_risk_premium: Multiplier applied to trailing realised vol to obtain
            the implied vol contracts are sold at. 1.0 means options are priced
            at fair realised vol -- unrealistically generous to the buyer.
            Empirically this sits around 1.1-1.3 for equity index options.
        spread_frac: Simulated bid-ask spread as a fraction of theoretical
            value. Paid on entry and again on exit.
        bars_per_year: Bars in a trading year, for annualising volatility.
            5-minute HK equity bars: ~48 bars/day * 246 days.
        strike_offsets: Moneyness levels offered in the simulated chain.
    """

    lookback: int = 480
    horizon: int = 48
    step: int = 48
    n_paths: int = 64
    vol_risk_premium: float = 1.15
    spread_frac: float = 0.06
    bars_per_year: int = 48 * 246
    strike_offsets: tuple[float, ...] = (0.98, 0.99, 1.0, 1.01, 1.02, 1.03)


@dataclass
class Trade:
    """One completed round trip."""

    symbol: str
    side: str
    strike: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    contracts: int
    reason: str

    @property
    def pnl(self) -> float:
        return (self.exit_price - self.entry_price) * 100.0 * self.contracts

    @property
    def risk(self) -> float:
        return self.entry_price * 100.0 * self.contracts


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    decisions: int = 0
    entries_blocked_by_risk: int = 0
    no_signal: int = 0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def total_risked(self) -> float:
        return sum(t.risk for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)

    @property
    def return_on_risk(self) -> float:
        """P&L as a fraction of total premium committed."""
        r = self.total_risked
        return self.total_pnl / r if r > 0 else 0.0

    def summary(self) -> str:
        if not self.trades:
            return (
                f"decisions={self.decisions} trades=0 "
                f"(no_signal={self.no_signal}, risk_blocked={self.entries_blocked_by_risk})\n"
                "No trades taken. For a strict long-premium filter this is a "
                "normal and healthy outcome."
            )
        pnls = [t.pnl for t in self.trades]
        return (
            f"decisions={self.decisions}  trades={len(self.trades)}  "
            f"no_signal={self.no_signal}  risk_blocked={self.entries_blocked_by_risk}\n"
            f"total P&L      ${self.total_pnl:,.2f}\n"
            f"total risked   ${self.total_risked:,.2f}\n"
            f"return on risk {self.return_on_risk:+.1%}\n"
            f"win rate       {self.win_rate:.1%}\n"
            f"best / worst   ${max(pnls):,.2f} / ${min(pnls):,.2f}\n"
            f"median trade   ${np.median(pnls):,.2f}"
        )


def realised_vol(closes: np.ndarray, bars_per_year: int) -> float:
    """Annualised volatility of log returns."""
    rets = np.diff(np.log(closes))
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets, ddof=1) * np.sqrt(bars_per_year))


def build_chain(spot: float, t_years: float, iv: float, cfg: BacktestConfig) -> list[OptionQuote]:
    """Construct a simulated option chain around spot."""
    quotes = []
    half = cfg.spread_frac / 2.0
    for off in cfg.strike_offsets:
        strike = spot * off
        for right in ("call", "put"):
            theo = bs_price(spot, strike, t_years, iv, right)
            if theo <= 0.01:
                continue
            quotes.append(
                OptionQuote(
                    symbol="SIM",
                    strike=strike,
                    expiry_years=t_years,
                    right=right,
                    bid=theo * (1 - half),
                    ask=theo * (1 + half),
                    open_interest=1000,
                )
            )
    return quotes


def run_backtest(
    df: pd.DataFrame,
    forecaster: Forecaster,
    cfg: BacktestConfig | None = None,
    signal_cfg: SignalConfig | None = None,
    limits: RiskLimits | None = None,
    max_decisions: int | None = None,
) -> BacktestResult:
    """Walk forward through `df`, deciding and settling one horizon at a time.

    Positions are held to the end of the horizon and settled at intrinsic
    value. This overstates realism in one direction (no early exit on the
    take-profit rule) and understates it in another (no early stop-out), so
    treat the result as a first-order estimate of the strategy's edge, not a
    faithful simulation of live behaviour.
    """
    cfg = cfg or BacktestConfig()
    result = BacktestResult()
    book = RiskBook(limits=limits or RiskLimits())

    closes_all = df["close"].to_numpy(dtype=float)
    t_years = cfg.horizon / cfg.bars_per_year

    start = cfg.lookback
    end = len(df) - cfg.horizon - 1
    for i in range(start, end, cfg.step):
        if max_decisions and result.decisions >= max_decisions:
            break
        result.decisions += 1

        hist = df.iloc[i - cfg.lookback : i]
        horizon_idx = df.index[i : i + cfg.horizon]
        spot = float(closes_all[i - 1])

        rv = realised_vol(closes_all[i - cfg.lookback : i], cfg.bars_per_year)
        if rv <= 0:
            continue
        iv = rv * cfg.vol_risk_premium

        ensemble = forecaster.forecast("SIM", hist, horizon_idx, n_paths=cfg.n_paths)
        quotes = build_chain(spot, t_years, iv, cfg)
        best, _ = best_signal(ensemble, quotes, signal_cfg)

        if best is None:
            result.no_signal += 1
            continue

        now = pd.Timestamp(df.index[i]).to_pydatetime()
        contracts = book.max_contracts(best.quote.ask)
        if contracts < 1:
            result.entries_blocked_by_risk += 1
            continue

        pos = Position(
            symbol=f"SIM-{i}",  # unique so one-per-underlying does not block the walk
            strike=best.quote.strike,
            right=best.side,
            expiry=(now + timedelta(days=30)).date(),
            contracts=contracts,
            entry_price=best.quote.ask,
            opened_at=now,
        )
        try:
            book.record_entry(pos, now)
        except RiskViolation:
            result.entries_blocked_by_risk += 1
            continue

        # Settle at horizon end: intrinsic value, minus the exit half-spread.
        terminal = float(closes_all[i + cfg.horizon - 1])
        intrinsic = (
            max(0.0, terminal - best.quote.strike)
            if best.side == "call"
            else max(0.0, best.quote.strike - terminal)
        )
        exit_price = max(0.0, intrinsic * (1 - cfg.spread_frac / 2.0))

        result.trades.append(
            Trade(
                symbol="SIM",
                side=best.side,
                strike=best.quote.strike,
                entry_time=now,
                exit_time=pd.Timestamp(df.index[i + cfg.horizon - 1]).to_pydatetime(),
                entry_price=best.quote.ask,
                exit_price=exit_price,
                contracts=contracts,
                reason=best.reason,
            )
        )
        book.record_exit(pos)

    return result
