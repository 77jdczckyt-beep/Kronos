"""Scan a watchlist for entry candidates, and optionally act on them.

`scan` is the useful half: forecast each symbol, score its option chain, apply
risk limits, and return ranked candidates with the reasoning attached --
including the rejections, so a quiet day can explain itself.

`run` adds order placement, which is inert unless handed an armed `LiveBroker`.
The default is a dry run that records what it would have sent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from .broker import Broker, DryRunBroker, OrderRequest, OrderResult
from .data import MarketData, MarketDataError
from .forecast import Forecaster
from .risk import ExitRules, Position, RiskBook, RiskLimits, RiskViolation
from .signals import Signal, SignalConfig, best_signal


@dataclass
class ScanConfig:
    """Parameters for one scan pass.

    Attributes:
        lookback: Bars of history fed to the forecaster.
        horizon: Bars ahead to forecast.
        n_paths: Ensemble size. Below ~64 the tail estimates are too noisy to
            support a signal; the signal layer enforces its own floor too.
        expiry_days: Nominal option lifetime used when building the order.
    """

    lookback: int = 480
    horizon: int = 48
    n_paths: int = 64
    expiry_days: int = 30


@dataclass
class Candidate:
    symbol: str
    signal: Signal
    contracts: int
    order: OrderRequest

    @property
    def risk(self) -> float:
        return self.order.max_loss


@dataclass
class ScanReport:
    """Everything a scan concluded, including why it did nothing."""

    candidates: list[Candidate] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        if self.candidates:
            lines.append(f"{len(self.candidates)} candidate(s):")
            for c in self.candidates:
                lines.append(
                    f"  {c.symbol} {c.signal.side} {c.signal.quote.strike:g} "
                    f"x{c.contracts} @ ${c.order.limit_price:.2f} "
                    f"(${c.risk:.2f} at risk) -- {c.signal.reason}"
                )
        else:
            lines.append("no candidates")
        for sym, why in self.skipped.items():
            lines.append(f"  skipped {sym}: {why}")
        for sym, why in self.errors.items():
            lines.append(f"  ERROR {sym}: {why}")
        return "\n".join(lines)


def scan(
    symbols: list[str],
    market: MarketData,
    forecaster: Forecaster,
    book: RiskBook,
    now: datetime,
    cfg: ScanConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> ScanReport:
    """Evaluate each symbol and return ranked entry candidates.

    A data failure on one symbol is recorded and the scan continues; it never
    falls back to stale data, and it never substitutes a guess for a quote.
    """
    cfg = cfg or ScanConfig()
    report = ScanReport()

    for symbol in symbols:
        try:
            hist = market.bars(symbol, cfg.lookback, as_of=now)
            quotes = market.chain(symbol, as_of=now)
        except MarketDataError as e:
            report.errors[symbol] = str(e)
            continue

        if not quotes:
            report.skipped[symbol] = "empty option chain"
            continue

        # Forecast timestamps extend beyond the last bar; spacing is inferred
        # from recent history so intraday and daily data both work.
        step = hist.index[-1] - hist.index[-2]
        horizon_idx = pd.DatetimeIndex(
            [hist.index[-1] + step * (k + 1) for k in range(cfg.horizon)]
        )

        ensemble = forecaster.forecast(symbol, hist, horizon_idx, n_paths=cfg.n_paths)
        best, _all = best_signal(ensemble, list(quotes), signal_cfg)
        if best is None:
            report.skipped[symbol] = "no contract cleared the filters"
            continue

        contracts = book.max_contracts(best.quote.ask)
        if contracts < 1:
            report.skipped[symbol] = (
                f"risk budget allows 0 contracts at ${best.quote.ask:.2f}/share"
            )
            continue

        try:
            book.check_entry(symbol, best.quote.ask, contracts, now)
        except RiskViolation as e:
            report.skipped[symbol] = f"risk: {e}"
            continue

        order = OrderRequest(
            symbol=symbol,
            strike=best.quote.strike,
            right=best.side,
            expiry=(now + timedelta(days=cfg.expiry_days)).date().isoformat(),
            contracts=contracts,
            # Bid the mid, never the ask -- paying the full spread on entry is
            # a guaranteed loss on a strategy whose edge is measured in tens of
            # basis points.
            limit_price=round(best.quote.mid, 2),
        )
        report.candidates.append(Candidate(symbol, best, contracts, order))

    report.candidates.sort(key=lambda c: c.signal.edge_ratio, reverse=True)
    return report


def manage_exits(
    book: RiskBook,
    marks: dict[str, float],
    now: datetime,
    rules: ExitRules | None = None,
) -> list[tuple[Position, str]]:
    """Positions due to close. Always evaluated before any new entry."""
    return book.due_exits(marks, now, rules or ExitRules())


def run(
    symbols: list[str],
    market: MarketData,
    forecaster: Forecaster,
    broker: Broker | None = None,
    book: RiskBook | None = None,
    now: datetime | None = None,
    cfg: ScanConfig | None = None,
    signal_cfg: SignalConfig | None = None,
) -> tuple[ScanReport, list[OrderResult]]:
    """Scan, then submit candidates through `broker`.

    With the default `DryRunBroker` nothing leaves the process. Live placement
    requires an armed `LiveBroker`, which strategy code cannot arm itself.
    """
    now = now or datetime.now()
    book = book or RiskBook(limits=RiskLimits())
    broker = broker or DryRunBroker()

    report = scan(symbols, market, forecaster, book, now, cfg, signal_cfg)

    results: list[OrderResult] = []
    for c in report.candidates:
        check = broker.review(c.order)
        if not check.accepted:
            report.skipped[c.symbol] = f"broker review: {check.detail}"
            results.append(check)
            continue
        results.append(broker.place(c.order))
    return report, results
