"""Market data sources.

Everything the scanner needs is expressed through `MarketData`, so the source
can be a CSV, a vendor API, or the brokerage itself without the strategy
noticing. Two rules that exist because violating them silently destroys a
backtest:

1. `bars` must never return a bar that closed after `as_of`. Leaking one future
   bar into the lookback window inflates results in a way that is very hard to
   spot afterwards.
2. `chain` returns live quotes only. There is no historical options data here,
   and synthesising it (as `backtest.py` does) is a simulation, not a record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

import pandas as pd

from .pricing import OptionQuote


class MarketDataError(RuntimeError):
    """Raised when a source cannot answer. Never fall back to stale data."""


class MarketData(Protocol):
    def bars(self, symbol: str, count: int, as_of: datetime | None = None) -> pd.DataFrame:
        """Most recent `count` OHLCV bars at or before `as_of`."""
        ...

    def spot(self, symbol: str, as_of: datetime | None = None) -> float:
        """Last traded price."""
        ...

    def chain(self, symbol: str, as_of: datetime | None = None) -> Sequence[OptionQuote]:
        """Tradeable option contracts with live quotes."""
        ...


@dataclass
class CsvMarketData:
    """OHLCV from a dataframe. Supplies no option chain.

    Used for backtests and tests. `chain` raises rather than inventing quotes,
    so a caller that needs options cannot silently receive fabricated ones.
    """

    frame: pd.DataFrame
    symbol: str = "SIM"

    def __post_init__(self) -> None:
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            raise MarketDataError("frame must be indexed by timestamp")
        if not self.frame.index.is_monotonic_increasing:
            self.frame = self.frame.sort_index()

    def bars(self, symbol: str, count: int, as_of: datetime | None = None) -> pd.DataFrame:
        df = self.frame
        if as_of is not None:
            df = df.loc[df.index <= pd.Timestamp(as_of)]
        if len(df) < count:
            raise MarketDataError(
                f"need {count} bars for {symbol} at {as_of}, have {len(df)}"
            )
        return df.iloc[-count:]

    def spot(self, symbol: str, as_of: datetime | None = None) -> float:
        return float(self.bars(symbol, 1, as_of)["close"].iloc[-1])

    def chain(self, symbol: str, as_of: datetime | None = None) -> Sequence[OptionQuote]:
        raise MarketDataError(
            "CsvMarketData has no option chain. Backtests synthesise one via "
            "backtest.build_chain; live scans need a real quote source."
        )
