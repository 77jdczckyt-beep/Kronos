"""Trivial forecasters that a foundation model must beat to justify itself.

A neural forecaster that beats a random baseline has shown almost nothing. The
question that matters is whether it beats the cheap, well-known effects that
have been documented in equity returns for decades. If a one-line rule captures
the same signal, the one-line rule is the better model: it is faster, it cannot
silently break, and its failure modes are understood.

These produce degenerate ensembles -- every path shares the same drift with
noise around it -- which is fine, because only the median forecast is used by
the edge study. They are nulls, not strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import PathEnsemble, _validate_history


def _ensemble_from_drift(
    symbol: str,
    spot: float,
    drift: float,
    vol: float,
    horizon_timestamps: pd.DatetimeIndex,
    n_paths: int,
    rng: np.random.Generator,
) -> PathEnsemble:
    """Build paths with a fixed terminal drift and plausible dispersion."""
    h = len(horizon_timestamps)
    steps = rng.normal(drift / h, vol, size=(n_paths, h))
    closes = spot * np.exp(np.cumsum(steps, axis=1))
    return PathEnsemble(
        symbol=symbol,
        spot=spot,
        closes=closes,
        highs=closes,
        lows=closes,
        timestamps=pd.DatetimeIndex(horizon_timestamps),
    )


class MeanReversionForecaster:
    """Predicts reversion toward the trailing mean price.

    The oldest idea in the book: when price sits above its recent average,
    forecast a fall, and vice versa. `strength` scales how much of the gap is
    expected to close over the horizon.
    """

    def __init__(self, window: int = 256, strength: float = 0.5, seed: int | None = 0):
        self.window = window
        self.strength = strength
        self.rng = np.random.default_rng(seed)

    def forecast(self, symbol, history, horizon_timestamps, n_paths=32) -> PathEnsemble:
        hist = _validate_history(history)
        closes = hist["close"].to_numpy(dtype=float)
        spot = float(closes[-1])
        ref = float(np.mean(closes[-self.window:]))
        gap = np.log(ref / spot)
        vol = float(np.std(np.diff(np.log(closes[-self.window:])), ddof=1))
        return _ensemble_from_drift(
            symbol, spot, self.strength * gap, vol, horizon_timestamps, n_paths, self.rng
        )


class MomentumForecaster:
    """Predicts continuation of the trailing return."""

    def __init__(self, window: int = 60, strength: float = 0.5, seed: int | None = 0):
        self.window = window
        self.strength = strength
        self.rng = np.random.default_rng(seed)

    def forecast(self, symbol, history, horizon_timestamps, n_paths=32) -> PathEnsemble:
        hist = _validate_history(history)
        closes = hist["close"].to_numpy(dtype=float)
        spot = float(closes[-1])
        past = np.log(spot / float(closes[-self.window]))
        vol = float(np.std(np.diff(np.log(closes[-self.window:])), ddof=1))
        return _ensemble_from_drift(
            symbol, spot, self.strength * past, vol, horizon_timestamps, n_paths, self.rng
        )
