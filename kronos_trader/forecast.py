"""Ensemble forecasting for options signals.

The central problem this module solves:

``KronosPredictor.predict`` averages its sampled trajectories before returning
them (``preds = np.mean(preds, axis=1)`` in ``auto_regressive_inference``).
That average is a reasonable *point* forecast, but it is actively misleading
for options work. Averaging N independent sampled paths shrinks the dispersion
of the result by roughly sqrt(N); the mean path of 30 samples looks far calmer
than any real price path. An option is priced off the distribution -- off the
probability that spot travels past the strike -- so a signal computed from the
collapsed mean systematically understates the chance of a large move.

So we never call ``predict`` for signal generation. We call ``predict_batch``
with the same series repeated N times and ``sample_count=1``, which yields N
independent trajectories from one batched forward pass, and we keep all of them.

Everything downstream consumes a ``PathEnsemble`` -- the raw (n_paths, horizon)
matrix of simulated closes -- and never a point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

# Columns Kronos expects on its input frame.
OHLCV_COLS = ["open", "high", "low", "close", "volume", "amount"]


@dataclass(frozen=True)
class PathEnsemble:
    """A bundle of simulated future price paths for one underlying.

    Attributes:
        symbol: Underlying ticker the paths belong to.
        spot: Last observed close, i.e. the origin of every path.
        closes: ``(n_paths, horizon)`` matrix of simulated closing prices.
        highs: ``(n_paths, horizon)`` simulated highs, used for touch
            probabilities (an American option can be exercised early, and a
            stop can be hit intrabar, so the running extremum matters).
        lows: ``(n_paths, horizon)`` simulated lows.
        timestamps: Horizon timestamps, length ``horizon``.
    """

    symbol: str
    spot: float
    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    timestamps: pd.DatetimeIndex

    def __post_init__(self) -> None:
        if self.closes.ndim != 2:
            raise ValueError(f"closes must be 2-D (n_paths, horizon), got {self.closes.shape}")
        if self.closes.shape[0] < 2:
            raise ValueError(
                f"a single path is not an ensemble; got {self.closes.shape[0]} path(s). "
                "Distribution-based signals need many samples to be meaningful."
            )
        for name, arr in (("highs", self.highs), ("lows", self.lows)):
            if arr.shape != self.closes.shape:
                raise ValueError(f"{name} shape {arr.shape} != closes shape {self.closes.shape}")
        if self.spot <= 0:
            raise ValueError(f"spot must be positive, got {self.spot}")

    @property
    def n_paths(self) -> int:
        return int(self.closes.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.closes.shape[1])

    @property
    def terminal(self) -> np.ndarray:
        """Closing price at the end of the horizon, one entry per path."""
        return self.closes[:, -1]

    def prob_above(self, level: float) -> float:
        """P(terminal close > level), estimated from the ensemble."""
        return float(np.mean(self.terminal > level))

    def prob_below(self, level: float) -> float:
        """P(terminal close < level), estimated from the ensemble."""
        return float(np.mean(self.terminal < level))

    def prob_touch_above(self, level: float) -> float:
        """P(the path's running high reaches ``level`` at any point).

        Always >= ``prob_above(level)``. This is the number that matters for a
        position you intend to close before expiry, which is most of them.
        """
        return float(np.mean(self.highs.max(axis=1) >= level))

    def prob_touch_below(self, level: float) -> float:
        """P(the path's running low reaches ``level`` at any point)."""
        return float(np.mean(self.lows.min(axis=1) <= level))

    def terminal_log_return(self) -> np.ndarray:
        """Per-path log return over the full horizon."""
        return np.log(self.terminal / self.spot)

    def forecast_volatility(self) -> float:
        """Std-dev of terminal log returns -- the ensemble's implied move size.

        This is the quantity that the averaged forecast destroys, and the one
        that has to be compared against the option's implied volatility before
        any trade can be justified.
        """
        return float(np.std(self.terminal_log_return(), ddof=1))

    def summary(self) -> dict:
        lr = self.terminal_log_return()
        return {
            "symbol": self.symbol,
            "spot": self.spot,
            "n_paths": self.n_paths,
            "horizon": self.horizon,
            "median_return": float(np.median(lr)),
            "mean_return": float(np.mean(lr)),
            "vol": self.forecast_volatility(),
            "q05": float(np.quantile(lr, 0.05)),
            "q95": float(np.quantile(lr, 0.95)),
        }


class Forecaster(Protocol):
    """Anything that can turn a history window into a `PathEnsemble`."""

    def forecast(
        self,
        symbol: str,
        history: pd.DataFrame,
        horizon_timestamps: pd.DatetimeIndex,
        n_paths: int,
    ) -> PathEnsemble: ...


def _validate_history(history: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in ["open", "high", "low", "close"] if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing required columns: {missing}")
    if len(history) < 2:
        raise ValueError("history needs at least 2 bars")
    if history[["open", "high", "low", "close"]].isnull().values.any():
        raise ValueError("history contains NaNs in OHLC columns")
    out = history.copy()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if "amount" not in out.columns:
        out["amount"] = out["volume"] * out[["open", "high", "low", "close"]].mean(axis=1)
    return out[OHLCV_COLS]


class KronosEnsembleForecaster:
    """Draws independent Kronos trajectories and keeps every one of them.

    Args:
        predictor: A ``KronosPredictor``.
        temperature: Sampling temperature ``T``. Higher spreads the ensemble.
        top_p: Nucleus sampling cutoff.
        batch_size: Max paths per batched forward pass. Paths are drawn in
            chunks of this size so that large ensembles do not exhaust memory.
    """

    def __init__(
        self,
        predictor,
        temperature: float = 1.0,
        top_p: float = 0.9,
        batch_size: int = 16,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.predictor = predictor
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size

    def forecast(
        self,
        symbol: str,
        history: pd.DataFrame,
        horizon_timestamps: pd.DatetimeIndex,
        n_paths: int = 64,
    ) -> PathEnsemble:
        if n_paths < 2:
            raise ValueError("n_paths must be >= 2 to form a distribution")
        hist = _validate_history(history)
        x_timestamp = pd.Series(history.index if isinstance(history.index, pd.DatetimeIndex) else history["timestamps"])
        y_timestamp = pd.Series(horizon_timestamps)
        pred_len = len(horizon_timestamps)

        closes, highs, lows = [], [], []
        drawn = 0
        while drawn < n_paths:
            chunk = min(self.batch_size, n_paths - drawn)
            # Same series repeated `chunk` times with sample_count=1 gives
            # `chunk` independent trajectories from one forward pass. This is
            # the whole trick: sample_count>1 would average them away inside
            # auto_regressive_inference before we ever see them.
            dfs = self.predictor.predict_batch(
                df_list=[hist] * chunk,
                x_timestamp_list=[x_timestamp] * chunk,
                y_timestamp_list=[y_timestamp] * chunk,
                pred_len=pred_len,
                T=self.temperature,
                top_p=self.top_p,
                sample_count=1,
                verbose=False,
            )
            for d in dfs:
                closes.append(d["close"].to_numpy())
                highs.append(d["high"].to_numpy())
                lows.append(d["low"].to_numpy())
            drawn += chunk

        return PathEnsemble(
            symbol=symbol,
            spot=float(hist["close"].iloc[-1]),
            closes=np.asarray(closes, dtype=float),
            highs=np.asarray(highs, dtype=float),
            lows=np.asarray(lows, dtype=float),
            timestamps=pd.DatetimeIndex(horizon_timestamps),
        )


class BootstrapForecaster:
    """Block-bootstrap baseline that needs no model weights.

    This exists for two reasons, both important:

    1. It lets the entire pipeline below this layer be tested and backtested
       without downloading Kronos, which matters in locked-down environments.
    2. It is the null hypothesis. Kronos has to beat a resampling of the
       symbol's own recent returns before anyone should believe it adds
       anything. A strategy that is profitable on this forecaster is
       profitable on noise, which means the profit came from the trade
       structure, not from the forecast.

    Args:
        block: Length of the contiguous return blocks that get resampled.
            Blocks preserve short-range autocorrelation and volatility
            clustering that i.i.d. resampling would destroy.
        lookback: Number of recent bars to resample from.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, block: int = 8, lookback: int = 512, seed: int | None = None) -> None:
        if block < 1:
            raise ValueError("block must be >= 1")
        self.block = block
        self.lookback = lookback
        self.rng = np.random.default_rng(seed)

    def forecast(
        self,
        symbol: str,
        history: pd.DataFrame,
        horizon_timestamps: pd.DatetimeIndex,
        n_paths: int = 64,
    ) -> PathEnsemble:
        if n_paths < 2:
            raise ValueError("n_paths must be >= 2 to form a distribution")
        hist = _validate_history(history)
        closes = hist["close"].to_numpy(dtype=float)
        spot = float(closes[-1])
        pred_len = len(horizon_timestamps)

        rets = np.diff(np.log(closes[-self.lookback:]))
        if len(rets) < self.block + 1:
            raise ValueError(f"need > {self.block} returns to bootstrap, have {len(rets)}")

        # Typical intrabar range as a fraction of close, used to synthesise
        # plausible highs/lows around each simulated close.
        rng_frac = float(np.median((hist["high"] - hist["low"]) / hist["close"].replace(0, np.nan)).item()) \
            if len(hist) else 0.0
        rng_frac = 0.0 if not np.isfinite(rng_frac) else rng_frac

        n_blocks = int(np.ceil(pred_len / self.block))
        starts = self.rng.integers(0, len(rets) - self.block, size=(n_paths, n_blocks))
        idx = starts[:, :, None] + np.arange(self.block)[None, None, :]
        sampled = rets[idx].reshape(n_paths, -1)[:, :pred_len]

        path_closes = spot * np.exp(np.cumsum(sampled, axis=1))
        half = rng_frac / 2.0
        path_highs = path_closes * (1.0 + half)
        path_lows = path_closes * (1.0 - half)

        return PathEnsemble(
            symbol=symbol,
            spot=spot,
            closes=path_closes,
            highs=path_highs,
            lows=path_lows,
            timestamps=pd.DatetimeIndex(horizon_timestamps),
        )
