"""Does the forecaster predict anything? Measured directly, before any trading.

This is the experiment that should gate funding. It deliberately does not
simulate a strategy: a strategy adds thresholds, position sizing, and costs,
each of which is a free parameter that can manufacture a good-looking result
from a forecaster with no skill at all. Instead it measures the raw
relationship between what the model predicted and what happened.

Two metrics:

**Information coefficient (IC)** -- correlation between predicted return and
realised return across decision points. This is the standard measure of
forecast skill, and it uses every observation rather than only the ones a
threshold happened to select. An IC of 0 means no skill. Real, exploitable
equity signals typically land somewhere around 0.02-0.05; anything above 0.15
on out-of-sample data should be treated as a bug until proven otherwise.

**Directional hit rate** -- how often the sign was right. Easier to read, but
weaker evidence: a forecaster can be right on direction and still lose money by
being right only on small moves and wrong on large ones.

Both are reported against a skill-free baseline run over many seeds, because
the only meaningful question is whether the model beats noise, not whether its
number is positive.

Decision points are spaced at least one horizon apart so that outcome windows
never overlap. Overlapping windows correlate the observations and inflate
significance, which is the most common way this kind of study fools itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .forecast import Forecaster


@dataclass
class EdgeResult:
    """Predicted versus realised, across all decision points."""

    name: str
    predicted: list[float] = field(default_factory=list)
    realised: list[float] = field(default_factory=list)
    prob_up: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.predicted)

    def ic_pearson(self) -> float:
        if self.n < 3:
            return float("nan")
        p, r = np.array(self.predicted), np.array(self.realised)
        if p.std() == 0 or r.std() == 0:
            return float("nan")
        return float(np.corrcoef(p, r)[0, 1])

    def ic_spearman(self) -> float:
        """Rank correlation -- robust to the fat tails of return data."""
        if self.n < 3:
            return float("nan")
        p = pd.Series(self.predicted).rank()
        r = pd.Series(self.realised).rank()
        if p.std() == 0 or r.std() == 0:
            return float("nan")
        return float(np.corrcoef(p, r)[0, 1])

    def hit_rate(self) -> float:
        """Fraction of decisions where the predicted sign matched the outcome."""
        if self.n == 0:
            return float("nan")
        p, r = np.array(self.predicted), np.array(self.realised)
        mask = r != 0
        if mask.sum() == 0:
            return float("nan")
        return float(np.mean(np.sign(p[mask]) == np.sign(r[mask])))

    def ic_tstat(self) -> float:
        """t-statistic of the IC. |t| > 2 is the usual bar for significance."""
        ic = self.ic_pearson()
        if not np.isfinite(ic) or self.n < 4 or abs(ic) >= 1.0:
            return float("nan")
        return float(ic * np.sqrt((self.n - 2) / (1 - ic * ic)))

    def summary(self) -> str:
        return (
            f"{self.name:<22} n={self.n:<4} "
            f"IC={self.ic_pearson():+.4f} (rank {self.ic_spearman():+.4f})  "
            f"t={self.ic_tstat():+.2f}  hit={self.hit_rate():.1%}"
        )


def evaluate(
    df: pd.DataFrame,
    forecaster: Forecaster,
    name: str,
    lookback: int = 256,
    horizon: int = 12,
    n_paths: int = 32,
    n_points: int = 120,
    symbol: str = "SIM",
    verbose: bool = False,
) -> EdgeResult:
    """Walk the data, forecast at each point, and record predicted vs realised.

    Decision points are spread evenly across the whole sample and spaced at
    least `horizon` bars apart, so outcome windows never overlap.
    """
    result = EdgeResult(name=name)
    closes = df["close"].to_numpy(dtype=float)

    start, end = lookback, len(df) - horizon - 1
    if end <= start:
        raise ValueError("not enough data for the requested lookback and horizon")

    spacing = max(horizon, (end - start) // max(1, n_points))
    points = list(range(start, end, spacing))[:n_points]

    for k, i in enumerate(points):
        hist = df.iloc[i - lookback : i]
        horizon_idx = df.index[i : i + horizon]
        spot = float(closes[i - 1])

        ens = forecaster.forecast(symbol, hist, horizon_idx, n_paths=n_paths)
        pred_lr = float(np.median(ens.terminal_log_return()))
        actual_lr = float(np.log(closes[i + horizon - 1] / spot))

        result.predicted.append(pred_lr)
        result.realised.append(actual_lr)
        result.prob_up.append(ens.prob_above(spot))

        if verbose and (k + 1) % 10 == 0:
            print(f"  {name}: {k + 1}/{len(points)} IC={result.ic_pearson():+.4f}",
                  flush=True)

    return result


def noise_band(results: list[EdgeResult]) -> str:
    """Summarise a set of skill-free runs: the range luck alone produces."""
    ics = np.array([r.ic_pearson() for r in results if np.isfinite(r.ic_pearson())])
    hits = np.array([r.hit_rate() for r in results if np.isfinite(r.hit_rate())])
    if len(ics) == 0:
        return "no valid baseline runs"
    return (
        f"baseline over {len(ics)} seeds: "
        f"IC mean {ics.mean():+.4f}, sd {ics.std(ddof=1):.4f}, "
        f"range {ics.min():+.4f} to {ics.max():+.4f}  |  "
        f"hit mean {hits.mean():.1%}, range {hits.min():.1%} to {hits.max():.1%}"
    )
