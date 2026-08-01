"""Validation experiments for the signal pipeline.

Run with:  python -m kronos_trader.validate

Three experiments, in order of importance:

1. **Dispersion collapse** -- measures how much information averaging destroys,
   justifying why this package never uses ``KronosPredictor.predict``.
2. **Edge detection** -- feeds the pipeline a forecaster with known, tunable
   skill and confirms profit rises with skill. Without this, a filter that
   rejects everything is indistinguishable from a broken one.
3. **Noise floor** -- runs a forecaster with *no* skill across many seeds to
   establish what pure luck looks like. This is the yardstick any real result
   has to clear.

Experiment 3 is the one that matters for anybody planning to trade this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .forecast import BootstrapForecaster, PathEnsemble
from .signals import SignalConfig

ALI_CSV = "finetune_csv/data/HK_ali_09988_kline_5min_all.csv"


class SkilledForecaster:
    """Bootstrap paths blended toward the true future move.

    ``skill`` in [0, 1] interpolates between an honest bootstrap (0) and
    perfect foreknowledge (1). Strictly a diagnostic -- it reads the future and
    must never be used outside validation.
    """

    def __init__(self, df: pd.DataFrame, horizon: int, skill: float = 0.6, seed: int = 0):
        if not 0.0 <= skill <= 1.0:
            raise ValueError("skill must be in [0, 1]")
        self.horizon = horizon
        self.skill = skill
        self.base = BootstrapForecaster(seed=seed)
        self.closes = df["close"].to_numpy(dtype=float)
        self.pos = {ts: i for i, ts in enumerate(df.index)}

    def forecast(self, symbol, history, horizon_timestamps, n_paths=64) -> PathEnsemble:
        ens = self.base.forecast(symbol, history, horizon_timestamps, n_paths)
        i = self.pos[horizon_timestamps[0]]
        future = self.closes[min(i + self.horizon - 1, len(self.closes) - 1)]
        true_lr = np.log(future / ens.spot)
        shift = self.skill * (true_lr - ens.terminal_log_return().mean())
        f = float(np.exp(shift))
        return PathEnsemble(
            symbol=ens.symbol, spot=ens.spot,
            closes=ens.closes * f, highs=ens.highs * f, lows=ens.lows * f,
            timestamps=ens.timestamps,
        )


def load(path: str = ALI_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.set_index("timestamps")


def experiment_dispersion(df: pd.DataFrame, n: int = 32, reps: int = 40) -> None:
    print("=" * 68)
    print("1. DISPERSION COLLAPSE -- what averaging sampled paths destroys")
    print("=" * 68)
    hist, hor = df.iloc[-600:-100], df.index[-100:-20]
    single, averaged = [], []
    for seed in range(reps):
        e = BootstrapForecaster(seed=seed).forecast("X", hist, hor, n_paths=n)
        single.append(e.closes[0, -1])
        averaged.append(e.closes.mean(axis=0)[-1])
    ratio = np.std(single, ddof=1) / np.std(averaged, ddof=1)
    print(f"  spread of individual paths : {np.std(single, ddof=1):.3f}")
    print(f"  spread of {n}-path averages : {np.std(averaged, ddof=1):.3f}")
    print(f"  collapse factor            : {ratio:.1f}x   (sqrt({n}) = {np.sqrt(n):.1f})")
    print("  An option is priced on this spread. Averaging removes it.\n")


def experiment_edge_detection(df: pd.DataFrame, decisions: int = 400) -> None:
    print("=" * 68)
    print("2. EDGE DETECTION -- does the pipeline trade when edge exists?")
    print("=" * 68)
    cfg = BacktestConfig(vol_risk_premium=1.15)
    for skill in (0.0, 0.3, 0.6, 1.0):
        fc = SkilledForecaster(df, cfg.horizon, skill=skill, seed=11)
        r = run_backtest(df, fc, cfg=cfg, signal_cfg=SignalConfig(), max_decisions=decisions)
        print(
            f"  skill={skill:<4} trades={len(r.trades):<4} "
            f"P&L=${r.total_pnl:>10,.0f}  RoR={r.return_on_risk:>+7.1%}  "
            f"win={r.win_rate:>5.1%}"
        )
    print("  P&L must rise with skill, or the filters are broken.\n")


def experiment_noise_floor(df: pd.DataFrame, seeds: int = 15, decisions: int = 400) -> None:
    print("=" * 68)
    print("3. NOISE FLOOR -- what pure luck produces")
    print("=" * 68)
    cfg = BacktestConfig(vol_risk_premium=1.15)
    rors = []
    for seed in range(seeds):
        fc = SkilledForecaster(df, cfg.horizon, skill=0.0, seed=seed)
        r = run_backtest(df, fc, cfg=cfg, signal_cfg=SignalConfig(), max_decisions=decisions)
        rors.append(r.return_on_risk)
    a = np.array(rors)
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if a.std(ddof=1) > 0 else 0.0
    print(f"  mean return on risk : {a.mean():+.1%}")
    print(f"  median              : {np.median(a):+.1%}")
    print(f"  std across seeds    : {a.std(ddof=1):.1%}")
    print(f"  best / worst seed   : {a.max():+.1%} / {a.min():+.1%}")
    print(f"  profitable seeds    : {(a > 0).sum()}/{seeds}")
    print(f"  t-stat vs zero      : {t:+.2f}")
    print(
        "\n  A skill-free forecaster spans a vast range of outcomes. Any live\n"
        "  result inside this range is indistinguishable from luck, no matter\n"
        "  how good it looks.\n"
    )


def main() -> None:
    df = load()
    print(f"\ndata: {len(df):,} bars  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}\n")
    experiment_dispersion(df)
    experiment_edge_detection(df)
    experiment_noise_floor(df)


if __name__ == "__main__":
    main()
