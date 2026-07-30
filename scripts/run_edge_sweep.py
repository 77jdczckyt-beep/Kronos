"""Sweep forecast configurations and report which, if any, beats noise.

    python scripts/run_edge_sweep.py

Runs several (model, horizon) configurations over pooled US equity daily bars,
each against its own skill-free baseline.

**On testing several configurations at once.** Every extra configuration is
another chance for noise to look like signal. Running three and reporting the
best is not the same as testing one and finding it works: with three
independent tests, the chance that at least one clears a 5% threshold by luck
alone is about 14%, not 5%.

So the bar here is deliberately raised. A configuration is only called
interesting if its rank IC clears the baseline spread by |z| >= 2 AND it is
positive AND it survives on a held-out block of data that was not used to pick
it. The holdout is the real test; the sweep only nominates a candidate.
"""

from __future__ import annotations

import glob
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from kronos_trader.edge import EdgeResult, evaluate  # noqa: E402
from kronos_trader.forecast import (  # noqa: E402
    BootstrapForecaster,
    KronosEnsembleForecaster,
)

N_PATHS = 32
POINTS_PER_SYMBOL = 26
BASELINE_SEEDS = 12
LOOKBACK = 256
HOLDOUT_FRAC = 0.30  # most recent 30% of each series is reserved

CONFIGS = [
    ("kronos-small h5", "NeoQuasar/Kronos-small", 5),
    ("kronos-small h20", "NeoQuasar/Kronos-small", 20),
    ("kronos-base h5", "NeoQuasar/Kronos-base", 5),
]


def load_symbols(interval: str = "day") -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(glob.glob(f"data/us_equity/*_{interval}.csv")):
        sym = path.split("/")[-1].split("_")[0]
        df = pd.read_csv(path)
        df["timestamps"] = pd.to_datetime(df["timestamps"])
        out[sym] = df.set_index("timestamps")
    return out


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(df) * (1 - HOLDOUT_FRAC))
    return df.iloc[:cut], df.iloc[cut:]


def pooled(
    frames: dict[str, pd.DataFrame],
    forecaster_factory,
    name: str,
    horizon: int,
    points_per_symbol: int,
    verbose: bool = False,
) -> EdgeResult:
    """Evaluate across symbols and pool the observations into one result."""
    pooled_result = EdgeResult(name=name)
    for sym, df in frames.items():
        if len(df) < LOOKBACK + horizon + 10:
            continue
        r = evaluate(
            df, forecaster_factory(), f"{name}-{sym}",
            LOOKBACK, horizon, N_PATHS, points_per_symbol, symbol=sym,
        )
        pooled_result.predicted += r.predicted
        pooled_result.realised += r.realised
        pooled_result.prob_up += r.prob_up
        if verbose:
            print(f"    {sym}: n={r.n} IC={r.ic_spearman():+.4f}", flush=True)
    return pooled_result


def baseline_band(frames, horizon: int, points: int) -> np.ndarray:
    ics = []
    for seed in range(BASELINE_SEEDS):
        r = pooled(frames, lambda s=seed: BootstrapForecaster(seed=s),
                   f"bootstrap-{seed}", horizon, points)
        ics.append(r.ic_spearman())
    return np.array([x for x in ics if np.isfinite(x)])


def main() -> None:
    frames = load_symbols("day")
    if not frames:
        raise SystemExit("no data in data/us_equity -- run convert_rh_bars.py first")

    train, hold = {}, {}
    for sym, df in frames.items():
        a, b = split(df)
        train[sym], hold[sym] = a, b
    print("US equity daily bars")
    for sym, df in frames.items():
        print(f"  {sym:<6} {len(df):>5} bars  train {len(train[sym])}  holdout {len(hold[sym])}")
    print()

    from model import Kronos, KronosTokenizer, KronosPredictor

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    loaded: dict[str, object] = {}

    def get_predictor(repo: str):
        if repo not in loaded:
            mdl = Kronos.from_pretrained(repo)
            loaded[repo] = KronosPredictor(mdl, tok, device="cpu", max_context=512)
        return loaded[repo]

    results = []
    holdout_done: set[str] = set()
    for label, repo, horizon in CONFIGS:
        print(f"--- {label} ---", flush=True)
        t0 = time.time()
        band = baseline_band(train, horizon, POINTS_PER_SYMBOL)
        print(f"  baseline rank IC: mean {band.mean():+.4f} sd {band.std(ddof=1):.4f} "
              f"range {band.min():+.4f} to {band.max():+.4f}", flush=True)

        pred = get_predictor(repo)
        r = pooled(
            train,
            lambda p=pred: KronosEnsembleForecaster(p, batch_size=N_PATHS),
            label, horizon, POINTS_PER_SYMBOL, verbose=True,
        )
        z = (r.ic_spearman() - band.mean()) / band.std(ddof=1) if band.std(ddof=1) > 0 else np.nan
        print(f"  {label}: n={r.n} rank IC={r.ic_spearman():+.4f} "
              f"hit={r.hit_rate():.1%} z={z:+.2f}  ({time.time() - t0:.0f}s)", flush=True)
        results.append((label, repo, horizon, r, band, z))

        # Run the holdout immediately, not after every config. Sequencing the
        # decisive check behind the slowest job means a timeout kills the one
        # result that mattered -- which is exactly what happened the first time.
        if np.isfinite(z) and z >= 2.0 and r.ic_spearman() > 0:
            print(f"  {label} cleared the bar -- testing holdout now", flush=True)
            band_h = baseline_band(hold, horizon, POINTS_PER_SYMBOL)
            rh = pooled(hold, lambda p=pred: KronosEnsembleForecaster(p, batch_size=N_PATHS),
                        f"{label}-holdout", horizon, POINTS_PER_SYMBOL)
            zh = ((rh.ic_spearman() - band_h.mean()) / band_h.std(ddof=1)
                  if band_h.std(ddof=1) > 0 else np.nan)
            verdict = "HOLDS UP" if np.isfinite(zh) and zh >= 2.0 else "did not replicate"
            print(f"  {label} HOLDOUT: rank IC={rh.ic_spearman():+.4f} "
                  f"hit={rh.hit_rate():.1%} z={zh:+.2f}  {verdict}", flush=True)
            holdout_done.add(label)
        print(flush=True)

    print("=" * 72)
    print("SWEEP RESULTS (training split)")
    print("=" * 72)
    print(f"{'config':<20}{'n':>5}{'rank IC':>10}{'hit':>8}{'baseline':>11}{'z':>8}")
    for label, _repo, _h, r, band, z in results:
        print(f"{label:<20}{r.n:>5}{r.ic_spearman():>+10.4f}{r.hit_rate():>8.1%}"
              f"{band.mean():>+11.4f}{z:>+8.2f}")
    print()

    winners = [x for x in results
               if np.isfinite(x[5]) and x[5] >= 2.0 and x[3].ic_spearman() > 0
               and x[0] not in holdout_done]
    if not winners:
        print("No configuration cleared the baseline spread on the training split.")
        print("Nothing to take to holdout. The forecast-driven approach does not")
        print("show an edge on this data at these settings.")
        return

    print(f"{len(winners)} candidate(s) cleared the bar. Testing on the held-out")
    print("block, which was not used to select them.\n")
    for label, repo, horizon, _r, _band, _z in winners:
        band_h = baseline_band(hold, horizon, POINTS_PER_SYMBOL)
        rh = pooled(hold, lambda p=get_predictor(repo): KronosEnsembleForecaster(p, batch_size=N_PATHS),
                    f"{label}-holdout", horizon, POINTS_PER_SYMBOL)
        zh = ((rh.ic_spearman() - band_h.mean()) / band_h.std(ddof=1)
              if band_h.std(ddof=1) > 0 else np.nan)
        verdict = "HOLDS UP" if np.isfinite(zh) and zh >= 2.0 else "did not replicate"
        print(f"{label:<20} holdout rank IC={rh.ic_spearman():+.4f} z={zh:+.2f}  {verdict}")


if __name__ == "__main__":
    main()
