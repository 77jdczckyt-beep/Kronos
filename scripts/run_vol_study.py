"""Can Kronos forecast volatility, and does it add anything to trailing vol?

Direction is close to unpredictable at these horizons; volatility is not.
Volatility clusters -- calm follows calm, turbulence follows turbulence -- which
is among the most robust empirical facts in finance. So a forecaster can look
impressive at predicting volatility while contributing nothing, because simply
reporting *last month's* volatility already scores well.

That makes the raw correlation useless as evidence. The test that matters is
**incremental**: after trailing realised volatility has had its say, does the
model explain any of what is left?

Implemented by regressing log realised volatility on log trailing volatility
and correlating the model's forecast against the residual. If that correlation
is near zero, the model is re-deriving persistence and adding nothing -- and a
one-line trailing-vol estimate is the better tool.
"""

from __future__ import annotations

import glob
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from kronos_trader.forecast import KronosEnsembleForecaster  # noqa: E402

LOOKBACK, HORIZON, N_PATHS, POINTS = 256, 20, 24, 22


def rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = pd.Series(a).rank(), pd.Series(b).rank()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    frames = {}
    for p in sorted(glob.glob("data/us_equity/*_day.csv")):
        s = p.split("/")[-1].split("_")[0]
        d = pd.read_csv(p)
        d["timestamps"] = pd.to_datetime(d["timestamps"])
        frames[s] = d.set_index("timestamps")

    from model import Kronos, KronosTokenizer, KronosPredictor

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(mdl, tok, device="cpu", max_context=512)
    fc = KronosEnsembleForecaster(predictor, batch_size=N_PATHS)

    kron, trail, real = [], [], []
    t0 = time.time()
    for sym, df in frames.items():
        closes = df["close"].to_numpy(dtype=float)
        start, end = LOOKBACK, len(df) - HORIZON - 1
        if end <= start:
            continue
        spacing = max(HORIZON, (end - start) // POINTS)
        pts = list(range(start, end, spacing))[:POINTS]
        for i in pts:
            hist = df.iloc[i - LOOKBACK : i]
            hz = df.index[i : i + HORIZON]

            ens = fc.forecast(sym, hist, hz, n_paths=N_PATHS)
            # Ensemble dispersion of terminal returns, per-bar.
            k_vol = ens.forecast_volatility() / np.sqrt(HORIZON)

            past = np.diff(np.log(closes[i - LOOKBACK : i]))
            t_vol = float(np.std(past[-HORIZON:], ddof=1))

            fut = np.diff(np.log(closes[i - 1 : i + HORIZON]))
            r_vol = float(np.std(fut, ddof=1))

            if min(k_vol, t_vol, r_vol) <= 0:
                continue
            kron.append(k_vol)
            trail.append(t_vol)
            real.append(r_vol)
        print(f"  {sym}: {len(kron)} points so far ({time.time() - t0:.0f}s)", flush=True)

    k, t, r = np.array(kron), np.array(trail), np.array(real)
    lk, lt, lr = np.log(k), np.log(t), np.log(r)

    print("\n" + "=" * 68)
    print(f"VOLATILITY STUDY  n={len(k)}  horizon={HORIZON}d  paths={N_PATHS}")
    print("=" * 68)
    print(f"trailing vol -> realised vol   rank IC = {rank_ic(t, r):+.4f}")
    print(f"kronos   vol -> realised vol   rank IC = {rank_ic(k, r):+.4f}")
    print("  (both should be high -- volatility is persistent, so this is easy)")

    # Residualise: what trailing vol cannot explain.
    slope, intercept = np.polyfit(lt, lr, 1)
    resid = lr - (slope * lt + intercept)
    inc = rank_ic(lk, resid)
    print(f"\nkronos vol -> residual after trailing vol: rank IC = {inc:+.4f}")
    print(f"  regression: log(realised) = {slope:+.3f}*log(trailing) {intercept:+.3f}")

    # How much of Kronos's forecast is just trailing vol restated?
    print(f"\nkronos vol vs trailing vol     rank IC = {rank_ic(k, t):+.4f}")
    print("  (near 1.0 means the model is echoing trailing vol)")

    se = 1.0 / np.sqrt(max(1, len(k) - 3))
    print(f"\napprox SE on a rank IC at this n: {se:.3f}")
    print()
    if abs(inc) < 2 * se:
        print("VERDICT: no incremental information. Kronos volatility adds nothing")
        print("beyond trailing realised volatility, which is one line of code.")
    elif inc > 0:
        print("VERDICT: some incremental signal. Needs a holdout before it counts.")
    else:
        print("VERDICT: negative incremental signal -- check for a bug before")
        print("reading anything into it.")


if __name__ == "__main__":
    main()
