"""Does Kronos add anything to a trailing volatility estimate?

Volatility clusters, so any sane forecaster scores well against realised
volatility. The raw correlation is therefore not evidence of anything. Two
things decide whether a model is useful here:

1. Does it beat a trailing estimate outright?
2. After a trailing estimate has had its say, does it explain any of the
   remainder?

**The baseline must see the same history the model does.** An earlier version
of this script compared Kronos against a 20-day trailing window while handing
Kronos 256 days, and the resulting "incremental signal" was largely just the
advantage of a longer estimation window. The baseline here uses the full
lookback, with shorter windows and an EWMA reported alongside for context.

Per-point estimates are written to CSV so follow-up questions do not require
re-running two hours of forecasts.
"""

from __future__ import annotations

import glob
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from kronos_trader.forecast import KronosEnsembleForecaster  # noqa: E402

LOOKBACK, HORIZON, N_PATHS, POINTS = 256, 20, 24, 22
HOLDOUT_FRAC = 0.30
OUT = pathlib.Path("data/vol_study.csv")


def rank_ic(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = pd.Series(a[ok]).rank(), pd.Series(b[ok]).rank()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def residual_ic(predictor, baseline, realised) -> float:
    """Rank IC of `predictor` against what `baseline` leaves unexplained."""
    lb, lr = np.log(baseline), np.log(realised)
    slope, intercept = np.polyfit(lb, lr, 1)
    return rank_ic(np.log(predictor), lr - (slope * lb + intercept))


def collect(frames: dict[str, pd.DataFrame], fc, split: str) -> pd.DataFrame:
    rows, t0 = [], time.time()
    for sym, df in frames.items():
        closes = df["close"].to_numpy(dtype=float)
        start, end = LOOKBACK, len(df) - HORIZON - 1
        if end <= start:
            continue
        spacing = max(HORIZON, (end - start) // POINTS)
        for i in list(range(start, end, spacing))[:POINTS]:
            hist = df.iloc[i - LOOKBACK : i]
            hz = df.index[i : i + HORIZON]

            ens = fc.forecast(sym, hist, hz, n_paths=N_PATHS)
            k_vol = ens.forecast_volatility() / np.sqrt(HORIZON)

            past = np.diff(np.log(closes[i - LOOKBACK : i]))
            fut = np.diff(np.log(closes[i - 1 : i + HORIZON]))
            r_vol = float(np.std(fut, ddof=1))
            if r_vol <= 0 or k_vol <= 0:
                continue

            rows.append(
                dict(
                    split=split, symbol=sym, idx=i, date=df.index[i],
                    kronos=k_vol,
                    v20=float(np.std(past[-20:], ddof=1)),
                    v60=float(np.std(past[-60:], ddof=1)),
                    v256=float(np.std(past, ddof=1)),
                    ewma=float(pd.Series(past).ewm(halflife=20).std().iloc[-1]),
                    realised=r_vol,
                )
            )
        print(f"  [{split}] {sym}: {len(rows)} rows ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> None:
    df = df.dropna()
    print("\n" + "=" * 68)
    print(f"{label}   n={len(df)}")
    print("=" * 68)
    print("rank IC vs realised volatility (higher is better):")
    for col in ("v256", "ewma", "v60", "v20", "kronos"):
        mark = "  <- model" if col == "kronos" else ""
        print(f"  {col:<8}{rank_ic(df[col], df.realised):+.4f}{mark}")

    print("\nincremental information, after v256 (the fair-information baseline):")
    for col in ("kronos", "ewma", "v60", "v20"):
        mark = "  <- model" if col == "kronos" else ""
        print(f"  {col:<8}{residual_ic(df[col], df.v256, df.realised):+.4f}{mark}")

    se = 1.0 / np.sqrt(max(1, len(df) - 3))
    inc = residual_ic(df.kronos, df.v256, df.realised)
    print(f"\napprox SE at this n: {se:.3f}   kronos incremental z = {inc / se:+.2f}")
    if rank_ic(df.kronos, df.realised) < rank_ic(df.v256, df.realised):
        print("NOTE: Kronos scores below the one-line trailing estimate outright.")
    if abs(inc) < 2 * se:
        print("VERDICT: no incremental information over a fair baseline.")
    elif inc > 0:
        print("VERDICT: incremental signal present. Check it replicates on holdout.")
    else:
        print("VERDICT: negative incremental value.")


def main() -> None:
    frames = {}
    for p in sorted(glob.glob("data/us_equity/*_day.csv")):
        s = p.split("/")[-1].split("_")[0]
        d = pd.read_csv(p)
        d["timestamps"] = pd.to_datetime(d["timestamps"])
        frames[s] = d.set_index("timestamps")

    train, hold = {}, {}
    for sym, df in frames.items():
        cut = int(len(df) * (1 - HOLDOUT_FRAC))
        train[sym], hold[sym] = df.iloc[:cut], df.iloc[cut:]

    from model import Kronos, KronosTokenizer, KronosPredictor

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    fc = KronosEnsembleForecaster(
        KronosPredictor(mdl, tok, device="cpu", max_context=512), batch_size=N_PATHS
    )

    parts = [collect(train, fc, "train"), collect(hold, fc, "holdout")]
    data = pd.concat(parts, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(data)} rows)")

    for split in ("train", "holdout"):
        sub = data[data.split == split]
        if len(sub) > 10:
            report(sub, f"VOLATILITY STUDY -- {split}")


if __name__ == "__main__":
    main()
