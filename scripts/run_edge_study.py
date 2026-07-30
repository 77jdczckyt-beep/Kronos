"""Run the edge study: Kronos against a skill-free baseline.

    python scripts/run_edge_study.py

Writes results to stdout. Takes roughly 40 minutes on CPU.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from kronos_trader.edge import evaluate, noise_band  # noqa: E402
from kronos_trader.forecast import BootstrapForecaster, KronosEnsembleForecaster  # noqa: E402

LOOKBACK, HORIZON, N_PATHS, N_POINTS = 256, 12, 32, 120
BASELINE_SEEDS = 12


def main() -> None:
    df = pd.read_csv("finetune_csv/data/HK_ali_09988_kline_5min_all.csv")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.set_index("timestamps")
    print(f"data: {len(df):,} bars  {df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}")
    print(f"config: lookback={LOOKBACK} horizon={HORIZON} paths={N_PATHS} "
          f"points={N_POINTS}\n", flush=True)

    print("--- baseline (skill-free bootstrap) ---", flush=True)
    baselines = []
    t0 = time.time()
    for seed in range(BASELINE_SEEDS):
        r = evaluate(df, BootstrapForecaster(seed=seed), f"bootstrap-{seed}",
                     LOOKBACK, HORIZON, N_PATHS, N_POINTS)
        baselines.append(r)
        print("  " + r.summary(), flush=True)
    print(f"\n{noise_band(baselines)}")
    print(f"(baseline took {time.time() - t0:.0f}s)\n", flush=True)

    print("--- Kronos ---", flush=True)
    from model import Kronos, KronosTokenizer, KronosPredictor

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(mdl, tok, device="cpu", max_context=512)

    t0 = time.time()
    kr = evaluate(
        df,
        KronosEnsembleForecaster(predictor, temperature=1.0, top_p=0.9, batch_size=N_PATHS),
        "kronos-small", LOOKBACK, HORIZON, N_PATHS, N_POINTS, verbose=True,
    )
    print("\n" + kr.summary())
    print(f"(kronos took {time.time() - t0:.0f}s)\n")

    ics = np.array([b.ic_pearson() for b in baselines])
    k_ic = kr.ic_pearson()
    z = (k_ic - ics.mean()) / ics.std(ddof=1) if ics.std(ddof=1) > 0 else float("nan")
    better = int((ics >= k_ic).sum())

    print("=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"Kronos IC        {k_ic:+.4f}   (t={kr.ic_tstat():+.2f})")
    print(f"baseline IC      {ics.mean():+.4f} +/- {ics.std(ddof=1):.4f}")
    print(f"z vs baseline    {z:+.2f}")
    print(f"baseline seeds matching or beating Kronos: {better}/{len(ics)}")
    print()
    if not np.isfinite(k_ic) or abs(kr.ic_tstat()) < 2.0:
        print("Kronos IC is not statistically distinguishable from zero.")
    if better > 0:
        print(f"{better} skill-free seed(s) did as well or better. That is what")
        print("luck looks like at this sample size.")
    if abs(kr.ic_tstat()) >= 2.0 and better == 0 and k_ic > 0:
        print("Kronos beat every baseline seed with a significant IC.")
        print("Worth a second look on different data before trusting it.")


if __name__ == "__main__":
    main()
