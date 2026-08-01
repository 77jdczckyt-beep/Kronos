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

    # Rank IC is the primary metric. Returns are fat-tailed, so Pearson
    # correlation on 120 points can be dominated by two or three extreme moves.
    # When the two disagree sharply, the Pearson figure is the outlier artifact
    # and the rank figure is the honest read -- reporting only Pearson would
    # manufacture a verdict out of noise in either direction.
    p_ics = np.array([b.ic_pearson() for b in baselines])
    r_ics = np.array([b.ic_spearman() for b in baselines])
    k_p, k_r = kr.ic_pearson(), kr.ic_spearman()

    def z_of(x, arr):
        sd = arr.std(ddof=1)
        return (x - arr.mean()) / sd if sd > 0 else float("nan")

    print("=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"{'':<18}{'Kronos':>10}{'baseline mean':>16}{'baseline sd':>14}{'z':>8}")
    print(f"{'rank IC (primary)':<18}{k_r:>+10.4f}{r_ics.mean():>+16.4f}"
          f"{r_ics.std(ddof=1):>14.4f}{z_of(k_r, r_ics):>+8.2f}")
    print(f"{'Pearson IC':<18}{k_p:>+10.4f}{p_ics.mean():>+16.4f}"
          f"{p_ics.std(ddof=1):>14.4f}{z_of(k_p, p_ics):>+8.2f}")
    print(f"{'hit rate':<18}{kr.hit_rate():>10.1%}")
    print()

    if abs(r_ics.mean()) < 0.03 < abs(p_ics.mean()):
        print("NOTE: the baseline's Pearson IC is inflated relative to its rank IC,")
        print("so a few extreme returns are driving it. Its rank IC near zero is")
        print("the truthful statement that the baseline has no skill. Judge Kronos")
        print("on the rank row.")
        print()

    if abs(z_of(k_r, r_ics)) < 2.0:
        print("Kronos rank IC sits inside the skill-free baseline's spread.")
        print("No edge detected. This does not mean Kronos is worse than noise --")
        print("it means neither is distinguishable from zero on this data.")
    elif k_r > 0:
        print("Kronos rank IC clears the baseline spread. Replicate on different")
        print("data before trusting it.")
    else:
        print("Kronos rank IC is below the baseline spread. Check for a sign or")
        print("alignment error before concluding it is anti-predictive.")


if __name__ == "__main__":
    main()
