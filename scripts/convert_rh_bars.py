"""Convert Robinhood `get_equity_historicals` JSON into Kronos-ready CSVs.

    python scripts/convert_rh_bars.py <json-file> [<json-file> ...]

Two things this does that matter for correctness:

- **Drops interpolated bars.** Robinhood marks synthesised gap-fillers with
  ``interpolated: true``. They repeat the previous price at zero volume and
  carry no information. Feeding them to a forecaster invents flat stretches
  that never happened and depresses measured volatility.
- **Drops zero-volume bars.** A bar that printed no shares is not a real
  observation of the market, whatever the flag says.

Output: ``data/us_equity/<SYMBOL>_<interval>.csv`` with the columns Kronos
expects.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

OUT_DIR = pathlib.Path("data/us_equity")


def convert(path: str) -> list[pathlib.Path]:
    payload = json.loads(pathlib.Path(path).read_text())
    written = []
    for series in payload["data"]["results"]:
        symbol = series["symbol"]
        interval = series.get("interval", "unknown")
        bars = series.get("bars") or []
        if not bars:
            print(f"  {symbol}: no bars, skipped")
            continue

        df = pd.DataFrame(bars)
        total = len(df)

        if "interpolated" in df.columns:
            df = df[df["interpolated"] != True]  # noqa: E712 -- may be object dtype
        df = df.rename(
            columns={
                "begins_at": "timestamps",
                "open_price": "open",
                "close_price": "close",
                "high_price": "high",
                "low_price": "low",
            }
        )
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamps"] = pd.to_datetime(df["timestamps"])

        df = df[df["volume"] > 0]
        df = df.dropna(subset=["open", "high", "low", "close"])
        df = df[df["close"] > 0]
        # Kronos expects an `amount` column (turnover).
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
        df = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]]
        df = df.sort_values("timestamps").drop_duplicates("timestamps")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{symbol}_{interval}.csv"
        df.to_csv(out, index=False)
        written.append(out)
        dropped = total - len(df)
        print(
            f"  {symbol:<6} {len(df):>6,} bars kept ({dropped:,} dropped) "
            f"{df['timestamps'].iloc[0]:%Y-%m-%d} -> {df['timestamps'].iloc[-1]:%Y-%m-%d}"
        )
    return written


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        print(f"{p}:")
        convert(p)
