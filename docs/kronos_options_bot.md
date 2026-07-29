# Kronos options signal pipeline

A distribution-based options signal layer built on Kronos forecasts, with hard
risk limits and a validation harness.

**Status: not connected to any brokerage. No live trading path is wired up.**
See [Execution](#execution) for why, and what would have to be true first.

## Why this exists in this shape

`KronosPredictor.predict` averages its sampled trajectories before returning
them (`preds = np.mean(preds, axis=1)`, `model/kronos.py`). That is a
reasonable point forecast and the wrong input for options.

Averaging preserves whatever drift the paths share but cancels their
independent noise, so dispersion shrinks by roughly `sqrt(N)`. Measured on
93,912 bars of 5-minute Alibaba HK data:

```
spread of individual paths : 5.372
spread of 32-path averages : 0.989
collapse factor            : 5.4x   (sqrt(32) = 5.7)
```

An option is priced on that spread. Feeding the averaged forecast into an
options model would systematically understate the chance of a large move.

So this package never calls `predict`. `KronosEnsembleForecaster` calls
`predict_batch` with the same series repeated N times at `sample_count=1`,
which yields N independent trajectories from one batched forward pass, and
keeps all of them. Everything downstream consumes the raw `(n_paths, horizon)`
matrix.

## Modules

| Module | Role |
| --- | --- |
| `forecast.py` | `PathEnsemble` + Kronos and bootstrap forecasters |
| `pricing.py` | Black-Scholes, implied-vol inversion, quote types |
| `signals.py` | Ensemble + option chain → accept/reject with a reason |
| `risk.py` | Hard caps, position sizing, exit rules |
| `backtest.py` | Walk-forward loop against a simulated option chain |
| `validate.py` | The three experiments below |

## The filter bug worth knowing about

The first version of `signals.py` capped `prob_itm` at 70% and required
forecast volatility to exceed implied volatility. Both looked sensible. Run
against a forecaster with *perfect knowledge of the future*, the pipeline took
**1 trade in 400 decisions and lost money**.

Both filters tested the shape of the model's distribution when the
economically correct test is modelled value against market price:

- A confident forecast produces `P(ITM)` near 100%. Capping it rejected
  precisely the signals worth trading. Moneyness — "am I paying mostly
  intrinsic value?" — is a property of the *contract* (strike vs spot), not of
  the model's confidence. Those are different quantities and conflating them
  vetoed every high-conviction call.
- Requiring forecast vol > implied vol vetoes every purely *directional* edge.
  If you know the direction, a call is profitable even when implied vol is
  rich; you only need the move to beat the premium. That filter is now a loose
  sanity guard at 0.6x, not an economic requirement.

`edge_ratio` (expected payoff under the forecast ÷ ask) is the primary test and
is economically complete. Everything else is a liquidity or robustness guard.
The regression tests for this are in `tests/test_pipeline.py`.

## Validation

`python -m kronos_trader.validate`

**Edge detection** — does the pipeline trade when edge exists? Without this, a
filter that rejects everything is indistinguishable from a broken one.

```
skill=0.0  trades=49   P&L=$      284  RoR=  +8.6%  win=12.2%
skill=0.3  trades=19   P&L=$    3,323  RoR=+256.9%  win=36.8%
skill=0.6  trades=30   P&L=$    9,120  RoR=+434.1%  win=40.0%
skill=1.0  trades=52   P&L=$   13,138  RoR=+356.1%  win=23.1%
```

**Noise floor** — the same forecaster with *no* skill, across 15 seeds:

```
mean return on risk : -23.0%
median              : -100.0%
std across seeds    : 130.6%
best / worst seed   : +320.0% / -100.0%
profitable seeds    : 5/15
t-stat vs zero      : -0.68
```

This is the most important table here. A forecaster with **zero** predictive
skill produced anything from −100% to **+320%** over 400 decisions. The mean is
negative, as theory says it must be when you buy options at 1.15× fair
volatility — but with a t-stat of −0.68 it is not statistically distinguishable
from zero even at this sample size.

Consequences, stated plainly:

- A profitable month proves nothing. It sits comfortably inside the range pure
  luck produces.
- Even the *perfect oracle* won only 23–40% of its trades. Long options lose
  most of the time and pay for it with rare large winners; that is the shape of
  the payoff, not a sign of a broken strategy.
- Anyone claiming this kind of strategy "makes money fast" is describing the
  best seed of a wide distribution.

## Backtest caveats

The harness is honest about being a simulation:

- **There is no historical options data in this repo.** The chain is synthesised
  with Black-Scholes at trailing realised vol × `vol_risk_premium` (default
  1.15). That markup encodes the empirically robust fact that implied vol
  trades above subsequent realised vol. It is the single most important
  parameter and the headwind every long-premium strategy faces.
- Positions settle at intrinsic value at horizon end. No early take-profit or
  stop-out, so live behaviour will differ in both directions.
- A skill-free forecaster *must* lose money here. If a strategy shows profit,
  suspect the harness before believing the edge.

## Risk limits

`RiskBook` gates every entry and can be overridden by no signal, however
strong: per-trade / per-day / total-open dollar caps, max entries per day, max
open positions, one position per underlying. Exits are deliberately **never**
gated — blocking a close because a daily cap was hit would turn a risk control
into a trap.

## Execution

There is no order-placement code in this package, deliberately.

- The Robinhood MCP server in this environment is unauthorised, and OAuth
  cannot be completed from a non-interactive session.
- This environment's network policy blocks all external hosts —
  `huggingface.co` included — so the Kronos weights cannot be downloaded and
  **the Kronos path in `forecast.py` has never been executed**. It is written
  against the `predict_batch` signature but is untested. Everything else is
  covered by the 23 tests, which run on the bootstrap forecaster.
- No forecaster in this repo has demonstrated edge on real data. The bootstrap
  baseline is noise by construction, and the oracle cheats.

Before any live path is worth building, in order: reachable weights → Kronos
run through the noise-floor experiment on real data → a result that clears the
luck band by a wide margin → paper trading → and only then a funded account,
with a human approving orders until the live numbers match the backtest.
