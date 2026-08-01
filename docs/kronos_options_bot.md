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
| `data.py` | `MarketData` sources; never leaks future bars |
| `broker.py` | Order types, dry-run broker, live arming gate |
| `runner.py` | `scan()` → ranked candidates; `run()` → gated placement |
| `backtest.py` | Walk-forward loop against a simulated option chain |
| `validate.py` | The three experiments below |

Entry scanning is the useful half:

```python
report = scan(["AAPL"], market, forecaster, RiskBook(), now=datetime.now())
print(report.summary())   # candidates, and why each skip happened
```

`run()` adds placement and is inert by default — it uses `DryRunBroker`, which
records the exact orders it would have sent without contacting anything. Live
placement needs a `LiveBroker` with `armed=True` and a per-session order
budget; strategy code cannot arm it. `OrderRequest` has no field capable of
expressing a short option, so selling to open is unrepresentable rather than
merely discouraged.

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

## Does Kronos actually predict anything?

`python scripts/run_edge_study.py`

This is the experiment that should gate funding, and it deliberately does not
simulate a strategy. A strategy adds thresholds, position sizing, and costs —
each a free parameter capable of manufacturing a good-looking result from a
forecaster with no skill. `edge.py` instead measures the raw relationship
between predicted and realised returns.

The metric is the **information coefficient**: correlation between predicted
and realised return across decision points. IC of 0 means no skill. Real
exploitable equity signals typically sit around 0.02–0.05. Anything above 0.15
out-of-sample should be treated as a bug until proven otherwise.

It is always reported against a skill-free bootstrap baseline over many seeds,
because the question is never "is the number positive" but "does it beat
noise". Decision points are spaced at least one horizon apart so outcome
windows never overlap — overlapping windows correlate observations and inflate
significance, which is the most common way a study like this fools itself.

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

`MCPBroker` is an explicit stub that raises `NotImplementedError`. That is the
deliberate choice, not an oversight.

Binding it requires reading the brokerage MCP server's real tool schemas — the
names, argument shapes, and return payloads. Guessing them produces code that
looks complete, reviews cleanly, and fails at the only moment that matters.
This repository already contains one artifact of that failure mode: a scheduled
trigger whose prompt instructed a session to run `python -m cli.main politrade`
and call `get_option_positions`, neither of which was ever committed. The stub's
docstring lists what a real adapter must do; fill it in against the schemas, not
against expectations.

Other blockers as of this writing:

- The Robinhood MCP server in this environment is unauthorised, and OAuth
  cannot be completed from a non-interactive session.
- No forecaster here has demonstrated edge on real data. The bootstrap baseline
  is noise by construction, and the oracle cheats. See the edge study above.

Network access note: reaching HuggingFace needs **both** `huggingface.co` /
`*.huggingface.co` **and** `hf.co` / `*.hf.co` on the environment's allowed
domains. Large files are served from `cas-server.xethub.hf.co` and
`us.aws.cdn.hf.co`, which are on the `hf.co` domain, not `huggingface.co`.
Allowing only the latter lets metadata resolve while every weight download
fails with a 403.

Kronos runs on CPU at roughly 18s per 32-path forecast at a 12-bar horizon,
so a multi-symbol scan is minutes, not seconds.

Suggested order: reachable weights → Kronos run through the noise-floor
experiment on real data → a result that clears the luck band by a wide margin →
dry-run scans against live quotes, comparing proposed orders to what actually
happened → paper trading → and only then a funded account, with orders reviewed
until live numbers match the backtest.
