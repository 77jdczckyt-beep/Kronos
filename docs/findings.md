# Findings: does Kronos support a trading strategy?

**No. Four directions were tested. None survived a fair baseline or a holdout.**

This document exists so the negative result is not re-discovered at cost later.
The infrastructure works; what is missing is a signal to put through it.

## Summary

| Direction | Result | Why it failed |
| --- | --- | --- |
| Direction, 1h horizon (Alibaba HK 5-min) | rank IC −0.055, inside baseline spread | No skill |
| Direction, 5d horizon (US daily) | rank IC +0.0423, z=+0.96 | Inside baseline spread |
| Direction, 20d horizon (US daily) | rank IC +0.2624, z=+4.73 | Mean reversion in disguise; a 4-line rule beat it, and that rule failed holdout |
| Volatility (US daily) | rank IC +0.598 train / +0.483 holdout | Worse than a plain trailing standard deviation |

## The recurring mistake

Three of the four promising results came from a null that was too weak. This is
the single most useful lesson here:

- The **bootstrap baseline extrapolates recent drift**, making it
  momentum-flavoured. Momentum scored −0.22 in the test window, so a
  mean-reverting signal was being measured against a null that was actively
  anti-correlated with it. That manufactured a z of +4.73 out of nothing.
- The **first volatility baseline used a 20-day window** while Kronos read 256
  days. Most of the resulting "+0.3233 incremental signal" was the advantage of
  a longer estimation window, not model skill.

A model beating a random baseline has shown almost nothing. It has to beat the
cheap, well-known effect — trailing volatility, mean reversion, momentum — or
the cheap effect is the better tool: faster, transparent, and with understood
failure modes.

## The mean-reversion episode, in detail

Worth reading before trusting any backtest.

`MeanReversionForecaster(window=60)` is four lines: if price is below its
60-day average, predict up. On the training split (2016–2023, US daily,
20-day horizon) it produced:

```
hit rate              60.8%     ("always predict up" scores 51.5%)
avg move when RIGHT   9.02%
avg move when WRONG   6.42%
mean P&L per trade   +2.96%
cumulative          +385.0%
worst single trade   −38.2%
```

Good on *every* dimension — right more often than chance, right on the bigger
moves, profitable per trade. There was no tell in these numbers.

On the holdout block (2023-05 onward, never used for selection): hit rate
**49.1%**, rank IC +0.093. A coin flip.

It was a regime, not a signal. INTC, T and PFE chopped sideways through the
training window, which is exactly the condition mean reversion collects; the
following period did not. Nothing in the training statistics warned of this.
Only the holdout caught it.

## Data quality notes

- **Robinhood hourly bars are mostly synthetic.** Of 3,852 hourly F bars, only
  897 were real; the rest carried `interpolated: true`, repeating the previous
  price at zero volume. Feeding those to a forecaster invents flat stretches and
  depresses measured volatility. Daily bars are clean (1 dropped in 2,656).
  `scripts/convert_rh_bars.py` filters both interpolated and zero-volume bars.
- **HuggingFace needs two domains** on a restricted network: `huggingface.co`
  *and* `hf.co`. Weights are served from `cas-server.xethub.hf.co` and
  `us.aws.cdn.hf.co`. Allowing only the first lets metadata resolve while every
  download fails with a 403.
- **MCP tool names are not stable.** The Robinhood server reconnected mid-session
  under a UUID prefix rather than its original name. Any broker adapter must
  discover its tools at runtime rather than hardcoding a prefix.

## What is worth keeping

- `edge.py` / `run_edge_sweep.py` — skill measurement with baselines and holdout
- `trivial.py` — the nulls a model must beat
- `risk.py`, `broker.py` — caps, exit rules, and the arming gate
- `convert_rh_bars.py` — clean US equity bars
- 49 tests

The harness caught three false positives in one session. That is what it is
for, and it will work for any forecaster, not just Kronos.

## Recommendation

Do not fund this. There is no measured edge to trade, with options or shares.

At the account size discussed ($80 cash, $75 max risk), even a genuine
IC of 0.03 would not be expressible — a real edge needs many small positions,
and $80 affords one contract of one thing.

If this is revisited, the honest next step is a signal with published,
replicated evidence behind it — not another sweep over this model's
hyperparameters. Continuing to search the same space guarantees that noise
eventually survives by luck; this session produced three examples of how
readily that happens.
