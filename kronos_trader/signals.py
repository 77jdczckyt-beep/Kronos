"""Turn a forecast distribution into a long-option trade decision, or no trade.

The decision rule is deliberately hostile to trading. Buying a single-leg
option is a negative-expectancy bet by default: you pay the spread on the way
in, you pay it again on the way out, and theta works against you the whole
time. The only justification for entering is that the forecast distribution is
meaningfully *wider or more skewed* than the one the market has priced into
implied volatility -- not merely that the forecast points up.

Every filter here exists to kill trades. If that leaves nothing on most days,
the module is working as intended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .forecast import PathEnsemble
from .pricing import OptionQuote, expected_payoff, implied_vol

Side = Literal["call", "put"]


@dataclass(frozen=True)
class SignalConfig:
    """Thresholds governing entry. Defaults are intentionally strict.

    Attributes:
        min_edge_ratio: Required ratio of modelled expected payoff to the ask
            before a contract qualifies. 1.0 is break-even *before* costs, so
            anything at or near 1.0 is a losing trade in practice. The default
            of 1.35 demands the model see 35% more value than it pays. This is
            the primary and economically complete test: expected payoff under
            the forecast versus what the contract actually costs.
        min_vol_ratio: Loose sanity guard, not an economic requirement. Rejects
            contracts where the forecast sees a *drastically* calmer world than
            implied vol prices. In that regime any apparent edge rests on
            predicted drift rather than on the distribution, which is the
            fragile kind of edge. This is deliberately NOT set above 1.0: a
            purely directional forecast is legitimately tradeable even when
            implied vol is rich, and requiring a vol premium too would veto
            every directional signal.
        max_spread_frac: Reject contracts whose bid-ask spread exceeds this
            fraction of mid. Wide markets eat the entire edge on round trip.
        min_open_interest: Liquidity floor.
        min_prob_itm: Reject lottery tickets -- contracts the model itself
            thinks almost certainly expire worthless, whose apparent edge comes
            from a handful of extreme paths and is therefore an artifact of
            tail estimation from a finite ensemble.
        max_moneyness_itm: Reject contracts more than this fraction *in* the
            money by strike. Deep ITM contracts are mostly intrinsic value and
            buy little convexity per dollar. Note this is a property of the
            contract (strike versus spot), NOT of the model's confidence --
            conflating the two would reject a high-conviction forecast, which
            is precisely the signal worth trading.
        min_paths: Refuse to act on an ensemble too small to estimate a tail.
    """

    min_edge_ratio: float = 1.35
    min_vol_ratio: float = 0.60
    max_spread_frac: float = 0.20
    min_open_interest: int = 100
    min_prob_itm: float = 0.05
    max_moneyness_itm: float = 0.03
    min_paths: int = 32


@dataclass(frozen=True)
class Signal:
    """A qualified trade candidate, or a rejection with its reason."""

    symbol: str
    quote: OptionQuote
    side: Side
    edge_ratio: float
    modelled_value: float
    prob_itm: float
    forecast_vol: float
    implied_vol_: float
    accepted: bool
    reason: str

    @property
    def cost_per_contract(self) -> float:
        """What one contract costs at the ask -- and the most it can lose."""
        return self.quote.ask * 100.0


def annualise_vol(horizon_vol: float, horizon_years: float) -> float:
    """Scale a horizon volatility to annual terms for comparison with IV."""
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    return horizon_vol / np.sqrt(horizon_years)


def evaluate_contract(
    ensemble: PathEnsemble,
    quote: OptionQuote,
    config: SignalConfig | None = None,
) -> Signal:
    """Score one option contract against the forecast distribution.

    Returns a `Signal` with ``accepted`` set and a human-readable ``reason``
    in every case, including rejections, so a run can explain exactly why it
    did nothing.
    """
    cfg = config or SignalConfig()
    side: Side = quote.right

    def reject(reason: str, **kw) -> Signal:
        base = dict(
            symbol=ensemble.symbol,
            quote=quote,
            side=side,
            edge_ratio=0.0,
            modelled_value=0.0,
            prob_itm=0.0,
            forecast_vol=0.0,
            implied_vol_=float("nan"),
            accepted=False,
            reason=reason,
        )
        base.update(kw)
        return Signal(**base)

    if ensemble.n_paths < cfg.min_paths:
        return reject(f"ensemble too small ({ensemble.n_paths} < {cfg.min_paths} paths)")
    if quote.ask <= 0:
        return reject("no ask -- contract not purchasable")
    if quote.spread_frac > cfg.max_spread_frac:
        return reject(f"spread {quote.spread_frac:.1%} > {cfg.max_spread_frac:.0%} of mid")
    if quote.open_interest < cfg.min_open_interest:
        return reject(f"open interest {quote.open_interest} < {cfg.min_open_interest}")

    terminal = ensemble.terminal
    prob_itm = (
        ensemble.prob_above(quote.strike) if side == "call" else ensemble.prob_below(quote.strike)
    )
    modelled = expected_payoff(terminal, quote.strike, side)
    edge_ratio = modelled / quote.ask if quote.ask > 0 else 0.0

    fvol_h = ensemble.forecast_volatility()
    fvol = annualise_vol(fvol_h, quote.expiry_years)
    ivol = implied_vol(quote.mid, ensemble.spot, quote.strike, quote.expiry_years, side)

    common = dict(
        edge_ratio=edge_ratio,
        modelled_value=modelled,
        prob_itm=prob_itm,
        forecast_vol=fvol,
        implied_vol_=ivol,
    )

    # Moneyness is a property of the contract, independent of the forecast.
    itm_frac = (
        (ensemble.spot - quote.strike) / ensemble.spot
        if side == "call"
        else (quote.strike - ensemble.spot) / ensemble.spot
    )
    if itm_frac > cfg.max_moneyness_itm:
        return reject(
            f"strike is {itm_frac:.1%} in the money > {cfg.max_moneyness_itm:.0%} "
            "-- mostly intrinsic, little convexity",
            **common,
        )
    if prob_itm < cfg.min_prob_itm:
        return reject(f"P(ITM) {prob_itm:.1%} < {cfg.min_prob_itm:.0%} -- lottery ticket", **common)
    if edge_ratio < cfg.min_edge_ratio:
        return reject(
            f"edge ratio {edge_ratio:.2f} < {cfg.min_edge_ratio:.2f} required", **common
        )
    if not np.isfinite(ivol):
        return reject("implied vol not solvable -- quote outside no-arbitrage band", **common)
    if fvol < ivol * cfg.min_vol_ratio:
        return reject(
            f"forecast vol {fvol:.1%} is below {cfg.min_vol_ratio:.2f}x implied "
            f"{ivol:.1%} -- forecast sees an implausibly calm market, edge likely spurious",
            **common,
        )

    return Signal(
        symbol=ensemble.symbol,
        quote=quote,
        side=side,
        accepted=True,
        reason=(
            f"edge {edge_ratio:.2f}x, P(ITM) {prob_itm:.1%}, "
            f"forecast vol {fvol:.1%} vs implied {ivol:.1%}"
        ),
        **common,
    )


def best_signal(
    ensemble: PathEnsemble,
    quotes: Sequence[OptionQuote],
    config: SignalConfig | None = None,
) -> tuple[Signal | None, list[Signal]]:
    """Evaluate a chain and return the single best accepted contract.

    Returns:
        ``(best, all_signals)``. ``best`` is ``None`` when nothing qualifies,
        which is the expected outcome most of the time. ``all_signals`` carries
        every evaluation including rejections, for reporting.
    """
    scored = [evaluate_contract(ensemble, q, config) for q in quotes]
    accepted = [s for s in scored if s.accepted]
    if not accepted:
        return None, scored
    # Among qualifying contracts prefer the strongest edge; break ties toward
    # the cheaper ticket, which caps absolute loss.
    best = max(accepted, key=lambda s: (s.edge_ratio, -s.cost_per_contract))
    return best, scored
