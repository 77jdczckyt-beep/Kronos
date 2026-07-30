"""Choosing which tickers to scan.

Two jobs, and the second one is the one that matters on a small account.

**Liquidity screening** keeps the universe to names whose options can actually
be traded out of. An edge measured in tens of basis points does not survive a
wide market.

**Affordability screening** rejects a symbol when nothing in its chain fits the
risk budget *without* being a lottery ticket. This is the screen that stops a
small account from drifting into far-OTM weeklies simply because they are the
only contracts it can afford. A symbol that offers nothing sensible under the
budget is dropped, and the reason is recorded -- it is not a failure to be
worked around by loosening the filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .data import MarketData, MarketDataError
from .pricing import OptionQuote

# Liquid, heavily-optioned US names cheap enough that a sub-$1.00 contract is
# not automatically far out of the money. A starting point, not a strategy.
DEFAULT_SEEDS: tuple[str, ...] = ("F", "SOFI", "INTC", "T", "PFE")


@dataclass(frozen=True)
class ScreenCriteria:
    """Filters applied before a symbol is worth forecasting.

    Attributes:
        min_open_interest: Contract-level liquidity floor.
        max_spread_frac: Widest acceptable bid-ask as a fraction of mid.
        min_dte_days / max_dte_days: Expiry window. Very short-dated contracts
            are dominated by theta; very long-dated ones tie up the budget.
        budget: Dollars available for one position. A symbol survives only if
            some contract in its chain costs no more than this.
        min_prob_itm_proxy: Moneyness floor used *before* any forecast exists,
            as a cheap stand-in for "is this a lottery ticket". A contract more
            than this fraction out of the money is dropped without spending a
            forward pass on it.
    """

    min_open_interest: int = 100
    max_spread_frac: float = 0.20
    min_dte_days: int = 7
    max_dte_days: int = 45
    budget: float = 75.0
    min_prob_itm_proxy: float = 0.08

    def __post_init__(self) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.min_dte_days > self.max_dte_days:
            raise ValueError("min_dte_days cannot exceed max_dte_days")


@dataclass
class ScreenResult:
    kept: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"universe: {', '.join(self.kept) if self.kept else '(empty)'}"]
        for sym, why in self.dropped.items():
            lines.append(f"  dropped {sym}: {why}")
        return "\n".join(lines)


def affordable_quotes(
    quotes: list[OptionQuote],
    spot: float,
    criteria: ScreenCriteria,
) -> list[OptionQuote]:
    """Contracts that are liquid, sensibly struck, and within budget."""
    out = []
    for q in quotes:
        if q.ask <= 0:
            continue
        if q.ask * 100.0 > criteria.budget:
            continue
        if q.open_interest < criteria.min_open_interest:
            continue
        if q.spread_frac > criteria.max_spread_frac:
            continue
        dte = q.expiry_years * 365.0
        if not (criteria.min_dte_days <= dte <= criteria.max_dte_days):
            continue
        # How far out of the money, as a fraction of spot.
        otm = (
            (q.strike - spot) / spot if q.right == "call" else (spot - q.strike) / spot
        )
        if otm > 0 and otm > (1.0 - criteria.min_prob_itm_proxy):
            continue
        if otm > 0.15:  # beyond ~15% OTM at these tenors is a lottery ticket
            continue
        out.append(q)
    return out


def screen(
    symbols: list[str],
    market: MarketData,
    criteria: ScreenCriteria | None = None,
    now: datetime | None = None,
) -> ScreenResult:
    """Reduce `symbols` to those worth forecasting, with reasons for the rest."""
    crit = criteria or ScreenCriteria()
    result = ScreenResult()

    for symbol in symbols:
        try:
            spot = market.spot(symbol, as_of=now)
            quotes = list(market.chain(symbol, as_of=now))
        except MarketDataError as e:
            result.dropped[symbol] = f"data unavailable: {e}"
            continue

        if not quotes:
            result.dropped[symbol] = "empty option chain"
            continue

        usable = affordable_quotes(quotes, spot, crit)
        if not usable:
            cheapest = min((q.ask * 100.0 for q in quotes if q.ask > 0), default=None)
            if cheapest is None:
                result.dropped[symbol] = "no contract has an ask"
            elif cheapest > crit.budget:
                result.dropped[symbol] = (
                    f"cheapest contract ${cheapest:.0f} exceeds ${crit.budget:.0f} budget"
                )
            else:
                result.dropped[symbol] = (
                    "nothing in budget is both liquid and sensibly struck "
                    "-- only lottery tickets fit"
                )
            continue

        result.kept.append(symbol)

    return result
