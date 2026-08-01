"""Live account settings.

Kept in one place so the numbers that govern real money are visible together
rather than scattered across defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .risk import ExitRules, RiskLimits
from .signals import SignalConfig
from .universe import DEFAULT_SEEDS, ScreenCriteria

# Cash available to the strategy. The existing long call in the account is a
# position, not buying power -- it contributes nothing here until it is sold.
ACCOUNT_CASH = 80.0

# Most that may be lost on one position. Must fit inside ACCOUNT_CASH with
# slack; a cap above available cash can never fill.
MAX_RISK_PER_TRADE = 75.0


@dataclass(frozen=True)
class LiveConfig:
    """The full parameter set for a live or dry-run session.

    Note the tension this configuration sits in: at $75 a contract must be
    priced at $0.75/share or less, and on liquid US names that mostly means far
    out of the money or nearly expired. `SignalConfig.min_prob_itm` and the
    universe screen both reject those. The expected outcome is therefore very
    few trades, and that is correct behaviour rather than a misconfiguration.
    Loosening the filters to force activity would trade away the only thing
    protecting the account.
    """

    seeds: tuple[str, ...] = DEFAULT_SEEDS
    screen_own_universe: bool = True

    limits: RiskLimits = field(
        default_factory=lambda: RiskLimits(
            max_risk_per_trade=MAX_RISK_PER_TRADE,
            max_risk_per_day=MAX_RISK_PER_TRADE,
            max_open_risk=MAX_RISK_PER_TRADE,
            max_entries_per_day=1,
            max_open_positions=1,
            one_per_underlying=True,
        )
    )
    exits: ExitRules = field(default_factory=ExitRules)
    signals: SignalConfig = field(default_factory=SignalConfig)
    screen: ScreenCriteria = field(
        default_factory=lambda: ScreenCriteria(budget=MAX_RISK_PER_TRADE)
    )

    def __post_init__(self) -> None:
        if self.limits.max_risk_per_trade > ACCOUNT_CASH:
            raise ValueError(
                f"max_risk_per_trade ${self.limits.max_risk_per_trade:.2f} exceeds "
                f"account cash ${ACCOUNT_CASH:.2f}; such an order can never fill"
            )
        if self.screen.budget > self.limits.max_risk_per_trade:
            raise ValueError(
                "screen budget must not exceed max_risk_per_trade, or the screen "
                "will surface contracts the risk book then refuses"
            )
