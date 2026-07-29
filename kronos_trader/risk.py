"""Hard risk limits and exit rules.

This layer is the reason the bot can be described as "limits losses". It is
enforced as a gate that every order passes through, not as advice the strategy
may take. Long options have a naturally bounded loss -- the premium -- so the
job here is to bound the *number* of those losses and to force exits before
theta finishes the job.

Nothing in this module can be overridden by a signal, however strong the
signal claims to be.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable


class RiskViolation(Exception):
    """Raised when an order would breach a hard limit."""


@dataclass(frozen=True)
class RiskLimits:
    """Caps applied to every entry.

    Attributes:
        max_risk_per_trade: Most that can be lost on one position, in dollars.
            For a long option this is the full premium paid.
        max_risk_per_day: Total new premium that may be committed in one day.
        max_open_risk: Total premium at risk across all open positions.
        max_entries_per_day: Cap on new positions opened per day.
        max_open_positions: Cap on simultaneous open positions.
        one_per_underlying: Forbid stacking multiple positions on one ticker,
            which would concentrate risk that the position count hides.
    """

    max_risk_per_trade: float = 150.0
    max_risk_per_day: float = 150.0
    max_open_risk: float = 150.0
    max_entries_per_day: int = 2
    max_open_positions: int = 3
    one_per_underlying: bool = True

    def __post_init__(self) -> None:
        for name in ("max_risk_per_trade", "max_risk_per_day", "max_open_risk"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_entries_per_day", "max_open_positions"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True)
class ExitRules:
    """When to close an open position.

    Attributes:
        take_profit: Close at this fractional gain on premium (0.6 = +60%).
        stop_loss: Close at this fractional loss on premium (0.5 = -50%).
        min_days_to_expiry: Close when expiry is this near. Gamma and theta
            both accelerate into expiry; holding to the last day converts a
            probabilistic edge into a coin flip.
        max_hold_days: Close after this many days regardless. A thesis that
            has not played out has usually been invalidated.
    """

    take_profit: float = 0.60
    stop_loss: float = 0.50
    min_days_to_expiry: int = 2
    max_hold_days: int = 10

    def __post_init__(self) -> None:
        if not 0 < self.stop_loss <= 1:
            raise ValueError("stop_loss must be in (0, 1]")
        if self.take_profit <= 0:
            raise ValueError("take_profit must be positive")


@dataclass
class Position:
    """An open long-option position."""

    symbol: str
    strike: float
    right: str
    expiry: date
    contracts: int
    entry_price: float  # per share
    opened_at: datetime

    @property
    def risk(self) -> float:
        """Maximum loss in dollars -- the full premium paid."""
        return self.entry_price * 100.0 * self.contracts

    def pnl_frac(self, mark: float) -> float:
        """Fractional P&L against entry, given a per-share mark."""
        if self.entry_price <= 0:
            return 0.0
        return (mark - self.entry_price) / self.entry_price

    def exit_reason(self, mark: float, now: datetime, rules: ExitRules) -> str | None:
        """Return why this position should close now, or None to hold."""
        pnl = self.pnl_frac(mark)
        if pnl >= rules.take_profit:
            return f"take profit {pnl:+.0%}"
        if pnl <= -rules.stop_loss:
            return f"stop loss {pnl:+.0%}"
        days_left = (self.expiry - now.date()).days
        if days_left <= rules.min_days_to_expiry:
            return f"{days_left}d to expiry"
        held = (now - self.opened_at).days
        if held >= rules.max_hold_days:
            return f"held {held}d"
        return None


@dataclass
class RiskBook:
    """Tracks exposure and vetoes orders that would breach limits.

    Exits are never gated -- only entries. Blocking a close because a daily cap
    was hit would turn a risk control into a way to be trapped in a loser.
    """

    limits: RiskLimits = field(default_factory=RiskLimits)
    positions: list[Position] = field(default_factory=list)
    _entries_today: int = 0
    _risk_today: float = 0.0
    _current_day: date | None = None

    def _roll_day(self, now: datetime) -> None:
        if self._current_day != now.date():
            self._current_day = now.date()
            self._entries_today = 0
            self._risk_today = 0.0

    @property
    def open_risk(self) -> float:
        return sum(p.risk for p in self.positions)

    def check_entry(self, symbol: str, premium_per_share: float, contracts: int, now: datetime) -> None:
        """Raise `RiskViolation` if this entry breaches any limit."""
        self._roll_day(now)
        if contracts < 1:
            raise RiskViolation("contracts must be >= 1")
        risk = premium_per_share * 100.0 * contracts

        if risk > self.limits.max_risk_per_trade:
            raise RiskViolation(
                f"trade risk ${risk:.2f} exceeds per-trade cap "
                f"${self.limits.max_risk_per_trade:.2f}"
            )
        if self._risk_today + risk > self.limits.max_risk_per_day:
            raise RiskViolation(
                f"daily risk ${self._risk_today + risk:.2f} would exceed cap "
                f"${self.limits.max_risk_per_day:.2f}"
            )
        if self.open_risk + risk > self.limits.max_open_risk:
            raise RiskViolation(
                f"open risk ${self.open_risk + risk:.2f} would exceed cap "
                f"${self.limits.max_open_risk:.2f}"
            )
        if self._entries_today + 1 > self.limits.max_entries_per_day:
            raise RiskViolation(f"already opened {self._entries_today} positions today")
        if len(self.positions) + 1 > self.limits.max_open_positions:
            raise RiskViolation(f"already holding {len(self.positions)} positions")
        if self.one_per_underlying_violated(symbol):
            raise RiskViolation(f"already hold a position in {symbol}")

    def one_per_underlying_violated(self, symbol: str) -> bool:
        return self.limits.one_per_underlying and any(p.symbol == symbol for p in self.positions)

    def max_contracts(self, premium_per_share: float) -> int:
        """Largest contract count that fits every dollar cap. May be 0."""
        if premium_per_share <= 0:
            return 0
        per_contract = premium_per_share * 100.0
        budget = min(
            self.limits.max_risk_per_trade,
            self.limits.max_risk_per_day - self._risk_today,
            self.limits.max_open_risk - self.open_risk,
        )
        return max(0, int(budget // per_contract))

    def record_entry(self, position: Position, now: datetime) -> None:
        """Commit an entry after `check_entry` has passed."""
        self.check_entry(position.symbol, position.entry_price, position.contracts, now)
        self.positions.append(position)
        self._entries_today += 1
        self._risk_today += position.risk

    def record_exit(self, position: Position) -> None:
        """Remove a closed position. Never gated by limits."""
        self.positions.remove(position)

    def due_exits(self, marks: dict[str, float], now: datetime, rules: ExitRules) -> list[tuple[Position, str]]:
        """Positions that should be closed now, with the reason for each."""
        out = []
        for p in self.positions:
            mark = marks.get(p.symbol)
            if mark is None:
                continue
            reason = p.exit_reason(mark, now, rules)
            if reason:
                out.append((p, reason))
        return out
