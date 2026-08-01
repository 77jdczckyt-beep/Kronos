"""Target-allocation investing: decide what to buy, and nothing cleverer.

This module makes no attempt to predict anything. It holds a set of target
weights and moves the portfolio toward them, which is the part of investing
that has durable evidence behind it. Every decision is arithmetic on current
weights, so the same inputs always produce the same orders and any order can be
explained by pointing at two numbers.

Design choices worth stating, because each rules out a way to lose money:

- **Contributions go to the most underweight holding.** At small balances this
  is strictly better than splitting deposits proportionally: it corrects drift
  using new money instead of selling, and selling to rebalance costs spread and
  realises tax.
- **Selling happens only on a hard drift breach**, never on price movement.
  There is no path here from "the price fell" to "sell", which is the single
  most reliable way retail accounts destroy returns.
- **Minimum order size.** A $2 order pays more in spread than it buys in
  accuracy. Below the floor, the bot waits.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class AllocationError(ValueError):
    pass


@dataclass(frozen=True)
class AllocationPolicy:
    """Target weights and the rules for moving toward them.

    Attributes:
        targets: Symbol -> target weight. Must sum to 1.0.
        rebalance_band: Absolute weight drift that triggers a corrective sell.
            0.05 means a holding must be 5 percentage points over target before
            anything is sold. Wide bands are deliberate: frequent rebalancing
            costs more than the drift it corrects.
        min_order: Smallest dollar order to place. Below this the bot waits for
            the next contribution rather than paying spread on dust.
        cash_buffer: Dollars deliberately left uninvested.
    """

    targets: dict[str, float]
    rebalance_band: float = 0.05
    min_order: float = 5.0
    cash_buffer: float = 0.0

    def __post_init__(self) -> None:
        if not self.targets:
            raise AllocationError("targets cannot be empty")
        total = sum(self.targets.values())
        if abs(total - 1.0) > 1e-6:
            raise AllocationError(f"target weights must sum to 1.0, got {total:.4f}")
        for sym, w in self.targets.items():
            if w <= 0:
                raise AllocationError(f"target for {sym} must be positive, got {w}")
        if not 0 < self.rebalance_band < 1:
            raise AllocationError("rebalance_band must be between 0 and 1")
        if self.min_order <= 0:
            raise AllocationError("min_order must be positive")
        if self.cash_buffer < 0:
            raise AllocationError("cash_buffer cannot be negative")


@dataclass(frozen=True)
class Holding:
    symbol: str
    quantity: float
    price: float

    @property
    def value(self) -> float:
        return self.quantity * self.price


@dataclass
class PortfolioState:
    """Current positions and cash, priced at a point in time.

    `unmanaged` names holdings that exist in the account but are not part of the
    policy -- legacy positions the user opened themselves. They are counted in
    total value for reporting but never traded. Automatically liquidating
    something the user chose to buy is not this bot's decision to make.
    """

    holdings: dict[str, Holding] = field(default_factory=dict)
    cash: float = 0.0
    unmanaged: set[str] = field(default_factory=set)

    @property
    def managed_value(self) -> float:
        return sum(h.value for s, h in self.holdings.items() if s not in self.unmanaged)

    @property
    def unmanaged_value(self) -> float:
        return sum(h.value for s, h in self.holdings.items() if s in self.unmanaged)

    @property
    def total_value(self) -> float:
        return self.managed_value + self.unmanaged_value + self.cash

    def weight(self, symbol: str, investable: float) -> float:
        if investable <= 0:
            return 0.0
        h = self.holdings.get(symbol)
        return (h.value if h else 0.0) / investable


@dataclass(frozen=True)
class Trade:
    """A proposed order, with the arithmetic that produced it."""

    symbol: str
    side: str  # "buy" or "sell"
    amount: float  # dollars
    reason: str

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise AllocationError(f"side must be buy or sell, got {self.side!r}")
        if self.amount <= 0:
            raise AllocationError("amount must be positive")


@dataclass
class Plan:
    trades: list[Trade] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total_buys(self) -> float:
        return sum(t.amount for t in self.trades if t.side == "buy")

    def summary(self) -> str:
        lines = []
        for t in self.trades:
            lines.append(f"  {t.side.upper():<4} ${t.amount:>8,.2f}  {t.symbol:<6} -- {t.reason}")
        if not self.trades:
            lines.append("  no trades")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def plan_investment(
    state: PortfolioState,
    policy: AllocationPolicy,
    prices: dict[str, float],
) -> Plan:
    """Produce the orders that move the portfolio toward its targets.

    Deploys available cash into the most underweight holdings first, then
    checks whether any holding has drifted far enough over target to warrant
    selling. Returns an empty plan when nothing is worth doing, which is the
    normal state most of the time.
    """
    plan = Plan()

    missing = [s for s in policy.targets if s not in prices or prices[s] <= 0]
    if missing:
        plan.notes.append(f"no usable price for {', '.join(missing)} -- skipped entirely")
        return plan

    investable = state.managed_value + max(0.0, state.cash - policy.cash_buffer)
    if investable <= 0:
        plan.notes.append("nothing investable after cash buffer")
        return plan

    deployable = max(0.0, state.cash - policy.cash_buffer)

    # --- deploy cash into the largest shortfalls -----------------------------
    if deployable >= policy.min_order:
        shortfalls = []
        for sym, target_w in policy.targets.items():
            current = state.holdings.get(sym).value if sym in state.holdings else 0.0
            desired = target_w * investable
            gap = desired - current
            if gap > 0:
                shortfalls.append((gap, sym, current, desired))
        shortfalls.sort(reverse=True)

        remaining = deployable
        for gap, sym, current, desired in shortfalls:
            if remaining < policy.min_order:
                break
            amount = min(gap, remaining)
            if amount < policy.min_order:
                continue
            cur_w = current / investable
            plan.trades.append(
                Trade(
                    symbol=sym,
                    side="buy",
                    amount=round(amount, 2),
                    reason=(
                        f"{cur_w:.1%} vs {policy.targets[sym]:.0%} target, "
                        f"${gap:,.2f} below"
                    ),
                )
            )
            remaining -= amount

        if remaining >= policy.min_order:
            plan.notes.append(f"${remaining:,.2f} left uninvested (all targets met)")
        elif deployable - remaining > 0:
            pass
    elif deployable > 0:
        plan.notes.append(
            f"${deployable:,.2f} deployable is below the ${policy.min_order:,.2f} "
            "minimum order -- waiting for the next contribution"
        )

    # --- corrective sells, only on a hard band breach ------------------------
    for sym, target_w in policy.targets.items():
        if sym not in state.holdings:
            continue
        cur_w = state.holdings[sym].value / investable
        drift = cur_w - target_w
        if drift > policy.rebalance_band:
            excess = (cur_w - target_w) * investable
            if excess >= policy.min_order:
                plan.trades.append(
                    Trade(
                        symbol=sym,
                        side="sell",
                        amount=round(excess, 2),
                        reason=(
                            f"{cur_w:.1%} vs {target_w:.0%} target, "
                            f"{drift:.1%} over the {policy.rebalance_band:.0%} band"
                        ),
                    )
                )

    if state.unmanaged:
        plan.notes.append(
            f"holding {', '.join(sorted(state.unmanaged))} outside the policy "
            f"(${state.unmanaged_value:,.2f}) -- never traded by the bot"
        )
    return plan
