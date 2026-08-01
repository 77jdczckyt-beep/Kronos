"""Broker abstraction and the order-placement gate.

The concrete Robinhood MCP adapter is deliberately absent. Writing it against
guessed tool names is how you end up with a scheduled job calling functions
that do not exist -- the failure this repository already has one example of.
`MCPBroker` is a documented stub describing exactly what a real adapter must
implement; it is filled in once the server's tool schemas can be read, not
before.

Live order placement is gated behind `LiveBroker.armed`, which no strategy code
can set. Arming is an explicit, human act.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Sequence

from .risk import Position


class BrokerError(RuntimeError):
    pass


class NotArmed(BrokerError):
    """Raised when a live order is attempted without explicit arming."""


@dataclass(frozen=True)
class OrderRequest:
    """A proposed single-leg long option order.

    Only buy-to-open is representable. Selling to open has unbounded or
    near-unbounded loss and is out of scope for this package by construction --
    there is no field here that could express it.
    """

    symbol: str
    strike: float
    right: str
    expiry: str
    contracts: int
    limit_price: float  # per share
    ref_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if self.right not in ("call", "put"):
            raise ValueError(f"right must be 'call' or 'put', got {self.right!r}")
        if self.contracts < 1:
            raise ValueError("contracts must be >= 1")
        if self.limit_price <= 0:
            raise ValueError("limit_price must be positive")

    @property
    def max_loss(self) -> float:
        """Dollars at risk. For a long option this is the whole premium."""
        return self.limit_price * 100.0 * self.contracts


@dataclass(frozen=True)
class OrderResult:
    accepted: bool
    detail: str
    order_id: str | None = None


class Broker(Protocol):
    def buying_power(self) -> float: ...
    def positions(self) -> Sequence[Position]: ...
    def review(self, order: OrderRequest) -> OrderResult: ...
    def place(self, order: OrderRequest) -> OrderResult: ...


@dataclass
class DryRunBroker:
    """Records orders without sending them anywhere.

    The default everywhere. A full scan can run against this and produce
    exactly the orders it would have sent, which is the artifact worth
    reviewing before anything touches a real account.
    """

    starting_buying_power: float = 0.0
    submitted: list[OrderRequest] = field(default_factory=list)
    _positions: list[Position] = field(default_factory=list)

    def buying_power(self) -> float:
        return self.starting_buying_power

    def positions(self) -> Sequence[Position]:
        return list(self._positions)

    def review(self, order: OrderRequest) -> OrderResult:
        if order.max_loss > self.starting_buying_power:
            return OrderResult(
                False,
                f"insufficient buying power: need ${order.max_loss:.2f}, "
                f"have ${self.starting_buying_power:.2f}",
            )
        return OrderResult(True, f"dry run: would buy {order.contracts}x "
                                 f"{order.symbol} {order.strike} {order.right} "
                                 f"@ ${order.limit_price:.2f} (${order.max_loss:.2f} at risk)")

    def place(self, order: OrderRequest) -> OrderResult:
        check = self.review(order)
        if not check.accepted:
            return check
        self.submitted.append(order)
        return OrderResult(True, f"dry run: recorded {order.ref_id}", order.ref_id)


class MCPBroker:
    """Adapter for a brokerage MCP server. **Not implemented.**

    Left as a stub on purpose. Implementing it requires reading the server's
    actual tool schemas -- names, argument shapes, return payloads. Guessing
    them produces code that looks complete, passes review, and fails at the
    only moment that matters.

    A real implementation must, at minimum:

    - Confirm the account is agent-accessible and has the required option level
      before anything else, and refuse to proceed otherwise.
    - Read live buying power, and stop if it is below the risk budget rather
      than working around it.
    - Call the server's review/preview endpoint before every placement and
      compare the returned cost against `OrderRequest.max_loss`, refusing on
      any mismatch. The broker's own arithmetic is the authority, not ours.
    - Send a fresh `ref_id` per order so a retry cannot double-fill.
    - Surface partial fills honestly; a partial fill changes the risk book.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "MCPBroker is a stub. Bind it to the brokerage MCP server's real "
            "tool schemas before use -- do not guess tool names or argument "
            "shapes. Until then, run with DryRunBroker."
        )


@dataclass
class LiveBroker:
    """Wraps a broker and refuses to place unless explicitly armed.

    `armed` is not a strategy parameter and must not be set from signal code.
    It exists so that turning on live trading is a separate, deliberate act
    from deciding a trade looks good.
    """

    inner: Broker
    armed: bool = False
    max_orders: int = 0
    _placed: int = 0

    def buying_power(self) -> float:
        return self.inner.buying_power()

    def positions(self) -> Sequence[Position]:
        return self.inner.positions()

    def review(self, order: OrderRequest) -> OrderResult:
        return self.inner.review(order)

    def place(self, order: OrderRequest) -> OrderResult:
        if not self.armed:
            raise NotArmed(
                "live placement requires LiveBroker.armed=True, set deliberately "
                "and outside strategy code"
            )
        if self._placed >= self.max_orders:
            raise NotArmed(
                f"session order budget exhausted ({self._placed}/{self.max_orders}). "
                "Raise max_orders deliberately to continue."
            )
        result = self.inner.place(order)
        if result.accepted:
            self._placed += 1
        return result
