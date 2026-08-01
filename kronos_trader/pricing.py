"""Black-Scholes pricing and implied-volatility inversion.

Only needed to answer one question: what does the market already think this
move is worth? A forecast that a stock will rise is not a trade signal. The
option is priced off implied volatility, so a long call only makes money if the
move exceeds what implied vol already pays for. This module supplies the
benchmark that the ensemble forecast gets measured against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SQRT_2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


@dataclass(frozen=True)
class OptionQuote:
    """A tradeable option contract with live market quotes.

    Attributes:
        symbol: Underlying ticker.
        strike: Strike price.
        expiry_years: Time to expiry in years.
        right: "call" or "put".
        bid: Best bid, per share (multiply by 100 for the contract).
        ask: Best ask, per share.
        open_interest: Contracts outstanding; a liquidity guard.
    """

    symbol: str
    strike: float
    expiry_years: float
    right: str
    bid: float
    ask: float
    open_interest: int = 0

    def __post_init__(self) -> None:
        if self.right not in ("call", "put"):
            raise ValueError(f"right must be 'call' or 'put', got {self.right!r}")
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.expiry_years <= 0:
            raise ValueError("expiry_years must be positive")
        if self.bid < 0 or self.ask < 0:
            raise ValueError("quotes cannot be negative")
        if self.ask < self.bid:
            raise ValueError(f"crossed quote: ask {self.ask} < bid {self.bid}")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_frac(self) -> float:
        """Bid-ask spread as a fraction of mid. A wide spread is a real cost.

        Returns ``inf`` for a zero mid, which correctly disqualifies the
        contract rather than dividing by zero.
        """
        m = self.mid
        return float("inf") if m <= 0 else (self.ask - self.bid) / m


def bs_price(spot: float, strike: float, t: float, vol: float, right: str, rate: float = 0.0) -> float:
    """Black-Scholes price for a European option, per share.

    Args:
        spot: Current underlying price.
        strike: Strike price.
        t: Time to expiry in years.
        vol: Annualised volatility (as a decimal, e.g. 0.35).
        right: "call" or "put".
        rate: Risk-free rate.
    """
    if t <= 0 or vol <= 0:
        intrinsic = spot - strike if right == "call" else strike - spot
        return max(0.0, intrinsic)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    disc = math.exp(-rate * t)
    if right == "call":
        return spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
    return strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    t: float,
    right: str,
    rate: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Invert Black-Scholes for volatility via bisection.

    Bisection rather than Newton: it cannot diverge, and option vega collapses
    for deep out-of-the-money contracts, which is exactly where Newton's method
    becomes unstable and where these strategies tend to shop.

    Returns ``nan`` if the price is outside the no-arbitrage band, which is a
    signal to skip the contract rather than to trust a fabricated number.
    """
    intrinsic = max(0.0, (spot - strike) if right == "call" else (strike - spot))
    upper_bound = spot if right == "call" else strike
    if price < intrinsic - tol or price > upper_bound + tol:
        return float("nan")

    lo, hi = 1e-6, 10.0
    if bs_price(spot, strike, t, hi, right, rate) < price:
        return float("nan")  # even 1000% vol cannot reach this price

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if bs_price(spot, strike, t, mid, right, rate) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def expected_payoff(terminal_prices: np.ndarray, strike: float, right: str) -> float:
    """Mean option payoff at expiry across simulated terminal prices, per share."""
    if right == "call":
        payoff = np.maximum(terminal_prices - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal_prices, 0.0)
    return float(np.mean(payoff))
