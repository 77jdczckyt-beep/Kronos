"""Distribution-based options signals on top of Kronos forecasts.

Re-exports are resolved lazily (PEP 562). The options and forecasting modules
pull in numpy, pandas and torch; the allocation side is pure arithmetic on
stdlib. Importing them eagerly here meant `import kronos_trader.allocation`
could not run without the whole ML stack installed, which broke the scheduled
auto-invest run -- a job whose entire point is that it does nothing cleverer
than divide two numbers.

`from kronos_trader import X` still works exactly as before; the submodule is
imported on first attribute access instead of at package import.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Attribute name -> submodule that defines it.
_EXPORTS = {
    "Broker": "broker",
    "DryRunBroker": "broker",
    "LiveBroker": "broker",
    "MCPBroker": "broker",
    "NotArmed": "broker",
    "OrderRequest": "broker",
    "OrderResult": "broker",
    "ACCOUNT_CASH": "config",
    "MAX_RISK_PER_TRADE": "config",
    "LiveConfig": "config",
    "CsvMarketData": "data",
    "MarketData": "data",
    "MarketDataError": "data",
    "BootstrapForecaster": "forecast",
    "KronosEnsembleForecaster": "forecast",
    "PathEnsemble": "forecast",
    "OptionQuote": "pricing",
    "bs_price": "pricing",
    "implied_vol": "pricing",
    "ExitRules": "risk",
    "Position": "risk",
    "RiskBook": "risk",
    "RiskLimits": "risk",
    "RiskViolation": "risk",
    "Candidate": "runner",
    "ScanConfig": "runner",
    "ScanReport": "runner",
    "manage_exits": "runner",
    "run": "runner",
    "scan": "runner",
    "Signal": "signals",
    "SignalConfig": "signals",
    "best_signal": "signals",
    "evaluate_contract": "signals",
    "DEFAULT_SEEDS": "universe",
    "ScreenCriteria": "universe",
    "ScreenResult": "universe",
    "screen": "universe",
}

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from .broker import (
        Broker,
        DryRunBroker,
        LiveBroker,
        MCPBroker,
        NotArmed,
        OrderRequest,
        OrderResult,
    )
    from .config import ACCOUNT_CASH, MAX_RISK_PER_TRADE, LiveConfig
    from .data import CsvMarketData, MarketData, MarketDataError
    from .forecast import BootstrapForecaster, KronosEnsembleForecaster, PathEnsemble
    from .pricing import OptionQuote, bs_price, implied_vol
    from .risk import ExitRules, Position, RiskBook, RiskLimits, RiskViolation
    from .runner import Candidate, ScanConfig, ScanReport, manage_exits, run, scan
    from .signals import Signal, SignalConfig, best_signal, evaluate_contract
    from .universe import DEFAULT_SEEDS, ScreenCriteria, ScreenResult, screen


def __getattr__(name: str):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = list(_EXPORTS)
