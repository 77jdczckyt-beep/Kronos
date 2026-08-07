"""Compute the investment plan from account state. JSON in, JSON out.

    echo '{"cash": 250.00, "positions": {...}, "prices": {...},
           "watchlist": ["PLTR", "MU", ...]}' \
        | python scripts/plan_orders.py

Exists so a scheduled run does arithmetic in code rather than in prose. An
agent reading balances and working out order sizes in its head is exactly where
a wrong number becomes a wrong trade; this makes the calculation deterministic,
testable, and identical every time.

`watchlist` is the "Stocks to buy" list, and it drives the satellite sleeve --
see `kronos_trader/invest_config.py`. It is optional: omit it and the core
sleeve is scaled to 100%. The caller passes the symbols in rather than the
script fetching them, because the watchlist lives behind an MCP tool the agent
holds and this script deliberately has no network access.

Output is a JSON object with `trades` (each carrying the reasoning that
produced it), `notes`, and a `checks` block the caller must verify before
placing anything.
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

from kronos_trader.allocation import (  # noqa: E402
    Holding,
    PortfolioState,
    plan_investment,
)
from kronos_trader.invest_config import (  # noqa: E402
    SATELLITE_LIST_ID,
    UNMANAGED,
    build_policy,
)


def main() -> None:
    payload = json.load(sys.stdin)

    cash = float(payload["cash"])
    prices = {k: float(v) for k, v in payload["prices"].items()}
    holdings = {}
    for sym, pos in payload.get("positions", {}).items():
        qty = float(pos["quantity"])
        price = float(pos.get("price", prices.get(sym, 0.0)))
        if qty > 0 and price > 0:
            holdings[sym] = Holding(sym, qty, price)

    priced = {s for s, p in prices.items() if p > 0}
    policy, policy_notes = build_policy(payload.get("watchlist", []), priced=priced)

    state = PortfolioState(holdings=holdings, cash=cash, unmanaged=set(UNMANAGED))
    plan = plan_investment(state, policy, prices)

    total_buys = round(plan.total_buys, 2)
    out = {
        "account_value": round(state.total_value, 2),
        "cash": round(state.cash, 2),
        "managed_value": round(state.managed_value, 2),
        "unmanaged_value": round(state.unmanaged_value, 2),
        "satellite_list_id": SATELLITE_LIST_ID,
        "targets": policy.targets,
        "trades": [
            {"symbol": t.symbol, "side": t.side, "amount": t.amount, "reason": t.reason}
            for t in plan.trades
        ],
        "notes": policy_notes + plan.notes,
        "checks": {
            # The caller must confirm each of these before placing. They are
            # cheap to verify and each one catches a different way a run can
            # spend money it should not.
            "total_buys": total_buys,
            "buys_within_cash": total_buys <= round(cash - policy.cash_buffer, 2) + 0.01,
            "no_unmanaged_traded": all(
                t.symbol not in UNMANAGED for t in plan.trades
            ),
            "all_symbols_in_policy": all(
                t.symbol in policy.targets for t in plan.trades
            ),
            "sell_count": sum(1 for t in plan.trades if t.side == "sell"),
            # Satellites the watchlist named but this run could not price. Not
            # fatal -- the rest of the sleeve absorbs the weight -- but the
            # caller should say so rather than let a ticker quietly vanish.
            "satellites_excluded": [
                n for n in policy_notes if "excluded" in n or "ignored" in n
            ],
        },
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
