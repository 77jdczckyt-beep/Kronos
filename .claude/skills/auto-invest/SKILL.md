---
name: auto-invest
description: Invest new cash in the Robinhood agentic account according to the committed target allocation. Use when a scheduled auto-invest run fires, or when asked to check for uninvested cash and deploy it.
---

# Auto-invest run

Deploys uninvested cash into the target allocation in `kronos_trader/invest_config.py`.
Fully automatic: place the orders, then report. Do not ask for confirmation —
that was granted deliberately, and buying during a downturn is the intended
behaviour, not a reason to stop.

**Never compute order sizes yourself.** Run the script. An agent doing this
arithmetic in prose is how a wrong number becomes a wrong trade.

## 1. Read the account

Account number is `ACCOUNT` in `kronos_trader/invest_config.py`.

- `get_portfolio` → cash and buying power
- `get_equity_positions` → current quantities
- `get_equity_quotes` → prices for every symbol in `POLICY.targets` **and**
  every symbol held

Use the regular-session `last_trade_price`. Outside market hours the bid/ask
spread widens to meaningless values — a $24 spread on VOO was observed at
8pm ET. Never size an order off an after-hours quote.

If any call fails, **stop and report**. Do not proceed on partial data, do not
substitute yesterday's price, and do not retry a policy denial.

## 2. Compute the plan

```bash
echo '{"cash": <cash>, "positions": {"SYM": {"quantity": N, "price": P}, ...},
       "prices": {"SYM": P, ...}}' | python scripts/plan_orders.py
```

## 3. Verify before placing

Refuse the run if any of these fail:

- `checks.buys_within_cash` is `true`
- `checks.no_unmanaged_traded` is `true`
- `checks.all_symbols_in_policy` is `true`
- `checks.sell_count` is `0` — a scheduled run deposits money; it should
  never be selling. A non-zero count means a drift breach, which is worth a
  human look before executing.
- `checks.total_buys` does not exceed the buying power reported by
  `get_portfolio`

If `trades` is empty, that is a normal outcome. Report it and stop.

## 4. Place the orders

For each trade, in the order returned:

1. `review_equity_order` — `type: market`, `dollar_amount`, `market_hours: regular_hours`
2. Check the response. Surface any non-empty `order_checks` and **stop** if it
   indicates a problem. Show `market_data_disclosure` verbatim in the report;
   it is a compliance requirement.
3. `place_equity_order` with the same parameters plus a **fresh UUID `ref_id`**.
   Re-send the *same* `ref_id` only when retrying a transport failure — that is
   what stops a retry becoming a double order. A new order always gets a new one.

Dollar-based and fractional orders execute in regular hours only. Placed
outside them, they queue for the next open — which is fine and expected.

If an order errors, it was **not** placed. Report verbatim, continue with the
remaining trades, and say clearly which ones did and did not go in.

## 5. Report

State: cash found, each order placed with its dollar amount and order ID, the
resulting allocation versus target, and anything skipped with the reason.

Orders come back `queued`, not filled. Say so — do not describe a queued order
as a completed purchase.

## Hard rules

- **Never sell on price movement.** Falling prices produce no sell orders. If a
  drop makes selling look sensible, that instinct is the thing this system
  exists to override.
- **Never trade a symbol in `UNMANAGED`**, or any symbol absent from
  `POLICY.targets`. The owner's own positions are theirs.
- **Never touch options.** This account holds an option position; it is not
  managed here.
- **Never edit `invest_config.py` during a run.** Changing the allocation is a
  human decision made deliberately, not a step in an automated job.
- **Never work around a blocker.** Insufficient buying power, a halted
  instrument, or a failed data call all mean stop and report.
