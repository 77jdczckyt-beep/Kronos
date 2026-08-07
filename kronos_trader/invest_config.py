"""The live allocation policy.

Chosen to match a brief of "large, established companies with steady growth and
strong sector positions". That description is, almost exactly, what a
capitalisation-weighted large-cap index fund holds: the biggest companies,
weighted by size, so sector leaders dominate by construction. Buying the index
expresses the brief without anyone having to judge which firms qualify.

Weights are a starting point and are meant to be edited. They are recorded here
rather than embedded in code so that changing the strategy is a visible,
reviewable act rather than a tweak buried in a function.

Two sleeves:

- **Core**, 80%, the index funds below. Fixed, and only changed by editing this
  file.
- **Satellite**, 20%, split evenly across whatever is on the "Stocks to buy"
  watchlist at the moment a run happens. Adding a ticker there is enough to put
  it in the buying rotation; no code change needed.

The satellite sleeve is a deliberate exception to the "visible, reviewable act"
rule above, and it is worth being clear-eyed about the trade. A ticker added to
that watchlist will be bought with real money by an unattended job, without
anyone reviewing the decision, and every addition silently shrinks every other
satellite position to make room. The 20% cap is what makes that acceptable: the
core 80% cannot be touched from a watchlist, so the worst case for a careless
addition is bounded.
"""

from __future__ import annotations

from .allocation import AllocationPolicy, AllocationError

# The agent-accessible account. Everything here targets this one only.
ACCOUNT = "460652704"

# Positions that already existed in the account. The bot prices them for
# reporting but never buys or sells them -- unwinding a position the user chose
# is not an automated decision.
UNMANAGED = {"F", "SOFI"}

# One bond sleeve, and not a defensive one -- see EDV below. Everything else
# here is the aggressive setting, and it is the honest lever for it. Higher
# expected return, deeper drawdowns. There is no allocation that raises return
# without raising risk.
#
# On QQQ: its holdings already sit inside VOO, so 10% here does not add a new
# asset -- it overweights the mega-cap technology names VOO already owns. That
# is the point of the tilt, and it is a concentrated sector bet rather than
# diversification. Sized small deliberately.
#
# On SCHD: a dividend is not extra return -- the share price drops by roughly
# the payout. In a taxable account like this one, dividends are taxed on
# receipt whether reinvested or not, so this sleeve carries a small permanent
# tax drag. It is held because the owner asked for dividend exposure, sized so
# that drag stays modest.
# On PICK: global metals and mining producers. This is a cyclical sector bet,
# not a core holding. Mining companies are capital-intensive price-takers whose
# earnings swing with commodity cycles, and over long periods the sector has
# lagged broad equities while being considerably more volatile. It is held for
# the commodity and inflation exposure the other sleeves lack, and because the
# owner asked for it. Note VOO and VXUS already contain materials companies, so
# this is an overweight rather than a new asset class.
#
# On EDV: zero-coupon Treasury STRIPS, roughly 24 years of duration -- the most
# rate-sensitive Treasury fund available. It is not ballast, and sizing it as
# though it were is the mistake to avoid. A short or intermediate bond fund
# damps a portfolio; EDV moves with equity-like amplitude off a different
# driver, and it lost about 40% in 2022 as long rates rose. It is held as a
# deliberate bet on long rates falling, and it is the sleeve most likely to sit
# deeply underwater while the others do fine. Funded from VOO, so the trade is
# US large-cap exposure for duration exposure.
# The core sleeve. Sums to CORE_WEIGHT, not to 1.0 -- the satellite sleeve
# supplies the rest.
CORE_TARGETS = {
    "VOO": 0.20,   # S&P 500 -- the large, established US companies
    "QQQ": 0.08,   # Nasdaq 100 -- deliberate mega-cap tech tilt
    "SCHD": 0.18,  # dividend / quality tilt
    "VXUS": 0.18,  # total international -- diversification outside the US
    "PICK": 0.08,  # global metals & mining -- cyclical commodity tilt
    "EDV": 0.08,   # long-duration Treasuries -- rate bet, not ballast
}

CORE_WEIGHT = 0.80
SATELLITE_WEIGHT = 0.20

# The Robinhood watchlist that drives the satellite sleeve. The ID is recorded
# so a run cannot pick up the wrong list by matching on a display name someone
# renamed.
SATELLITE_LIST_ID = "5976a5b5-ca37-4322-b980-c324d5720bbc"
SATELLITE_LIST_NAME = "Stocks to buy"

# 5 percentage points of drift before anything is sold. Wide on purpose:
# rebalancing more often costs more in spread than the drift it corrects.
REBALANCE_BAND = 0.05
# Orders below this are not worth the spread at this account size.
MIN_ORDER = 5.00
# Left uninvested so a settlement quirk cannot cause a rejected order.
CASH_BUFFER = 2.00


def build_policy(
    satellite_symbols: list[str] | tuple[str, ...] = (),
    priced: set[str] | None = None,
) -> tuple[AllocationPolicy, list[str]]:
    """Combine the fixed core sleeve with the watchlist-driven satellite sleeve.

    `satellite_symbols` is the watchlist contents, in list order. `priced`, when
    given, is the set of symbols a live quote was obtained for; satellites
    outside it are dropped. That keeps one delisted or halted ticker on the
    watchlist from failing the whole run -- the remaining satellites simply
    split the sleeve between them.

    Returns the policy and a list of human-readable notes about anything
    excluded, so the caller can report exclusions rather than hide them.

    Symbols already in the core, and symbols in UNMANAGED, are refused a
    satellite weight. The second case matters: dropping F onto the watchlist
    must not turn the owner's own position into something the bot buys.
    """
    notes: list[str] = []
    satellites: list[str] = []
    seen: set[str] = set()

    for raw in satellite_symbols:
        sym = raw.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        if sym in CORE_TARGETS:
            notes.append(f"{sym} is a core holding; ignored as a satellite")
        elif sym in UNMANAGED:
            notes.append(f"{sym} is unmanaged and is never bought; ignored")
        elif priced is not None and sym not in priced:
            notes.append(f"{sym} has no usable price; excluded from this run")
        else:
            satellites.append(sym)

    targets = dict(CORE_TARGETS)
    if satellites:
        each = SATELLITE_WEIGHT / len(satellites)
        for sym in satellites:
            targets[sym] = each
        notes.append(
            f"{len(satellites)} satellites at {each:.2%} each "
            f"({SATELLITE_WEIGHT:.0%} sleeve)"
        )
    else:
        # No usable satellites: scale the core back up to 1.0 rather than
        # inventing a target for cash. A run with an empty watchlist should
        # still invest into the core.
        targets = {s: w / CORE_WEIGHT for s, w in CORE_TARGETS.items()}
        notes.append("no usable satellites; core scaled to 100%")

    total = sum(targets.values())
    if abs(total - 1.0) > 1e-6:  # pragma: no cover - guards a bad edit above
        raise AllocationError(f"built weights sum to {total:.6f}, not 1.0")

    return (
        AllocationPolicy(
            targets=targets,
            rebalance_band=REBALANCE_BAND,
            min_order=MIN_ORDER,
            cash_buffer=CASH_BUFFER,
        ),
        notes,
    )
