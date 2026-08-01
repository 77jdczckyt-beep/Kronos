"""The live allocation policy.

Chosen to match a brief of "large, established companies with steady growth and
strong sector positions". That description is, almost exactly, what a
capitalisation-weighted large-cap index fund holds: the biggest companies,
weighted by size, so sector leaders dominate by construction. Buying the index
expresses the brief without anyone having to judge which firms qualify.

Weights are a starting point and are meant to be edited. They are recorded here
rather than embedded in code so that changing the strategy is a visible,
reviewable act rather than a tweak buried in a function.
"""

from __future__ import annotations

from .allocation import AllocationPolicy

# The agent-accessible account. Everything here targets this one only.
ACCOUNT = "460652704"

# Positions that already existed in the account. The bot prices them for
# reporting but never buys or sells them -- unwinding a position the user chose
# is not an automated decision.
UNMANAGED = {"F", "SOFI"}

# No bonds: this is the aggressive setting, and it is the honest lever for it.
# Higher expected return, deeper drawdowns. There is no allocation that raises
# return without raising risk.
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
POLICY = AllocationPolicy(
    targets={
        "VOO": 0.50,   # S&P 500 -- the large, established US companies
        "QQQ": 0.10,   # Nasdaq 100 -- deliberate mega-cap tech tilt
        "SCHD": 0.20,  # dividend / quality tilt
        "VXUS": 0.20,  # total international -- diversification outside the US
    },
    # 5 percentage points of drift before anything is sold. Wide on purpose:
    # rebalancing more often costs more in spread than the drift it corrects.
    rebalance_band=0.05,
    # Orders below this are not worth the spread at this account size.
    min_order=5.00,
    # Left uninvested so a settlement quirk cannot cause a rejected order.
    cash_buffer=2.00,
)
