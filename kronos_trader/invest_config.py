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

POLICY = AllocationPolicy(
    targets={
        "VOO": 0.70,   # S&P 500 -- the large, established US companies
        "VXUS": 0.20,  # total international -- diversification outside the US
        "BND": 0.10,   # total bond market -- ballast
    },
    # 5 percentage points of drift before anything is sold. Wide on purpose:
    # rebalancing more often costs more in spread than the drift it corrects.
    rebalance_band=0.05,
    # Orders below this are not worth the spread at this account size.
    min_order=5.00,
    # Left uninvested so a settlement quirk cannot cause a rejected order.
    cash_buffer=2.00,
)
