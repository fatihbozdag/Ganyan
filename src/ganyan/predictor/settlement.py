"""Pool settlement shared by the ledger and historical evaluations.

Units preserve the repository's recorded payout conventions. They are not odds
estimates. Update a convention in this one place when source evidence changes.
"""
POOL_UNIT_TL = {
    "ganyan": 1.0, "plase": 1.0, "ikili": 1.0,
    "sirali_ikili": 1.0, "uclu": 2.0, "dortlu": 1.0,
}
STRATEGY_POOL = {
    "ganyan_top1": "ganyan", "plase_top1": "plase",
    "sirali_ikili_top1": "sirali_ikili", "uclu_top1": "uclu",
    "uclu_box6": "uclu",
}


def winning_payout(pool: str, published: float, ticket_stake: float) -> float:
    if published < 0 or ticket_stake < 0:
        raise ValueError("Payout and stake must be nonnegative")
    return published * ticket_stake / POOL_UNIT_TL[pool]


def plase_pool_confirmed(entries) -> bool:
    return any(e.plase_payout_tl is not None for e in entries)
