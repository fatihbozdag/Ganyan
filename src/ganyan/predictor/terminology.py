"""TJK-aligned display names for bet strategies and bet-slip terms.

Keep internal strategy identifiers (uclu_top1 etc.) stable for DB, but map to
official TJK e-bayi terminology for CLI and web output so the user isn't
translating in their head.

Learned 2026-04-23:
  - "K" toggle on the TJK bet slip is Komple, NOT Kutu.
  - Kutu is auto-applied when same 3 horses selected in all 3 columns.
  - Sıralı Üçlü tek kombinasyon has 20 TL minimum ticket.
  - "Bahis sayısı" means different things pre-bet vs post-bet screens.
"""

from __future__ import annotations

STRATEGY_DISPLAY_TR: dict[str, str] = {
    "uclu_top1": "Sıralı Üçlü Bahis (tek kombinasyon)",
    "uclu_box6": "Sıralı Üçlü Bahis (Kutu 6)",
    "sirali_ikili_top1": "Sıralı İkili Bahis (tek kombinasyon)",
    "ganyan_top1": "Ganyan (referans)",
    "plase_top1": "Plase (banko, top-2)",
}

STRATEGY_DISPLAY_SHORT_TR: dict[str, str] = {
    "uclu_top1": "Üçlü Tek",
    "uclu_box6": "Üçlü Kutu 6",
    "sirali_ikili_top1": "İkili Sıralı Tek",
    "ganyan_top1": "Ganyan",
    "plase_top1": "Plase",
}




MINIMUM_STAKES_TL: dict[str, float] = {
    "uclu_top1": 20.0,
    "uclu_box6": 12.0,
    "sirali_ikili_top1": 2.0,
    "ganyan_top1": 1.0,
    "plase": 1.0,
    "plase_top1": 1.0,
}


def strategy_display(strategy: str, short: bool = False) -> str:
    """Return TJK-aligned Turkish display name for a strategy."""
    table = STRATEGY_DISPLAY_SHORT_TR if short else STRATEGY_DISPLAY_TR
    return table.get(strategy, strategy)


def min_stake_tl(strategy: str) -> float:
    """Return the TJK minimum ticket value for a strategy, in TL."""
    return MINIMUM_STAKES_TL.get(strategy, 0.0)
