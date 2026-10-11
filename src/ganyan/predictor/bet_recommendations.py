"""Per-race bet recommendations across every TJK structure.

Project goal (2026-05-03 master directive): pick winners across all bet
types — Tek, Plase, İkili (Sıralı + Tek), Üçlü (Sıralı + Tek + Virgüllü
+ K/Komple), 4'lü, 5'lı, 6'lı, 7'li.

This module turns a race's win-probability distribution into concrete
ticket suggestions for each structure: which horses to play, how many
tickets, what the model thinks the combination probability is, and the
TJK birim (per-bilet stake) for the combo so the user can see the
actual TL outlay.

The picks ledger (``predictor/picks.py``) keeps writing the original 4
strategies for ROI continuity; that's a separate concern from what we
*surface* to the user as betting suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass

from ganyan.predictor.exotics import (
    ganyan_probabilities,
    ikili_probabilities,
    plase_probabilities,
    sirali_ikili_probabilities,
    uclu_probabilities,
    dortlu_probabilities,
)


# Per-pool minimum bilet stake ("birim fiyat") in TL.  Verified from
# real bilets where possible — see picks.py:73-78 for provenance.
BIRIM_TL = {
    "ganyan": 1.0,
    "plase": 1.0,
    "sirali_ikili": 1.0,
    "ikili": 1.0,
    "uclu_sirali": 2.0,
    "uclu": 1.0,            # Üçlü Tek (any order) birim
    "uclu_virgullu": 2.0,   # Sıralı Üçlü Virgüllü uses sıralı birim
    "dortlu": 2.0,
}


@dataclass
class BetPick:
    """One recommended bet for one TJK structure on one race."""

    bet_type: str          # internal key
    label: str             # Turkish label shown to user
    horses: list[int]      # gate numbers in display order
    horse_names: list[str]
    separator: str         # "→" for sıralı, "+" for unordered, "/" for spread
    tickets: int           # number of bilets
    birim_tl: float        # per-bilet stake
    stake_tl: float        # total TL outlay (tickets × birim)
    model_prob_pct: float  # model's combination probability
    note: str = ""         # optional one-line context
    spread_groups: list[list[str]] | None = None  # for virgüllü display

    def to_dict(self) -> dict:
        return {
            "bet_type": self.bet_type,
            "label": self.label,
            "horses": self.horse_names,
            "separator": self.separator,
            "tickets": self.tickets,
            "birim_tl": self.birim_tl,
            "stake_tl": round(self.stake_tl, 2),
            "model_prob_pct": round(self.model_prob_pct, 2),
            "note": self.note,
            "spread_groups": self.spread_groups,
        }


def _name(name_for: dict[int, str], h: int) -> str:
    return name_for.get(h, f"#{h}")


def _gate(gate_for: dict[int, int], h: int) -> int:
    return gate_for.get(h, h)


def compute_bet_recommendations(
    win_probs: dict[int, float],
    horse_names: dict[int, str],
    gates: dict[int, int] | None = None,
) -> list[BetPick]:
    """Build a recommendation per bet type for one race.

    Args:
        win_probs: ``{horse_id: probability_0_to_1}`` — already
            normalised or not, the function will normalise.
        horse_names: ``{horse_id: display_name}``
        gates: ``{horse_id: gate_number}`` — when provided, ``BetPick.horses``
            is populated with gate numbers (program NO) instead of horse
            IDs.  Recommended for user-facing output.

    Returns recommendations in display order: Tek, Plase, İkili Sıralı,
    İkili, Üçlü Sıralı, Üçlü, Üçlü Virgüllü, Üçlü Komple-4, 4'lü.
    """
    if not win_probs:
        return []

    total = sum(max(p, 0.0) for p in win_probs.values())
    if total <= 0:
        return []
    mp = {h: max(p, 0.0) / total for h, p in win_probs.items()}

    g = gates or {h: h for h in mp}
    n = len(mp)
    out: list[BetPick] = []

    # ---- Tek (Ganyan) — banko on the model's #1 ----
    # 2026-05-04: when prob ≥ 30%, flag as primary ticket per CLAUDE.md
    # invariant 5 — Tek+Plase is the default at high conviction.
    gan = ganyan_probabilities(mp)
    if gan:
        top = gan[0]
        h = top.horses[0]
        prob_pct = top.probability * 100.0
        note = "Modelin en güvendiği at"
        if prob_pct >= 30.0:
            note = f"ÖNCELİKLİ — model %{prob_pct:.0f} güvende, Tek+Plase ana taktik"
        out.append(BetPick(
            bet_type="ganyan",
            label="Tek (Ganyan) — banko",
            horses=[_gate(g, h)],
            horse_names=[_name(horse_names, h)],
            separator="",
            tickets=1,
            birim_tl=BIRIM_TL["ganyan"],
            stake_tl=BIRIM_TL["ganyan"],
            model_prob_pct=prob_pct,
            note=note,
        ))

    # ---- Plase — top-2 finish coverage ----
    if n >= 2:
        plase = plase_probabilities(mp, top_k=2)
        if plase:
            top = plase[0]
            h = top.horses[0]
            out.append(BetPick(
                bet_type="plase",
                label="Plase — banko",
                horses=[_gate(g, h)],
                horse_names=[_name(horse_names, h)],
                separator="",
                tickets=1,
                birim_tl=BIRIM_TL["plase"],
                stake_tl=BIRIM_TL["plase"],
                model_prob_pct=top.probability * 100.0,
                note="İlk 2'de bitirme olasılığı en yüksek at",
            ))

    # ---- Sıralı İkili — Tek (top-2 ordered) ----
    if n >= 2:
        si = sirali_ikili_probabilities(mp)
        if si:
            top = si[0]
            out.append(BetPick(
                bet_type="sirali_ikili",
                label="Sıralı İkili — Tek",
                horses=[_gate(g, h) for h in top.horses],
                horse_names=[_name(horse_names, h) for h in top.horses],
                separator="→",
                tickets=1,
                birim_tl=BIRIM_TL["sirali_ikili"],
                stake_tl=BIRIM_TL["sirali_ikili"],
                model_prob_pct=top.probability * 100.0,
                note="Modelin tahmin ettiği 1.→2. sırası",
            ))

    # ---- İkili (unordered) — top-2 ----
    if n >= 2:
        ik = ikili_probabilities(mp)
        if ik:
            top = ik[0]
            out.append(BetPick(
                bet_type="ikili",
                label="İkili — Tek (sıralamasız)",
                horses=[_gate(g, h) for h in top.horses],
                horse_names=[_name(horse_names, h) for h in top.horses],
                separator="/",
                tickets=1,
                birim_tl=BIRIM_TL["ikili"],
                stake_tl=BIRIM_TL["ikili"],
                model_prob_pct=top.probability * 100.0,
                note="Hangi sırada gelirse gelsin ilk iki",
            ))

    # ---- Üçlü Sıralı — Tek (top-3 ordered) ----
    if n >= 3:
        us = uclu_probabilities(mp)
        if us:
            top = us[0]
            out.append(BetPick(
                bet_type="uclu_sirali",
                label="Üçlü Sıralı — Tek",
                horses=[_gate(g, h) for h in top.horses],
                horse_names=[_name(horse_names, h) for h in top.horses],
                separator="→",
                tickets=1,
                birim_tl=BIRIM_TL["uclu_sirali"],
                stake_tl=BIRIM_TL["uclu_sirali"],
                model_prob_pct=top.probability * 100.0,
                note="Modelin 1.→2.→3. tahmini",
            ))

    # ---- Üçlü Komple-3 (any order of model's top 3) ----
    # 2026-05-04: flag fragility when top-3 cluster <70% OR #4 ≥10%.
    # On 2026-05-04, K-3 set match was 15% across 13 races even with
    # 77% top-3 hit rate — the structure is brittle when #4 has real
    # probability mass.
    if n >= 3:
        us = uclu_probabilities(mp)
        if us:
            top = us[0]
            top3_set = set(top.horses)
            any_order = sum(c.probability for c in us if set(c.horses) == top3_set)
            ordered_top3 = sorted(top3_set, key=lambda h: -mp[h])
            top3_cluster = sum(mp[h] for h in top3_set)
            ranked = sorted(mp.items(), key=lambda kv: -kv[1])
            fourth_prob = ranked[3][1] if len(ranked) >= 4 else 0.0
            note = "Aynı 3 atın 6 olası sıralaması — varyans düşük"
            if top3_cluster < 0.70 or fourth_prob >= 0.10:
                note = (
                    f"DİKKAT — top-3 kümesi %{top3_cluster*100:.0f}, "
                    f"#4 at %{fourth_prob*100:.0f}: K-3 kırılgan, Tek tercih edilebilir"
                )
            out.append(BetPick(
                bet_type="uclu_komple3",
                label="Üçlü — Komple 3 (her sıralama)",
                horses=[_gate(g, h) for h in ordered_top3],
                horse_names=[_name(horse_names, h) for h in ordered_top3],
                separator="K",
                tickets=6,
                birim_tl=BIRIM_TL["uclu_sirali"],
                stake_tl=6 * BIRIM_TL["uclu_sirali"],
                model_prob_pct=any_order * 100.0,
                note=note,
            ))

    # ---- Üçlü Virgüllü — banko + spread (1 banko, top-2 for 2nd, top-2 for 3rd)
    if n >= 4:
        gan = ganyan_probabilities(mp)
        rest = [c.horses[0] for c in gan[1:]]  # rest sorted by win prob desc
        if rest and len(rest) >= 3:
            banko = gan[0].horses[0]
            second_pool = rest[:2]   # top 2 among non-banko
            third_pool = rest[:3]    # top 3 among non-banko (overlap with second is fine)

            # Compute combination probability: P(banko=1st AND any second_pool=2nd AND any third_pool=3rd)
            virg_prob = 0.0
            for s in second_pool:
                if s == banko:
                    continue
                for t in third_pool:
                    if t == banko or t == s:
                        continue
                    virg_prob += _ordered_perm_probability((banko, s, t), mp)

            # Tickets: 2 (second_pool) × 2 (third_pool excluding overlap) at most.
            # Computed precisely below.
            tickets = 0
            for s in second_pool:
                if s == banko:
                    continue
                for t in third_pool:
                    if t == banko or t == s:
                        continue
                    tickets += 1
            if tickets == 0:
                tickets = 1

            spread_groups = [
                [_name(horse_names, banko)],
                [_name(horse_names, h) for h in second_pool if h != banko],
                [_name(horse_names, h) for h in third_pool if h != banko],
            ]

            out.append(BetPick(
                bet_type="uclu_virgullu",
                label="Üçlü Sıralı — Virgüllü (banko + yelpaze)",
                horses=[_gate(g, banko)] +
                       [_gate(g, h) for h in second_pool if h != banko] +
                       [_gate(g, h) for h in third_pool if h != banko],
                horse_names=[
                    f"{_name(horse_names, banko)}",
                    " | ".join(_name(horse_names, h) for h in second_pool if h != banko),
                    " | ".join(_name(horse_names, h) for h in third_pool if h != banko),
                ],
                separator=",",
                tickets=tickets,
                birim_tl=BIRIM_TL["uclu_virgullu"],
                stake_tl=tickets * BIRIM_TL["uclu_virgullu"],
                model_prob_pct=virg_prob * 100.0,
                note="1. banko, 2. iki at, 3. iki-üç at — orta risk/orta maliyet",
                spread_groups=spread_groups,
            ))

    # ---- 4'lü Sıralı — Tek (top-4 ordered) ----
    if n >= 4:
        d = dortlu_probabilities(mp)
        if d:
            top = d[0]
            out.append(BetPick(
                bet_type="dortlu",
                label="4'lü Sıralı — Tek",
                horses=[_gate(g, h) for h in top.horses],
                horse_names=[_name(horse_names, h) for h in top.horses],
                separator="→",
                tickets=1,
                birim_tl=BIRIM_TL["dortlu"],
                stake_tl=BIRIM_TL["dortlu"],
                model_prob_pct=top.probability * 100.0,
                note="Modelin ilk 4 sıralı tahmini — düşük olasılık, yüksek ödeme",
            ))

    return out


def _ordered_perm_probability(perm: tuple[int, ...], probs: dict[int, float]) -> float:
    """P(perm[0]=1st, perm[1]=2nd, ...) under Plackett-Luce."""
    remaining = dict(probs)
    p = 1.0
    for h in perm:
        s = sum(remaining.values())
        if s <= 0:
            return 0.0
        p *= remaining[h] / s
        remaining.pop(h, None)
    return p
