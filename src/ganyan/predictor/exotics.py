"""Exotic-pool joint-probability estimates via the Harville model.

The Ganyan (win) probabilities we already produce are a one-dimensional
view of a race.  Turkish parimutuel pools go further:

- Plase (place): horse finishes in the top k (usually 2, sometimes 3)
- İkili: two specific horses finish top-2 in any order
- Sıralı İkili: two horses in exact 1st-2nd order
- Üçlü: three horses in exact 1-2-3 order
- Dörtlü: four horses in exact 1-2-3-4 order

These payouts are much larger than Ganyan because the combination space
is larger and liquidity is thinner, so pool efficiency is lower too —
which is exactly where a reasonable win-probability model can hope to
find edge.

We use the Harville (1973) model::

    P(1st = i, 2nd = j) = p_i * p_j / (1 - p_i)
    P(1st = i, 2nd = j, 3rd = k) = p_i * (p_j/(1-p_i)) * (p_k/(1-p_i-p_j))

It is well-documented and computationally cheap.  More elaborate
alternatives (Henery, Plackett-Luce with position-specific strength)
exist but rarely beat Harville outside large sample studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations


@dataclass(frozen=True)
class Combo:
    """A specific exotic combination and its probability."""

    horses: tuple[int, ...]  # horse_id tuple, positionally
    probability: float  # 0..1
    ordered: bool  # True = 1-2-3 fixed order; False = top-k as a set


def _normalize(probs: dict[int, float]) -> dict[int, float]:
    """Defensive: rescale to sum=1 so the Harville formulae stay well-defined."""
    total = sum(probs.values())
    if total <= 0:
        n = len(probs)
        return {h: 1.0 / n for h in probs} if n else {}
    return {h: p / total for h, p in probs.items()}


def ganyan_probabilities(win_probs: dict[int, float]) -> list[Combo]:
    """Win probabilities as single-horse Combos, sorted descending."""
    probs = _normalize(win_probs)
    combos = [
        Combo(horses=(h,), probability=p, ordered=True)
        for h, p in probs.items()
    ]
    combos.sort(key=lambda c: c.probability, reverse=True)
    return combos


def plase_probabilities(
    win_probs: dict[int, float], top_k: int = 2,
) -> list[Combo]:
    """P(horse i finishes in the top ``top_k``) under Harville.

    Returns a list of single-horse Combos sorted descending by the
    top-k probability (not the win probability).
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    probs = _normalize(win_probs)
    horses = list(probs.keys())

    results: dict[int, float] = {h: 0.0 for h in horses}
    # Sum P(horse i finishes in position 1..top_k) across ordered perms.
    for perm in permutations(horses, top_k):
        p = _perm_probability(perm, probs)
        # Each horse in this ordered prefix finishes top_k.
        for h in perm:
            results[h] += p

    combos = [
        Combo(horses=(h,), probability=prob, ordered=False)
        for h, prob in results.items()
    ]
    combos.sort(key=lambda c: c.probability, reverse=True)
    return combos


def sirali_ikili_probabilities(
    win_probs: dict[int, float],
) -> list[Combo]:
    """All ordered (1st, 2nd) pair probabilities, sorted descending."""
    probs = _normalize(win_probs)
    combos: list[Combo] = []
    for i, j in permutations(probs.keys(), 2):
        p = _perm_probability((i, j), probs)
        combos.append(Combo(horses=(i, j), probability=p, ordered=True))
    combos.sort(key=lambda c: c.probability, reverse=True)
    return combos


def ikili_probabilities(
    win_probs: dict[int, float],
) -> list[Combo]:
    """Unordered top-2 combination probabilities, sorted descending.

    ``P({i,j} = top 2) = P(i=1st, j=2nd) + P(j=1st, i=2nd)``.
    """
    probs = _normalize(win_probs)
    horses = list(probs.keys())
    combos: list[Combo] = []
    for a_idx in range(len(horses)):
        for b_idx in range(a_idx + 1, len(horses)):
            i, j = horses[a_idx], horses[b_idx]
            p = _perm_probability((i, j), probs) + _perm_probability((j, i), probs)
            combos.append(Combo(horses=(i, j), probability=p, ordered=False))
    combos.sort(key=lambda c: c.probability, reverse=True)
    return combos


def uclu_probabilities(
    win_probs: dict[int, float],
) -> list[Combo]:
    """All ordered (1st, 2nd, 3rd) triple probabilities, sorted descending."""
    probs = _normalize(win_probs)
    if len(probs) < 3:
        return []
    combos: list[Combo] = []
    for triple in permutations(probs.keys(), 3):
        p = _perm_probability(triple, probs)
        combos.append(Combo(horses=triple, probability=p, ordered=True))
    combos.sort(key=lambda c: c.probability, reverse=True)
    return combos


def dortlu_probabilities(
    win_probs: dict[int, float],
) -> list[Combo]:
    """All ordered top-4 tuple probabilities, sorted descending.

    With 14-horse fields this is 14*13*12*11 = 24,024 permutations —
    slow enough that callers should usually truncate via :func:`top_n`.
    """
    probs = _normalize(win_probs)
    if len(probs) < 4:
        return []
    combos: list[Combo] = []
    for quad in permutations(probs.keys(), 4):
        p = _perm_probability(quad, probs)
        combos.append(Combo(horses=quad, probability=p, ordered=True))
    combos.sort(key=lambda c: c.probability, reverse=True)
    return combos


def _perm_probability(
    perm: tuple[int, ...], probs: dict[int, float],
) -> float:
    """Harville probability of a specific ordered finishing prefix.

    ``perm`` is ``(i, j, k, ...)`` meaning horse i finishes 1st, j 2nd,
    etc.  Returns ``P(1st=i, 2nd=j, ...)`` under Harville.
    """
    remaining = 1.0
    p = 1.0
    for horse in perm:
        pi = probs.get(horse, 0.0)
        if remaining <= 0.0 or pi <= 0.0:
            return 0.0
        p *= pi / remaining
        remaining -= pi
    return p


# ---------------------------------------------------------------------------
# Coverage helpers — how many combinations to play to hit a target prob
# ---------------------------------------------------------------------------


def cumulative_coverage(combos: list[Combo]) -> list[float]:
    """Cumulative hit probability as you add combinations in order.

    Useful for deciding "how many tickets do I need?" — e.g. if the
    first 6 Üçlü combinations cumulatively cover 45% of outcomes, that's
    your coverage/cost trade-off.
    """
    total = 0.0
    out: list[float] = []
    for c in combos:
        total += c.probability
        out.append(total)
    return out


def top_n(combos: list[Combo], n: int) -> list[Combo]:
    """First ``n`` combos — thin wrapper for readability in callers."""
    return combos[:n]
