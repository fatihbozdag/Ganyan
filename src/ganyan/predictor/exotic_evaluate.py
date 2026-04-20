"""Back-test exotic-pool strategies against actual TJK payouts.

Given a predictor that produces win probabilities per horse, the
Harville module derives joint probabilities for each exotic pool.
This module replays resulted races and computes:

- **Hit rate** — fraction of races where our top-N combinations
  included the actual winning combo.
- **ROI** — flat 100 TL per ticket × N tickets per race; payout
  materialises only when we hit; minus total stake.
- Per-pool breakdown.

Use via :func:`evaluate_pool` or the ``ganyan exotics-backtest`` CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type
from typing import Callable, Iterable

from sqlalchemy.orm import Session

from ganyan.db.models import Race, RaceEntry, RaceStatus
from ganyan.predictor.bayesian import BayesianPredictor
from ganyan.predictor.exotics import (
    Combo, dortlu_probabilities, ganyan_probabilities,
    ikili_probabilities, sirali_ikili_probabilities, uclu_probabilities,
)


# Which race-level payout column corresponds to which pool.
_PAYOUT_COLUMN = {
    "ganyan": "ganyan_payout_tl",
    "ikili": "ikili_payout_tl",
    "sirali_ikili": "sirali_ikili_payout_tl",
    "uclu": "uclu_payout_tl",
    "dortlu": "dortlu_payout_tl",
}

_COMBO_FUNCS = {
    "ganyan": ganyan_probabilities,
    "ikili": ikili_probabilities,
    "sirali_ikili": sirali_ikili_probabilities,
    "uclu": uclu_probabilities,
    "dortlu": dortlu_probabilities,
}

_COMBO_SIZE = {
    "ganyan": 1,
    "ikili": 2,
    "sirali_ikili": 2,
    "uclu": 3,
    "dortlu": 4,
}


@dataclass
class PoolResult:
    """Aggregate result for a single pool over the evaluated date range."""

    pool: str
    top_n: int
    races: int = 0
    hits: int = 0
    total_stake_tl: float = 0.0
    total_payout_tl: float = 0.0
    misses_without_payout: int = 0  # races where our combo won but payout is NULL
    per_race: list[dict] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return (self.hits / self.races) * 100.0 if self.races else 0.0

    @property
    def roi(self) -> float:
        if self.total_stake_tl <= 0:
            return 0.0
        return (self.total_payout_tl - self.total_stake_tl) / self.total_stake_tl

    def summary_row(self) -> dict:
        return {
            "pool": self.pool,
            "top_n": self.top_n,
            "races": self.races,
            "hits": self.hits,
            "hit_rate_pct": round(self.hit_rate, 2),
            "stake_tl": round(self.total_stake_tl, 2),
            "payout_tl": round(self.total_payout_tl, 2),
            "roi_pct": round(self.roi * 100.0, 2),
            "skipped_missing_payout": self.misses_without_payout,
        }


def _actual_winning_combo(
    entries: list[RaceEntry], pool: str,
) -> tuple[int, ...] | None:
    """Horses (by horse_id) that make up the winning combination."""
    size = _COMBO_SIZE[pool]
    top = [e for e in entries if e.finish_position is not None]
    top = sorted(top, key=lambda e: e.finish_position)[:size]
    if len(top) < size:
        return None
    return tuple(e.horse_id for e in top)


def _combo_matches(
    our: Combo, actual: tuple[int, ...], pool: str,
) -> bool:
    """Does ``our`` exotic combination cover the actual winning combo?"""
    if pool in ("sirali_ikili", "uclu", "dortlu"):
        # Ordered — must match position-by-position.
        return our.horses == actual
    if pool == "ikili":
        # Unordered — set equality.
        return set(our.horses) == set(actual)
    # Ganyan — single horse.
    return our.horses[0] == actual[0]


def evaluate_pool(
    session: Session,
    pool: str,
    top_n: int,
    *,
    from_date: date_type | None = None,
    to_date: date_type | None = None,
    predictor_factory: Callable[[Session], object] | None = None,
    ticket_stake_tl: float = 100.0,
    detail: bool = False,
) -> PoolResult:
    """Replay resulted races for ``pool`` and score our top-N strategy.

    Parameters
    ----------
    pool:
        One of ``ganyan``, ``ikili``, ``sirali_ikili``, ``uclu``, ``dortlu``.
    top_n:
        Number of combinations we'd bet per race (a breadth-coverage
        strategy — more tickets = higher hit rate at higher total stake).
    predictor_factory:
        Callable that returns a predictor instance bound to a session.
        Defaults to :class:`BayesianPredictor`, which is AGF-aware and
        therefore the right baseline for exotic-pool probability work.
    ticket_stake_tl:
        Flat stake per ticket.  Race-level stake is ``top_n *
        ticket_stake_tl``; only races where the pool published a payout
        participate in the stake total.
    """
    if pool not in _COMBO_FUNCS:
        raise ValueError(f"unknown pool: {pool!r}")

    predictor_factory = predictor_factory or (lambda s: BayesianPredictor(s))
    predictor = predictor_factory(session)

    payout_col = _PAYOUT_COLUMN[pool]
    combo_func = _COMBO_FUNCS[pool]

    q = (
        session.query(Race)
        .filter(Race.status == RaceStatus.resulted)
    )
    if from_date is not None:
        q = q.filter(Race.date >= from_date)
    if to_date is not None:
        q = q.filter(Race.date <= to_date)

    result = PoolResult(pool=pool, top_n=top_n)
    for race in q.order_by(Race.date.asc(), Race.race_number.asc()).all():
        entries = list(race.entries)
        if len(entries) < _COMBO_SIZE[pool]:
            continue
        actual = _actual_winning_combo(entries, pool)
        if actual is None:
            continue

        preds = predictor.predict(race.id)
        if not preds:
            continue
        win_probs = {p.horse_id: p.probability / 100.0 for p in preds}
        combos = combo_func(win_probs)[:top_n]
        if not combos:
            continue

        hit = any(_combo_matches(c, actual, pool) for c in combos)
        stake = ticket_stake_tl * top_n
        payout = getattr(race, payout_col)

        if hit and payout is None:
            # We "won" but TJK didn't publish a payout for this pool —
            # means our data is incomplete, so skip rather than inflate.
            result.misses_without_payout += 1
            continue

        result.races += 1
        result.total_stake_tl += stake
        if hit:
            result.hits += 1
            # Payout column is TL per 1 TL ticket on the winning combo.
            # We bought one winning ticket out of our top_n.
            result.total_payout_tl += float(payout) * ticket_stake_tl

        if detail:
            result.per_race.append({
                "race_id": race.id,
                "date": race.date.isoformat(),
                "hit": hit,
                "actual": list(actual),
                "our_top": [list(c.horses) for c in combos],
                "payout_tl": float(payout) if payout is not None else None,
            })

    return result


def evaluate_all_pools(
    session: Session,
    pools: Iterable[str] = ("ganyan", "ikili", "sirali_ikili", "uclu"),
    top_ns: Iterable[int] = (1, 3, 6, 10),
    **kwargs,
) -> list[PoolResult]:
    """Cartesian product of pools × top-N values.  Results are a flat list."""
    out: list[PoolResult] = []
    for pool in pools:
        for top_n in top_ns:
            out.append(evaluate_pool(session, pool, top_n, **kwargs))
    return out
