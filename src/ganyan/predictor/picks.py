"""Strategy-level bet ledger: generate picks and grade them later.

Per race we record three Picks, one per strategy we actually recommend:

  - ``uclu_top1``        100 TL, one ticket,       backtest +150% ROI
  - ``uclu_box6``        600 TL, 6 tickets (box),  backtest +112% ROI
  - ``sirali_ikili_top1`` 100 TL, one ticket,      backtest  -2.7% ROI

``generate_picks_for_race()`` computes them from the current model's
predicted probabilities and inserts a :class:`~ganyan.db.models.Pick`
row each (idempotent — the ``(race_id, strategy)`` unique constraint
prevents duplicates; re-running simply no-ops).

``grade_race()`` fills in ``hit``, ``payout_tl``, ``net_tl`` using the
scraped finish positions and TJK payouts, once the race is resulted.
"""

from __future__ import annotations

import itertools
import logging
from datetime import datetime
from typing import Iterable

from sqlalchemy.orm import Session

from ganyan.db.models import Pick, Race, RaceEntry, RaceStatus
from ganyan.predictor.exotics import (
    Combo, sirali_ikili_probabilities, uclu_probabilities,
)


logger = logging.getLogger(__name__)


STAKE_PER_TICKET_TL = 100.0


STRATEGIES = ("uclu_top1", "uclu_box6", "sirali_ikili_top1")


def generate_picks_for_race(
    session: Session, race_id: int,
) -> list[Pick]:
    """Generate (or refresh) this race's picks from stored predictions.

    Reads ``RaceEntry.predicted_probability`` — written by the
    predictor's ``predict_and_save`` — and derives Harville joint
    probabilities.  Safe to call repeatedly; existing rows are
    untouched, missing ones are inserted.
    """
    race = session.get(Race, race_id)
    if race is None:
        return []
    entries = list(race.entries)
    if not entries:
        return []

    win_probs: dict[int, float] = {}
    name_for: dict[int, str] = {}
    for e in entries:
        if e.predicted_probability is not None:
            win_probs[e.horse_id] = max(float(e.predicted_probability), 0.0) / 100.0
            name_for[e.horse_id] = e.horse.name if e.horse else "?"
    if sum(win_probs.values()) <= 0:
        return []

    existing = {
        p.strategy: p for p in
        session.query(Pick).filter(Pick.race_id == race_id).all()
    }
    added: list[Pick] = []

    # uclu_top1
    uclu = uclu_probabilities(win_probs)
    if uclu and len(win_probs) >= 3 and "uclu_top1" not in existing:
        top = uclu[0]
        added.append(_make_pick(
            race_id=race_id,
            strategy="uclu_top1",
            combination=list(top.horses),
            combination_names=[name_for.get(h, "?") for h in top.horses],
            stake=STAKE_PER_TICKET_TL,
            tickets=1,
            model_prob_pct=top.probability * 100.0,
        ))

    # uclu_box6 — the same top-3 horses, all 6 orderings = 6 tickets
    if uclu and len(win_probs) >= 3 and "uclu_box6" not in existing:
        base = list(uclu[0].horses)
        top3_set = set(base)
        any_order_prob = sum(
            c.probability for c in uclu_probabilities(win_probs)
            if set(c.horses) == top3_set
        )
        added.append(_make_pick(
            race_id=race_id,
            strategy="uclu_box6",
            combination=base,
            combination_names=[name_for.get(h, "?") for h in base],
            stake=STAKE_PER_TICKET_TL * 6,
            tickets=6,
            model_prob_pct=any_order_prob * 100.0,
        ))

    # sirali_ikili_top1
    si = sirali_ikili_probabilities(win_probs)
    if si and len(win_probs) >= 2 and "sirali_ikili_top1" not in existing:
        top = si[0]
        added.append(_make_pick(
            race_id=race_id,
            strategy="sirali_ikili_top1",
            combination=list(top.horses),
            combination_names=[name_for.get(h, "?") for h in top.horses],
            stake=STAKE_PER_TICKET_TL,
            tickets=1,
            model_prob_pct=top.probability * 100.0,
        ))

    for p in added:
        session.add(p)
    session.flush()
    return added


def _make_pick(
    *, race_id: int, strategy: str, combination: list[int],
    combination_names: list[str], stake: float, tickets: int,
    model_prob_pct: float,
) -> Pick:
    return Pick(
        race_id=race_id,
        strategy=strategy,
        combination=combination,
        combination_names=combination_names,
        stake_tl=stake,
        ticket_count=tickets,
        model_prob_pct=round(model_prob_pct, 3),
    )


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _actual_top3(entries: Iterable[RaceEntry]) -> tuple[int, ...] | None:
    ordered = sorted(
        [e for e in entries if e.finish_position in (1, 2, 3)],
        key=lambda e: e.finish_position,
    )
    if len(ordered) < 2:
        return None
    return tuple(e.horse_id for e in ordered[:3])


def _strategy_hit(strategy: str, combination: list[int], actual: tuple[int, ...]) -> bool | None:
    """Did our combination match the actual finish for this strategy?"""
    if strategy == "uclu_top1":
        if len(actual) < 3: return None
        return tuple(combination) == actual[:3]
    if strategy == "uclu_box6":
        if len(actual) < 3: return None
        return set(combination) == set(actual[:3])
    if strategy == "sirali_ikili_top1":
        if len(actual) < 2: return None
        return tuple(combination) == actual[:2]
    return None


def _strategy_payout_tl(strategy: str, race: Race) -> float | None:
    if strategy == "sirali_ikili_top1":
        v = race.sirali_ikili_payout_tl
    else:  # uclu_top1 / uclu_box6 both pay from the üçlü pool
        v = race.uclu_payout_tl
    return float(v) if v is not None else None


def grade_race(session: Session, race_id: int) -> int:
    """Grade every ungraded pick for ``race_id``.  Returns count graded."""
    race = session.get(Race, race_id)
    if race is None or race.status != RaceStatus.resulted:
        return 0
    entries = list(race.entries)
    actual = _actual_top3(entries)
    if actual is None:
        return 0

    picks = (
        session.query(Pick)
        .filter(Pick.race_id == race_id, Pick.graded == False)  # noqa: E712
        .all()
    )
    now = datetime.now()
    graded = 0
    for pick in picks:
        hit = _strategy_hit(pick.strategy, pick.combination, actual)
        if hit is None:
            continue

        payout_per_tl = _strategy_payout_tl(pick.strategy, race)
        # Skip grading when TJK didn't publish a payout for this pool:
        # the bet literally could not have been placed, so it doesn't
        # belong in the strategy's P&L ledger.  Leave graded=False so
        # a later scrape that fills the payout still triggers grading.
        if payout_per_tl is None:
            continue

        pick.hit = hit
        pick.graded = True
        pick.graded_at = now
        if hit:
            # One winning ticket out of ``ticket_count``; rest lost.
            winning_ticket_payout = payout_per_tl * STAKE_PER_TICKET_TL
            pick.payout_tl = round(winning_ticket_payout, 2)
            pick.net_tl = round(winning_ticket_payout - float(pick.stake_tl), 2)
        else:
            pick.payout_tl = 0.0
            pick.net_tl = -float(pick.stake_tl)
        graded += 1

    if graded:
        session.flush()
    return graded


def grade_all_pending(session: Session) -> int:
    """Grade every ungraded pick whose race has resulted.  Returns count."""
    q = (
        session.query(Pick.race_id)
        .join(Race, Race.id == Pick.race_id)
        .filter(
            Pick.graded == False,  # noqa: E712
            Race.status == RaceStatus.resulted,
        )
        .distinct()
    )
    total = 0
    for (race_id,) in q.all():
        total += grade_race(session, race_id)
    return total


def strategy_summary(
    session: Session, *, strategy: str | None = None, since=None,
) -> dict:
    """Aggregate ROI per strategy over graded picks.

    Returns::

        {
          "uclu_top1": {n, hits, stake_tl, payout_tl, net_tl, roi_pct, hit_rate_pct},
          ...
        }
    """
    q = session.query(Pick).filter(Pick.graded == True)  # noqa: E712
    if strategy:
        q = q.filter(Pick.strategy == strategy)
    if since is not None:
        q = q.filter(Pick.generated_at >= since)

    agg: dict[str, dict] = {}
    for p in q.all():
        row = agg.setdefault(p.strategy, {
            "n": 0, "hits": 0, "stake_tl": 0.0, "payout_tl": 0.0, "net_tl": 0.0,
        })
        row["n"] += 1
        row["stake_tl"] += float(p.stake_tl)
        if p.hit:
            row["hits"] += 1
        if p.payout_tl is not None:
            row["payout_tl"] += float(p.payout_tl)
        if p.net_tl is not None:
            row["net_tl"] += float(p.net_tl)

    for row in agg.values():
        row["hit_rate_pct"] = (row["hits"] / row["n"]) * 100 if row["n"] else 0
        row["roi_pct"] = (row["net_tl"] / row["stake_tl"]) * 100 if row["stake_tl"] else 0
    return agg
