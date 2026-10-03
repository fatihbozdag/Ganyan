"""6'lı / 5'lı / 7'lı GANYAN coupon generation and grading.

A multi-race coupon's profit comes from **asymmetric leg sizing**, not
horse-picking accuracy: lock 1-2 horses where the model is confident,
spread 4-6 where it's uncertain, and rely on a budget cap to keep
ticket count tractable. Mirrors the human strategy observed in
winning Turkish 6'lı coupons (e.g. baba Orhan İzmir 2026-05-02).

Per-leg conviction tiers are derived from the ensemble's top-1
``mean_probability`` — that's a calibrated within-race share, so
0.50 means "model gives 50% chance to its top pick" not "50/50
flip". Tier widths are deliberately conservative; the budget cap
narrows the widest legs first when the product would blow past it.

Grading is straightforward boolean AND across legs: every leg's
actual winner must be in our kept list. Dead-heat alternatives in
the pool's winning combo (commas inside a leg) count as multiple
acceptable winners.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date as date_type, datetime, timezone
from functools import reduce
from operator import mul
from typing import Sequence

from sqlalchemy.orm import Session

from ganyan.db.models import (
    MultiRacePick, MultiRacePool, Race, Track,
)
from ganyan.predictor.ml.ensemble import EnsemblePredictor
from ganyan.time import is_upcoming, utcnow


logger = logging.getLogger(__name__)


# Default ticket unit. TJK's published 6'lı pool payouts are quoted
# per 1 TL hisse (per "share"); ticket_unit_tl scales the final
# payout linearly. Half-bilet (0.5) and standard 2.50 TL bets are
# common; we pick 1.0 as the unit for clean reporting.
DEFAULT_TICKET_UNIT_TL = 1.0
DEFAULT_MAX_TICKETS = 512


# Conviction tiers — top-1 mean_probability → leg width.
# Calibrated against an 11-horse field where 1/N = 9.1% would be the
# "no information" baseline; >0.18 indicates the model has actually
# picked a top horse out.
_TIERS: tuple[tuple[float, int], ...] = (
    (0.50, 1),  # lock
    (0.35, 2),
    (0.25, 3),
    (0.18, 4),
    (0.00, 5),  # spread (model is barely above uniform)
)


def _leg_width(top1_prob: float) -> int:
    for threshold, width in _TIERS:
        if top1_prob >= threshold:
            return width
    return _TIERS[-1][1]


@dataclass
class CouponDraft:
    """In-memory representation of a generated coupon, before persistence."""
    kept_horses_per_leg: list[list[int]]
    conviction_per_leg: list[float]
    total_tickets: int


def generate_coupon(
    session: Session,
    target_date: date_type,
    track_name: str,
    start_race_no: int,
    pool_type: str = "6li",
    *,
    max_tickets: int = DEFAULT_MAX_TICKETS,
    predictor: EnsemblePredictor | None = None,
) -> CouponDraft:
    """Generate a multi-race coupon for a window of consecutive races.

    Pulls the ensemble's predictions for each leg, tiers the leg width
    by top-1 conviction, and shrinks the widest legs until the
    ticket-count product fits under ``max_tickets``. Raises if the
    target races don't all exist or the predictor returns nothing.
    """
    if max_tickets < 1:
        raise ValueError("max_tickets must be positive")
    leg_count = {"5li": 5, "6li": 6, "7li": 7}[pool_type]
    end_race_no = start_race_no + leg_count - 1

    track = session.query(Track).filter(Track.name == track_name).first()
    if track is None:
        raise ValueError(f"unknown track {track_name!r}")

    races = (
        session.query(Race)
        .filter(
            Race.date == target_date,
            Race.track_id == track.id,
            Race.race_number >= start_race_no,
            Race.race_number <= end_race_no,
        )
        .order_by(Race.race_number)
        .all()
    )
    if len(races) != leg_count:
        raise ValueError(
            f"expected {leg_count} races starting at R{start_race_no} on "
            f"{track_name} {target_date}, got {len(races)}"
        )

    pred = predictor or EnsemblePredictor(session)

    legs: list[list[int]] = []
    convictions: list[float] = []
    for race in races:
        preds = pred.predict(race.id)
        if not preds:
            raise ValueError(
                f"no predictions for race {race.id} (R{race.race_number})"
            )

        top1_prob = float(preds[0].mean_probability) / 100.0
        convictions.append(round(top1_prob, 4))

        width = _leg_width(top1_prob)
        kept_ids = [p.horse_id for p in preds[:width]]
        # Map horse_id → gate_number (program NO).
        gate_by_hid = {
            e.horse_id: e.gate_number
            for e in race.entries
            if e.gate_number is not None
        }
        kept_gates = [
            int(gate_by_hid[hid]) for hid in kept_ids if hid in gate_by_hid
        ]
        if not kept_gates:
            raise ValueError(
                f"no gate numbers for kept horses in race {race.id}"
            )
        legs.append(kept_gates)

    # Budget cap: shrink the widest leg (preferring lower-conviction
    # to lower-conviction so we don't strip a "spread" leg's coverage
    # before a "medium" one).
    while _product(legs) > max_tickets:
        # Sort legs by (width desc, conviction asc) — widest+lowest
        # conviction first.
        target = max(
            range(len(legs)),
            key=lambda i: (len(legs[i]), -convictions[i]),
        )
        if len(legs[target]) <= 1:
            break  # can't shrink further
        legs[target] = legs[target][:-1]

    return CouponDraft(
        kept_horses_per_leg=legs,
        conviction_per_leg=convictions,
        total_tickets=_product(legs),
    )


def _product(legs: Sequence[Sequence]) -> int:
    return reduce(mul, (len(leg) for leg in legs), 1)


def persist_coupon(
    session: Session,
    target_date: date_type,
    track_name: str,
    start_race_no: int,
    draft: CouponDraft,
    *,
    pool_type: str = "6li",
    pool_index: int | None = None,
    strategy: str = "asymmetric_v1",
    ticket_unit_tl: float = DEFAULT_TICKET_UNIT_TL,
) -> MultiRacePick:
    """Write a CouponDraft to ``multi_race_picks``. Idempotent on
    (date, track, pool_type, pool_index, strategy) — re-running with
    the same content returns the immutable existing row."""
    track = session.query(Track).filter(Track.name == track_name).first()
    if track is None:
        raise ValueError(f"unknown track {track_name!r}")

    leg_count = len(draft.kept_horses_per_leg)
    end_race_no = start_race_no + leg_count - 1
    stake_tl = draft.total_tickets * ticket_unit_tl

    if ticket_unit_tl <= 0 or draft.total_tickets != _product(draft.kept_horses_per_leg):
        raise ValueError("Invalid coupon stake or ticket count")
    if leg_count != {"5li": 5, "6li": 6, "7li": 7}.get(pool_type):
        raise ValueError("Pool leg count mismatch")
    pools = session.query(MultiRacePool).filter(
        MultiRacePool.date == target_date, MultiRacePool.track_id == track.id,
        MultiRacePool.pool_type == pool_type).all()
    if pool_index is None:
        matches = [p for p in pools if (p.start_race_no, p.end_race_no) == (start_race_no, end_race_no)]
        if len(matches) != 1:
            raise ValueError("Pool window is ambiguous; explicitly provide pool_index and start race")
        pool_index = matches[0].pool_index
    if pool_index < 1:
        raise ValueError("pool_index must be positive")
    pool = next((p for p in pools if p.pool_index == pool_index), None)
    if pool and pool.start_race_no is not None and (pool.start_race_no, pool.end_race_no) != (start_race_no, end_race_no):
        raise ValueError("Pool index belongs to a different race window")
    existing = session.query(MultiRacePick).filter_by(
        date=target_date, track_id=track.id, pool_type=pool_type,
        pool_index=pool_index, strategy=strategy).first()
    if existing is not None:
        if (existing.start_race_no, existing.end_race_no, existing.kept_horses_per_leg,
            float(existing.ticket_unit_tl)) != (start_race_no, end_race_no, draft.kept_horses_per_leg, ticket_unit_tl):
            raise ValueError("Recorded coupons are immutable; use a new strategy revision before post time")
        return existing
    races = session.query(Race).filter(Race.date == target_date, Race.track_id == track.id,
                Race.race_number.between(start_race_no, end_race_no)).all()
    if len(races) != leg_count or not all(is_upcoming(r) for r in races):
        raise ValueError("Coupons may only be recorded before every leg's known post time")
    if pool is None:
        pool = MultiRacePool(date=target_date, track_id=track.id, pool_type=pool_type,
                            pool_index=pool_index, start_race_no=start_race_no, end_race_no=end_race_no)
        session.add(pool)
    else:
        pool.start_race_no, pool.end_race_no = start_race_no, end_race_no
    if existing is None:
        existing = MultiRacePick(
            date=target_date,
            generated_at=utcnow(),
            track_id=track.id,
            pool_type=pool_type,
            pool_index=pool_index,
            strategy=strategy,
            start_race_no=start_race_no,
            end_race_no=end_race_no,
            kept_horses_per_leg=draft.kept_horses_per_leg,
            total_tickets=draft.total_tickets,
            ticket_unit_tl=ticket_unit_tl,
            stake_tl=stake_tl,
            conviction_per_leg=draft.conviction_per_leg,
        )
        session.add(existing)
    session.flush()
    return existing


def _parse_winning_combo(combo: str) -> list[list[int]]:
    """``"1/4/2/1/3,6,7,10/4,11"`` → ``[[1],[4],[2],[1],[3,6,7,10],[4,11]]``."""
    out = []
    for leg_str in combo.split("/"):
        winners = [int(x) for x in leg_str.split(",") if x.strip().isdigit()]
        out.append(winners)
    return out


def grade_pick(
    session: Session, pick: MultiRacePick,
) -> bool | None:
    """Compare ``pick`` against the matching ``MultiRacePool`` row.

    Returns True if all legs hit, False if any leg missed, None if
    the pool isn't resulted yet. Writes the outcome onto the pick
    row (``hit``, ``payout_tl``, ``net_tl``, ``graded``, ``graded_at``).

    Payout: TJK quotes pool payouts per 1 TL hisse; we scale by
    ``pick.ticket_unit_tl`` and multiply by the number of *our*
    winning combinations (commas inside a leg of the pool's combo
    can produce >1 winner per leg, so coupons that kept multiple
    of those alternates collect more).
    """
    pool = (
        session.query(MultiRacePool)
        .filter(
            MultiRacePool.date == pick.date,
            MultiRacePool.track_id == pick.track_id,
            MultiRacePool.pool_type == pick.pool_type,
            MultiRacePool.pool_index == pick.pool_index,
        )
        .first()
    )
    if pool is None or pool.winning_combo is None or pool.payout_tl is None:
        return None

    if (pool.start_race_no, pool.end_race_no) != (pick.start_race_no, pick.end_race_no):
        logger.warning("Unverified pool window for pick %s", pick.id)
        return None
    if pick.graded and pick.payout_tl is not None:
        return pick.hit
    leg_winners = _parse_winning_combo(pool.winning_combo)
    if len(leg_winners) != len(pick.kept_horses_per_leg):
        logger.warning(
            "leg-count mismatch grading pick %s: pool=%d kept=%d",
            pick.id, len(leg_winners), len(pick.kept_horses_per_leg),
        )
        return None

    all_hit = all(
        any(w in kept for w in winners)
        for kept, winners in zip(pick.kept_horses_per_leg, leg_winners)
    )

    payout_tl = 0.0
    if all_hit and pool.payout_tl is not None:
        n_winning_combos = 1
        for kept, winners in zip(pick.kept_horses_per_leg, leg_winners):
            n_winning_combos *= sum(1 for w in winners if w in kept)
        payout_tl = float(pool.payout_tl) * float(pick.ticket_unit_tl) * n_winning_combos

    pick.hit = all_hit
    pick.payout_tl = round(payout_tl, 2)
    pick.net_tl = round(payout_tl - float(pick.stake_tl), 2)
    pick.graded = True
    pick.graded_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session.flush()
    return all_hit


def multi_strategy_summary(
    session: Session, *, strategy: str | None = None, since=None,
) -> dict:
    """Aggregate ROI per strategy over graded multi-race picks.

    Returns the same shape as ``ganyan.predictor.picks.strategy_summary``::

        {
          "asymmetric_v1": {n, hits, stake_tl, payout_tl, net_tl,
                            roi_pct, hit_rate_pct},
          ...
        }
    """
    q = session.query(MultiRacePick).filter(
        MultiRacePick.graded == True,  # noqa: E712
    )
    if strategy:
        q = q.filter(MultiRacePick.strategy == strategy)
    if since is not None:
        q = q.filter(MultiRacePick.generated_at >= since)

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


def grade_all_pending_multi(session: Session) -> int:
    """Grade every ungraded MultiRacePick whose pool has a winning_combo.

    Mirrors ``ganyan.predictor.picks.grade_all_pending`` for the
    program-level multi-race coupons. Returns the number of picks
    actually graded (i.e., whose pool was resulted at scan time).
    """
    pending = (
        session.query(MultiRacePick)
        .filter(MultiRacePick.graded == False)  # noqa: E712
        .all()
    )
    graded = 0
    for pick in pending:
        outcome = grade_pick(session, pick)
        if outcome is not None:
            graded += 1
    return graded
