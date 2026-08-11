"""Strategy-level bet ledger: generate picks and grade them later.

Per race we record four Picks, one per strategy in :data:`STRATEGIES`.
This ledger is audit-of-record and stays continuous across advice-list
changes (e.g., a strategy that's been retired from the advice gate
keeps writing rows here for ROI tracking).

Live ledger ROI on graded picks since 2025-01-01 (as of 2026-05-02):

  - ``ganyan_top1``       1857 picks, 37.6% top-1 hit, ROI −19.9%
  - ``uclu_top1``          854 picks,  3.9% hit,        ROI −32.7%
  - ``uclu_box6``          854 picks, 13.6% hit,        ROI −5.3%
  - ``sirali_ikili_top1`` 1857 picks, 12.4% hit,        ROI −12.5%

The advice gate (CLI ``ganyan advice`` + web ``/advice``) advises a
subset — see ``BETTING_STRATEGIES`` in ``cli/main.py:2031`` and
``web/routes.py:1279``. Currently advice = ``("uclu_box6",
"sirali_ikili_top1")``; ``uclu_top1`` was retired 2026-05-02 (gated
ROI −100% on n=16). See ``feedback_advice_excludes_uclu_top1.md``.

``generate_picks_for_race()`` computes picks from the current model's
predicted probabilities and inserts a :class:`~ganyan.db.models.Pick`
row each. Idempotent — the ``(race_id, strategy)`` unique constraint
prevents duplicates; re-running with ``refresh=True`` rewrites
ungraded rows.

``grade_race()`` fills in ``hit``, ``payout_tl``, ``net_tl`` using
the scraped finish positions and TJK payouts once the race resolves.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Iterable

from sqlalchemy.orm import Session

from ganyan.db.models import Pick, Race, RaceEntry, RaceStatus
from ganyan.predictor.exotics import (
    ganyan_probabilities, plase_probabilities,
    sirali_ikili_probabilities, uclu_probabilities,
)


logger = logging.getLogger(__name__)


STAKE_PER_TICKET_TL = 100.0


# Ledger set — every Pick row written by generate_picks_for_race uses
# one of these.  Distinct from the *advice* set: BETTING_STRATEGIES
# in cli/main.py:2031 and web/routes.py:1279 chooses a subset to surface
# in /advice.  Keep this tuple unchanged when retiring a strategy from
# advice — the ledger must stay continuous for ROI tracking.
STRATEGIES = (
    "ganyan_top1", "uclu_top1", "uclu_box6",
    "sirali_ikili_top1", "plase_top1",
)


# Per-pool minimum bilet stake ("birim fiyat") in TL.
#
# Critical: TJK publishes pool payouts as "what one bilet at the default
# birim wins" — NOT per 1 TL staked.  For ganyan/sirali_ikili the birim
# is 1 TL so per-bilet equals per-TL and our older math accidentally
# came out right.  For Sıralı Üçlü the birim is 2 TL, so naïve
# `payout_value × stake` overstates the winnings by 2×.
#
# Verified Apr 30 2026 against operator's actual bilet 68671854006416
# on Ankara R8 (uclu_box6 with K-Komple, 4 horses, Misli=4):
#   Tutar 192 ₺, pool figure 92.50, real payout 370 ₺.
#   Reconciles only as `92.50 × Misli=4` — i.e. 92.50 is per-bilet at
#   2 ₺ birim, not per-TL.  Per-TL rate was 46.25.
BIRIM_TL_BY_STRATEGY = {
    "ganyan_top1": 1.0,
    "sirali_ikili_top1": 1.0,  # tentative — to verify with a real bilet
    "uclu_top1": 2.0,           # verified
    "uclu_box6": 2.0,           # verified
    "plase_top1": 1.0,          # min stake; payout pool not scraped (2026-05-06)
}


def _birim_tl(strategy: str) -> float:
    return BIRIM_TL_BY_STRATEGY.get(strategy, 1.0)


def generate_picks_for_race(
    session: Session, race_id: int, *, refresh: bool = False,
) -> list[Pick]:
    """Generate (or refresh) this race's picks from stored predictions.

    Reads ``RaceEntry.predicted_probability`` — written by the
    predictor's ``predict_and_save`` — and derives Harville joint
    probabilities.

    When ``refresh`` is False (default), safe to call repeatedly;
    existing rows are untouched, missing ones are inserted.

    When ``refresh`` is True, **ungraded** picks (those without a
    settled hit/payout) are deleted and regenerated from the current
    predictions.  Already-graded picks are preserved as historical
    truth of what we actually staked.  Use this after a re-predict
    (e.g. surface change, fresh scrape) to keep picks in sync with
    the latest probabilities.
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
        if getattr(e, "scratched", False):
            continue
        if e.predicted_probability is not None:
            win_probs[e.horse_id] = max(float(e.predicted_probability), 0.0) / 100.0
            name_for[e.horse_id] = e.horse.name if e.horse else "?"
    if sum(win_probs.values()) <= 0:
        return []
    # Renormalise so probabilities sum to 1 across remaining runners.
    total = sum(win_probs.values())
    if total > 0:
        win_probs = {h: v / total for h, v in win_probs.items()}

    if refresh:
        # Delete ungraded picks so we can regenerate from fresh probs.
        # An "ungraded" pick is one whose hit field is still NULL —
        # i.e. the race hasn't been graded yet, or the pick was
        # generated for a pool that hadn't formed at grade time.
        (
            session.query(Pick)
            .filter(Pick.race_id == race_id, Pick.hit.is_(None))
            .delete(synchronize_session=False)
        )
        session.flush()

    existing = {
        p.strategy: p for p in
        session.query(Pick).filter(Pick.race_id == race_id).all()
    }
    added: list[Pick] = []

    # ganyan_top1 — the reference/baseline.  High hit rate (~37% on
    # AGF favourites) but long-run losing due to takeout.  Useful for
    # comparison and for psychological feedback ("did we pick the
    # winner?") that exotic-pool ROI alone doesn't give.
    gan = ganyan_probabilities(win_probs)
    if gan and "ganyan_top1" not in existing:
        top = gan[0]
        added.append(_make_pick(
            race_id=race_id,
            strategy="ganyan_top1",
            combination=list(top.horses),
            combination_names=[name_for.get(h, "?") for h in top.horses],
            stake=STAKE_PER_TICKET_TL,
            tickets=1,
            model_prob_pct=top.probability * 100.0,
        ))

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

    # plase_top1 — top-2 finish coverage on the most-likely-top-2 horse.
    # Field-size gate: TJK plase pool only runs with field ≥ 8 (pays
    # 1st-2nd; ≥16 pays 1st-2nd-3rd). On 2026-05-11 audit, 24% of
    # plase_top1 picks were in fields ≤ 7 where the pool can NEVER pay
    # — those are -100% drag on the strategy. Skip them at source.
    if (
        "plase_top1" not in existing
        and len(win_probs) >= 8
    ):
        plase = plase_probabilities(win_probs, top_k=2)
        if plase:
            top = plase[0]
            added.append(_make_pick(
                race_id=race_id,
                strategy="plase_top1",
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


def refresh_picks_for_date(session: Session, target_date) -> int:
    """Delete ungraded picks for a date and regenerate from current predictions.

    Use after a re-predict (intraday surface change, fresh scrape, etc.)
    so the ``picks`` table reflects the latest probabilities rather than
    the morning snapshot.  Graded picks (races whose hit/payout is
    already settled) are preserved as historical truth.

    Returns the total number of picks newly inserted.
    """
    from ganyan.db.models import Race
    races = (
        session.query(Race)
        .filter(Race.date == target_date)
        .all()
    )
    total_added = 0
    for race in races:
        added = generate_picks_for_race(session, race.id, refresh=True)
        total_added += len(added)
    session.flush()
    return total_added


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
    if strategy == "ganyan_top1":
        if len(actual) < 1:
            return None
        return combination[0] == actual[0]
    if strategy == "uclu_top1":
        if len(actual) < 3:
            return None
        return tuple(combination) == actual[:3]
    if strategy == "uclu_box6":
        if len(actual) < 3:
            return None
        return set(combination) == set(actual[:3])
    if strategy == "sirali_ikili_top1":
        if len(actual) < 2:
            return None
        return tuple(combination) == actual[:2]
    if strategy == "plase_top1":
        if len(actual) < 2:
            return None
        return combination[0] in actual[:2]
    return None


def _strategy_payout_tl(strategy: str, race: Race) -> float | None:
    if strategy == "ganyan_top1":
        v = race.ganyan_payout_tl
    elif strategy == "sirali_ikili_top1":
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

        # plase_top1 — Plase pool publishes per-horse payouts as
        # ``PLASE <program_no> <amount>``.  We persist that on the
        # picked horse's race_entry; on hit, the bilet's settled value
        # is `entry.plase_payout_tl × stake_per_ticket`.  When the
        # scrape ran before the plase_payout_tl column existed (any
        # race scraped before 2026-05-06) the entry value will be NULL
        # — keep payout/net NULL so the row carries hit-rate signal
        # only and a future re-scrape can fill the ROI side.
        if pick.strategy == "plase_top1":
            pick.hit = hit
            pick.graded = True
            pick.graded_at = now
            picked_horse_id = pick.combination[0] if pick.combination else None
            picked_entry = next(
                (e for e in entries if e.horse_id == picked_horse_id), None,
            )
            plase_value = (
                float(picked_entry.plase_payout_tl)
                if picked_entry is not None
                and picked_entry.plase_payout_tl is not None
                else None
            )
            if hit and plase_value is not None:
                winning_ticket_payout = (
                    plase_value * STAKE_PER_TICKET_TL / _birim_tl(pick.strategy)
                )
                pick.payout_tl = round(winning_ticket_payout, 2)
                pick.net_tl = round(winning_ticket_payout - float(pick.stake_tl), 2)
            elif not hit and plase_value is not None:
                # We have payout-pool confirmation that the bet failed
                # → record the loss in TL (mirrors other strategies'
                # miss path).
                pick.payout_tl = 0.0
                pick.net_tl = -float(pick.stake_tl)
            else:
                # Pool data missing — hit is recorded for hit-rate
                # tracking, ROI deferred until re-scrape lands.
                pick.payout_tl = None
                pick.net_tl = None
            graded += 1
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
            # Pool figure is per-bilet at the pool's birim — divide by
            # birim so that `value × stake_per_ticket / birim` gives the
            # actual TL payout. (Pre-2026-04-30 this divisor was missing;
            # üçlü hits were overstated 2×.)
            birim = _birim_tl(pick.strategy)
            winning_ticket_payout = (
                payout_per_tl * STAKE_PER_TICKET_TL / birim
            )
            pick.payout_tl = round(winning_ticket_payout, 2)
            pick.net_tl = round(winning_ticket_payout - float(pick.stake_tl), 2)
        else:
            pick.payout_tl = 0.0
            pick.net_tl = -float(pick.stake_tl)
        graded += 1

    if graded:
        session.flush()
    return graded


def resettle_plase_picks(session: Session) -> int:
    """Fill payout/net on already-graded plase_top1 picks whose horse's
    plase_payout_tl has since been populated.

    Use after a re-scrape adds ``race_entries.plase_payout_tl`` to races
    we'd previously graded with NULL payout (every race scraped before
    2026-05-06 falls in this bucket until re-fetched).  Idempotent.
    Returns the number of picks whose payout/net were updated.
    """
    rows = (
        session.query(Pick)
        .filter(
            Pick.strategy == "plase_top1",
            Pick.graded.is_(True),
            Pick.payout_tl.is_(None),
        )
        .all()
    )
    updated = 0
    for pick in rows:
        if not pick.combination:
            continue
        picked_horse_id = pick.combination[0]
        entry = (
            session.query(RaceEntry)
            .filter(
                RaceEntry.race_id == pick.race_id,
                RaceEntry.horse_id == picked_horse_id,
            )
            .first()
        )
        if entry is None or entry.plase_payout_tl is None:
            continue
        if pick.hit:
            winning_ticket_payout = (
                float(entry.plase_payout_tl)
                * STAKE_PER_TICKET_TL
                / _birim_tl(pick.strategy)
            )
            pick.payout_tl = round(winning_ticket_payout, 2)
            pick.net_tl = round(winning_ticket_payout - float(pick.stake_tl), 2)
        else:
            pick.payout_tl = 0.0
            pick.net_tl = -float(pick.stake_tl)
        updated += 1
    if updated:
        session.flush()
    return updated


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
            "n": 0, "hits": 0, "priced_n": 0,
            "stake_tl": 0.0, "priced_stake_tl": 0.0,
            "payout_tl": 0.0, "net_tl": 0.0,
        })
        row["n"] += 1
        row["stake_tl"] += float(p.stake_tl)
        if p.hit:
            row["hits"] += 1
        if p.payout_tl is not None:
            row["priced_n"] += 1
            row["priced_stake_tl"] += float(p.stake_tl)
            row["payout_tl"] += float(p.payout_tl)
        if p.net_tl is not None:
            row["net_tl"] += float(p.net_tl)

    for row in agg.values():
        row["hit_rate_pct"] = (row["hits"] / row["n"]) * 100 if row["n"] else 0
        # ROI is computed over PRICED stake only — picks without payout
        # data don't belong in the denominator (they'd silently inflate
        # losses). For strategies where every pick is priced (sirali_ikili,
        # uclu, ganyan), priced_stake_tl == stake_tl and the math matches
        # the legacy formula. Callers needing the full-stake denominator
        # can divide net_tl by stake_tl themselves.
        denom = row["priced_stake_tl"] or row["stake_tl"]
        row["roi_pct"] = (row["net_tl"] / denom) * 100 if denom else 0
    return agg
