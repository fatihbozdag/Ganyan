"""Evaluation module for prediction accuracy on resulted races."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date as date_type

from sqlalchemy.orm import Session

from ganyan.db.models import Race, RaceEntry, RaceStatus


@dataclass
class RaceEvaluation:
    race_id: int
    track: str
    date: date_type
    race_number: int
    num_horses: int
    winner_name: str
    winner_predicted_prob: float | None  # probability we gave the winner
    winner_predicted_rank: int | None  # where we ranked the winner (1 = top pick)
    top1_correct: bool  # did our #1 pick win?
    top3_correct: bool  # was winner in our top 3?
    agf_leader_correct: bool | None = None  # market (AGF) baseline top-1


@dataclass
class CalibrationBucket:
    lower: float  # bucket lower bound (0-100)
    upper: float
    count: int  # number of horses with predictions in this bucket
    mean_predicted: float  # mean predicted probability in the bucket
    actual_win_rate: float  # observed win rate in the bucket


@dataclass
class EvaluationSummary:
    total_races: int
    top1_accuracy: float  # % of races where top pick won
    top3_accuracy: float  # % of races where winner was in top 3
    avg_winner_rank: float  # average rank we gave to the actual winner
    avg_winner_probability: float  # average probability we assigned to winner
    log_loss: float  # lower is better
    brier_score: float  # proper multiclass scoring rule (lower is better)
    random_baseline_top1: float  # expected accuracy from uniform random picking
    agf_baseline_top1: float | None  # top-1 accuracy of the AGF (market) leader
    roi_simulation: float  # return on flat-bet top pick (synthetic odds)
    calibration: list[CalibrationBucket] = field(default_factory=list)
    cutoff_date: date_type | None = None  # earliest race date evaluated
    skipped_unresulted: int = 0  # races without finish data
    skipped_unpredicted: int = 0  # resulted races with no prediction slot


def evaluate_race(session: Session, race_id: int) -> RaceEvaluation | None:
    """Evaluate predictions for a single resulted race.

    Returns None if the race is not resulted, has no predictions,
    or has no identifiable winner.
    """
    race = session.get(Race, race_id)
    if race is None or race.status != RaceStatus.resulted:
        return None

    entries = (
        session.query(RaceEntry)
        .filter(RaceEntry.race_id == race_id)
        .all()
    )
    if not entries:
        return None

    # We need entries that have predicted_probability set.
    predicted_entries = [
        e for e in entries if e.predicted_probability is not None
    ]
    if not predicted_entries:
        return None

    # Find the winner (finish_position == 1).
    winners = [e for e in entries if e.finish_position == 1]
    if not winners:
        return None
    winner = winners[0]

    # Sort predicted entries by predicted_probability descending to get ranks.
    ranked = sorted(
        predicted_entries,
        key=lambda e: float(e.predicted_probability),
        reverse=True,
    )

    # Find rank for the winner (1-based).
    winner_predicted_prob = (
        float(winner.predicted_probability)
        if winner.predicted_probability is not None
        else None
    )

    winner_predicted_rank: int | None = None
    for rank_idx, entry in enumerate(ranked, 1):
        if entry.horse_id == winner.horse_id:
            winner_predicted_rank = rank_idx
            break

    top1_correct = (
        winner_predicted_rank == 1 if winner_predicted_rank is not None else False
    )
    top3_correct = (
        winner_predicted_rank is not None and winner_predicted_rank <= 3
    )

    # AGF (market) baseline: does the AGF leader win?
    agf_leader_correct: bool | None = None
    agf_candidates = [e for e in entries if e.agf is not None]
    if agf_candidates:
        agf_leader = max(agf_candidates, key=lambda e: float(e.agf))
        agf_leader_correct = agf_leader.horse_id == winner.horse_id

    track_name = race.track.name if race.track else "?"

    return RaceEvaluation(
        race_id=race_id,
        track=track_name,
        date=race.date,
        race_number=race.race_number,
        num_horses=len(entries),
        winner_name=winner.horse.name if winner.horse else "?",
        winner_predicted_prob=winner_predicted_prob,
        winner_predicted_rank=winner_predicted_rank,
        top1_correct=top1_correct,
        top3_correct=top3_correct,
        agf_leader_correct=agf_leader_correct,
    )


def evaluate_all(
    session: Session,
    cutoff_date: date_type | None = None,
    num_calibration_bins: int = 10,
) -> tuple[EvaluationSummary, list[RaceEvaluation]]:
    """Evaluate resulted races that have predictions.

    Parameters
    ----------
    cutoff_date:
        If provided, only evaluate races on or after this date.  Use this
        to create a proper temporal hold-out: train features on races
        before ``cutoff_date`` and measure accuracy on later races.
    num_calibration_bins:
        Number of equal-width probability buckets for the reliability
        diagram (default 10, i.e. deciles of predicted probability).
    """
    races_q = (
        session.query(Race)
        .filter(Race.status == RaceStatus.resulted)
    )
    if cutoff_date is not None:
        races_q = races_q.filter(Race.date >= cutoff_date)
    resulted_races = (
        races_q.order_by(Race.date.desc(), Race.race_number.desc()).all()
    )

    evaluations: list[RaceEvaluation] = []
    skipped_unresulted = 0
    skipped_unpredicted = 0
    for race in resulted_races:
        ev = evaluate_race(session, race.id)
        if ev is None:
            # evaluate_race returns None for three reasons: race not
            # resulted, no predictions, or no winner identified.  We
            # conservatively classify the last two as unpredicted; the
            # first is already filtered by the query.
            skipped_unpredicted += 1
            continue
        evaluations.append(ev)

    if not evaluations:
        return (
            EvaluationSummary(
                total_races=0,
                top1_accuracy=0.0,
                top3_accuracy=0.0,
                avg_winner_rank=0.0,
                avg_winner_probability=0.0,
                log_loss=0.0,
                brier_score=0.0,
                random_baseline_top1=0.0,
                agf_baseline_top1=None,
                roi_simulation=0.0,
                calibration=[],
                cutoff_date=cutoff_date,
                skipped_unresulted=skipped_unresulted,
                skipped_unpredicted=skipped_unpredicted,
            ),
            [],
        )

    total = len(evaluations)
    top1_count = sum(1 for ev in evaluations if ev.top1_correct)
    top3_count = sum(1 for ev in evaluations if ev.top3_correct)

    ranks = [
        ev.winner_predicted_rank
        for ev in evaluations
        if ev.winner_predicted_rank is not None
    ]
    avg_rank = sum(ranks) / len(ranks) if ranks else 0.0

    probs = [
        ev.winner_predicted_prob
        for ev in evaluations
        if ev.winner_predicted_prob is not None
    ]
    avg_prob = sum(probs) / len(probs) if probs else 0.0

    # Log loss: -mean(log(predicted_prob_of_winner / 100)).
    log_losses: list[float] = []
    for ev in evaluations:
        if ev.winner_predicted_prob is not None and ev.winner_predicted_prob > 0:
            log_losses.append(-math.log(ev.winner_predicted_prob / 100.0))
    log_loss = sum(log_losses) / len(log_losses) if log_losses else 0.0

    # Brier score (multi-class): mean over races of sum_i (p_i - y_i)^2,
    # where y_i is 1 for the winner and 0 otherwise.  Lower = sharper
    # AND more accurate.  Measured against the full predicted field per
    # race, not only the winner's probability.
    brier_score = _compute_brier_score(session, evaluations)

    # Random-picking baseline: if a race has N horses, random top-1 hits
    # 1/N of the time.  Averaged over all evaluated races.
    random_baseline = (
        sum(1.0 / ev.num_horses for ev in evaluations if ev.num_horses > 0) / total
    ) * 100.0

    # AGF market baseline.
    agf_decisions = [
        ev.agf_leader_correct
        for ev in evaluations
        if ev.agf_leader_correct is not None
    ]
    agf_baseline = (
        (sum(1 for x in agf_decisions if x) / len(agf_decisions)) * 100.0
        if agf_decisions else None
    )

    # ROI simulation.  Without real parimutuel odds we use an AGF-implied
    # payout proxy when available: payout ≈ 1 / (AGF/100).  Falls back to
    # 1 / (predicted_prob/100) if AGF is missing — which is the same
    # circular estimate as before, flagged here as a known limitation.
    roi = _simulate_roi(session, evaluations)

    # Reliability diagram (calibration).
    calibration = _compute_calibration(
        session, evaluations, num_bins=num_calibration_bins,
    )

    summary = EvaluationSummary(
        total_races=total,
        top1_accuracy=(top1_count / total) * 100.0,
        top3_accuracy=(top3_count / total) * 100.0,
        avg_winner_rank=avg_rank,
        avg_winner_probability=avg_prob,
        log_loss=log_loss,
        brier_score=brier_score,
        random_baseline_top1=random_baseline,
        agf_baseline_top1=agf_baseline,
        roi_simulation=roi,
        calibration=calibration,
        cutoff_date=cutoff_date,
        skipped_unresulted=skipped_unresulted,
        skipped_unpredicted=skipped_unpredicted,
    )
    return summary, evaluations


def _compute_brier_score(
    session: Session, evaluations: list[RaceEvaluation],
) -> float:
    """Multi-class Brier score averaged over evaluated races."""
    total = 0.0
    count = 0
    for ev in evaluations:
        entries = (
            session.query(RaceEntry).filter(RaceEntry.race_id == ev.race_id).all()
        )
        predicted = [
            (float(e.predicted_probability) / 100.0, e.finish_position == 1)
            for e in entries if e.predicted_probability is not None
        ]
        if not predicted:
            continue
        total += sum((p - (1.0 if won else 0.0)) ** 2 for p, won in predicted)
        count += 1
    return total / count if count else 0.0


def _simulate_roi(
    session: Session, evaluations: list[RaceEvaluation],
) -> float:
    """Simulate flat 100 TL bets on the model's top pick.

    Prefers AGF-implied odds (a rough TJK market proxy) when available
    and falls back to model-implied odds.  Result is fractional ROI:
    0.0 = break-even, 0.15 = +15%, -0.20 = -20%.
    """
    total_bet = 0.0
    total_payout = 0.0
    for ev in evaluations:
        bet = 100.0
        total_bet += bet
        if not ev.top1_correct:
            continue

        entries = (
            session.query(RaceEntry).filter(RaceEntry.race_id == ev.race_id).all()
        )
        top_pick = max(
            (e for e in entries if e.predicted_probability is not None),
            key=lambda e: float(e.predicted_probability),
            default=None,
        )
        if top_pick is None:
            continue
        # Prefer AGF-implied payout.
        if top_pick.agf is not None and float(top_pick.agf) > 0:
            implied_prob = float(top_pick.agf) / 100.0
        elif top_pick.predicted_probability is not None and float(top_pick.predicted_probability) > 0:
            implied_prob = float(top_pick.predicted_probability) / 100.0
        else:
            continue
        total_payout += bet / implied_prob

    return (total_payout - total_bet) / total_bet if total_bet > 0 else 0.0


def _compute_calibration(
    session: Session,
    evaluations: list[RaceEvaluation],
    num_bins: int = 10,
) -> list[CalibrationBucket]:
    """Reliability diagram: bucket predictions, compare mean pred to actual."""
    race_ids = [ev.race_id for ev in evaluations]
    if not race_ids:
        return []

    entries = (
        session.query(RaceEntry)
        .filter(
            RaceEntry.race_id.in_(race_ids),
            RaceEntry.predicted_probability.isnot(None),
        )
        .all()
    )

    bin_width = 100.0 / num_bins
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(num_bins)]
    for e in entries:
        prob = float(e.predicted_probability)
        # Clamp to [0, 100) so bin index stays in range.
        idx = min(int(prob / bin_width), num_bins - 1)
        bins[idx].append((prob, e.finish_position == 1))

    out: list[CalibrationBucket] = []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        lower = i * bin_width
        upper = lower + bin_width
        mean_pred = sum(p for p, _ in bucket) / len(bucket)
        actual = sum(1 for _, won in bucket if won) / len(bucket) * 100.0
        out.append(
            CalibrationBucket(
                lower=lower,
                upper=upper,
                count=len(bucket),
                mean_predicted=mean_pred,
                actual_win_rate=actual,
            )
        )
    return out
