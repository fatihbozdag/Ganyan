"""Evaluation module for prediction accuracy on resulted races."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass
class EvaluationSummary:
    total_races: int
    top1_accuracy: float  # % of races where top pick won
    top3_accuracy: float  # % of races where winner was in top 3
    avg_winner_rank: float  # average rank we gave to the actual winner
    avg_winner_probability: float  # average probability we assigned to winner
    log_loss: float  # lower is better
    roi_simulation: float  # return on investment if betting top pick


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
    )


def evaluate_all(
    session: Session,
) -> tuple[EvaluationSummary, list[RaceEvaluation]]:
    """Evaluate all resulted races that have predictions.

    Returns an EvaluationSummary and a list of per-race evaluations.
    """
    resulted_races = (
        session.query(Race)
        .filter(Race.status == RaceStatus.resulted)
        .order_by(Race.date.desc(), Race.race_number.desc())
        .all()
    )

    evaluations: list[RaceEvaluation] = []
    for race in resulted_races:
        ev = evaluate_race(session, race.id)
        if ev is not None:
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
                roi_simulation=0.0,
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

    # Log loss: -mean(log(predicted_prob_of_winner / 100))
    # Clamp probability to avoid log(0).
    log_losses: list[float] = []
    for ev in evaluations:
        if ev.winner_predicted_prob is not None and ev.winner_predicted_prob > 0:
            log_losses.append(-math.log(ev.winner_predicted_prob / 100.0))
    log_loss = sum(log_losses) / len(log_losses) if log_losses else 0.0

    # ROI simulation: flat 100 TL on top pick each race.
    # If top pick wins, payout ~ 100 / (predicted_prob / 100) = 10000 / predicted_prob.
    total_bet = total * 100.0
    total_payout = 0.0
    for ev in evaluations:
        if ev.top1_correct and ev.winner_predicted_prob is not None and ev.winner_predicted_prob > 0:
            total_payout += 10000.0 / ev.winner_predicted_prob
    roi = (total_payout - total_bet) / total_bet if total_bet > 0 else 0.0

    summary = EvaluationSummary(
        total_races=total,
        top1_accuracy=(top1_count / total) * 100.0,
        top3_accuracy=(top3_count / total) * 100.0,
        avg_winner_rank=avg_rank,
        avg_winner_probability=avg_prob,
        log_loss=log_loss,
        roi_simulation=roi,
    )
    return summary, evaluations
