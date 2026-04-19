from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import numpy as np
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from ganyan.db.models import Race, RaceEntry, RaceStatus


# Bayesian-smoothing prior for win-rate style features — keeps jockeys
# with a handful of races from dominating purely by low sample size.
_WINRATE_PRIOR_MEAN = 0.10  # typical baseline ≈ 10% across all jockeys
_WINRATE_PRIOR_WEIGHT = 20  # equivalent to "20 pseudo-races at 10%"


@dataclass
class HorseFeatures:
    speed_figure: float | None = None
    form_cycle: float | None = None
    weight_delta: float | None = None
    rest_fitness: float | None = None
    class_indicator: float | None = None
    jockey_win_rate: float | None = None
    trainer_win_rate: float | None = None
    gate_bias: float | None = None
    surface_affinity: float | None = None
    track_affinity: float | None = None  # retained for compatibility


def compute_speed_figure(
    eid_seconds: float | None, distance_meters: int | None
) -> float | None:
    """Normalize EID into speed figure (meters per second)."""
    if eid_seconds is None or distance_meters is None or eid_seconds <= 0:
        return None
    return distance_meters / eid_seconds


def compute_form_cycle(
    positions: list[int | None] | None,
) -> float | None:
    """Compute form score from recent finishing positions.

    Exponential decay weighting (most recent = highest weight).
    Returns 0-1 where 1 = best form.
    """
    if not positions:
        return None
    valid = [(i, p) for i, p in enumerate(positions) if p is not None]
    if not valid:
        return None
    n = len(positions)
    scores = []
    weights = []
    for i, pos in valid:
        weight = np.exp((i - n + 1) * 0.7)
        score = max(0.0, 1.0 - (pos - 1) * 0.15)
        scores.append(score)
        weights.append(weight)
    weights = np.array(weights)
    scores = np.array(scores)
    return float(np.average(scores, weights=weights))


def compute_weight_delta(
    horse_weight: float | None, field_avg_weight: float | None
) -> float | None:
    """Positive = lighter than average (advantage)."""
    if horse_weight is None or field_avg_weight is None:
        return None
    return (field_avg_weight - horse_weight) / field_avg_weight


def compute_rest_fitness(kgs: int | None) -> float | None:
    """Gaussian curve centered on 21 days optimal rest."""
    if kgs is None:
        return None
    optimal = 21.0
    sigma = 15.0
    return float(np.exp(-((kgs - optimal) ** 2) / (2 * sigma**2)))


def compute_class_indicator(
    hp: float | None, field_avg_hp: float | None
) -> float | None:
    """Positive = horse has higher HP than field average."""
    if hp is None or field_avg_hp is None or field_avg_hp == 0:
        return None
    return (hp - field_avg_hp) / field_avg_hp


def compute_jockey_win_rate(
    session: Session,
    jockey: str | None,
    before_date: date_type | None = None,
) -> float | None:
    """Smoothed jockey win rate over historical resulted races.

    Returns a value in ``[0, 1]`` where 0.1 is the population baseline.
    ``before_date`` enforces temporal integrity — never look at races
    that happened on or after the race being predicted.
    """
    if not jockey:
        return None
    return _smoothed_person_win_rate(
        session, RaceEntry.jockey, jockey, before_date,
    )


def compute_trainer_win_rate(
    session: Session,
    trainer: str | None,
    before_date: date_type | None = None,
) -> float | None:
    """Smoothed trainer win rate over historical resulted races.

    Trainer is stored on :class:`Horse` (a snapshot of the *current*
    trainer).  This currently reflects that: a horse is credited to its
    present trainer across all historical runs.  Good enough for a
    first-pass feature; accuracy improves when we add a trainer-history
    table.
    """
    if not trainer:
        return None
    from ganyan.db.models import Horse  # local import to avoid cycle

    base = (
        session.query(RaceEntry)
        .join(Horse, Horse.id == RaceEntry.horse_id)
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            Horse.trainer == trainer,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date is not None:
        base = base.filter(Race.date < before_date)

    runs = base.with_entities(func.count(RaceEntry.id)).scalar() or 0
    if runs == 0:
        return None
    wins = (
        base.filter(RaceEntry.finish_position == 1)
        .with_entities(func.count(RaceEntry.id))
        .scalar()
    ) or 0
    return _bayesian_smoothed_rate(wins, runs)


def _smoothed_person_win_rate(
    session: Session,
    column,
    value: str,
    before_date: date_type | None,
) -> float | None:
    """Internal: compute win rate where a direct RaceEntry column equals value."""
    q = (
        session.query(
            func.count(RaceEntry.id).label("runs"),
        )
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            column == value,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date is not None:
        q = q.filter(Race.date < before_date)
    runs_row = q.one()
    runs = runs_row.runs or 0
    if runs == 0:
        return None

    wins_q = (
        session.query(func.count(RaceEntry.id))
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            column == value,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position == 1,
        )
    )
    if before_date is not None:
        wins_q = wins_q.filter(Race.date < before_date)
    wins = wins_q.scalar() or 0
    return _bayesian_smoothed_rate(wins, runs)


def _bayesian_smoothed_rate(wins: int, runs: int) -> float:
    """Apply a Beta(prior_mean·weight, (1-prior_mean)·weight) smoothing."""
    alpha = _WINRATE_PRIOR_MEAN * _WINRATE_PRIOR_WEIGHT
    beta = (1.0 - _WINRATE_PRIOR_MEAN) * _WINRATE_PRIOR_WEIGHT
    return float((wins + alpha) / (runs + alpha + beta))


def compute_gate_bias(
    gate_number: int | None,
    distance_meters: int | None,
    surface: str | None,
) -> float | None:
    """Heuristic gate-bias score in ``[-1, 1]``.

    Short sand (kum) races advantage inside gates; long turf (çim) races
    are relatively neutral.  This is a coarse prior — feed the empirical
    gate-specific win rate when enough data is accumulated.
    """
    if gate_number is None:
        return None
    if distance_meters is None:
        return 0.0

    # Normalise gate to 0..1 assuming a typical 14-horse field.
    normalised = (gate_number - 1) / 13.0

    if surface and surface.lower().startswith("kum") and distance_meters <= 1400:
        # Inside gates slightly favoured on short sand.
        return float(1.0 - 2 * normalised) * 0.5
    if surface and surface.lower().startswith("çim") and distance_meters >= 1800:
        # Mid gates slightly favoured on long turf (draw matters less).
        return float(1.0 - abs(normalised - 0.5) * 2.0) * 0.3

    # Default: inside-tilt is small.
    return float(0.5 - normalised) * 0.2


def compute_surface_affinity(
    session: Session,
    horse_id: int,
    surface: str | None,
    distance_meters: int | None,
    before_date: date_type | None = None,
    distance_band: int = 200,
) -> float | None:
    """Horse's win rate on similar surface/distance combination.

    Returns ``None`` if the horse has no historical runs matching the
    profile.  Otherwise returns a smoothed win rate in ``[0, 1]``.
    """
    if horse_id is None:
        return None

    filters = [
        RaceEntry.horse_id == horse_id,
        RaceEntry.finish_position.isnot(None),
        Race.status == RaceStatus.resulted,
    ]
    if surface is not None:
        filters.append(Race.surface == surface)
    if distance_meters is not None:
        filters.append(
            and_(
                Race.distance_meters >= distance_meters - distance_band,
                Race.distance_meters <= distance_meters + distance_band,
            )
        )
    if before_date is not None:
        filters.append(Race.date < before_date)

    runs = (
        session.query(func.count(RaceEntry.id))
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(*filters)
        .scalar()
    ) or 0
    if runs == 0:
        return None

    wins = (
        session.query(func.count(RaceEntry.id))
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(*filters, RaceEntry.finish_position == 1)
        .scalar()
    ) or 0
    return _bayesian_smoothed_rate(wins, runs)


def extract_features(
    eid_seconds: float | None = None,
    distance_meters: int | None = None,
    last_six_parsed: list[int | None] | None = None,
    weight_kg: float | None = None,
    field_avg_weight: float | None = None,
    kgs: int | None = None,
    hp: float | None = None,
    field_avg_hp: float | None = None,
    *,
    session: Session | None = None,
    jockey: str | None = None,
    trainer: str | None = None,
    horse_id: int | None = None,
    gate_number: int | None = None,
    surface: str | None = None,
    race_date: date_type | None = None,
) -> HorseFeatures:
    features = HorseFeatures(
        speed_figure=compute_speed_figure(eid_seconds, distance_meters),
        form_cycle=compute_form_cycle(last_six_parsed),
        weight_delta=compute_weight_delta(weight_kg, field_avg_weight),
        rest_fitness=compute_rest_fitness(kgs),
        class_indicator=compute_class_indicator(hp, field_avg_hp),
        gate_bias=compute_gate_bias(gate_number, distance_meters, surface),
    )
    if session is not None:
        features.jockey_win_rate = compute_jockey_win_rate(
            session, jockey, before_date=race_date,
        )
        features.trainer_win_rate = compute_trainer_win_rate(
            session, trainer, before_date=race_date,
        )
        if horse_id is not None:
            features.surface_affinity = compute_surface_affinity(
                session, horse_id, surface, distance_meters,
                before_date=race_date,
            )
            features.track_affinity = features.surface_affinity
    return features
