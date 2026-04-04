from dataclasses import dataclass

import numpy as np


@dataclass
class HorseFeatures:
    speed_figure: float | None = None
    form_cycle: float | None = None
    weight_delta: float | None = None
    rest_fitness: float | None = None
    class_indicator: float | None = None
    track_affinity: float | None = None  # requires history, filled later


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


def extract_features(
    eid_seconds: float | None = None,
    distance_meters: int | None = None,
    last_six_parsed: list[int | None] | None = None,
    weight_kg: float | None = None,
    field_avg_weight: float | None = None,
    kgs: int | None = None,
    hp: float | None = None,
    field_avg_hp: float | None = None,
) -> HorseFeatures:
    return HorseFeatures(
        speed_figure=compute_speed_figure(eid_seconds, distance_meters),
        form_cycle=compute_form_cycle(last_six_parsed),
        weight_delta=compute_weight_delta(weight_kg, field_avg_weight),
        rest_fitness=compute_rest_fitness(kgs),
        class_indicator=compute_class_indicator(hp, field_avg_hp),
    )
