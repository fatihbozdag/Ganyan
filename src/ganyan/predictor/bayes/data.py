"""Build training/holdout frames for the Bayesian Plackett-Luce model.

Encode horse, jockey, sire, and (track × distance-bucket) as dense
integer indices so PyMC's coords machinery can use them directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ganyan.db.models import Race, RaceStatus
from ganyan.scraper.parser import parse_last_six


DISTANCE_BUCKETS: Tuple[int, ...] = (1300, 1600, 1900, 2400)


def distance_bucket_for(meters: int) -> int:
    for i, boundary in enumerate(DISTANCE_BUCKETS):
        if meters < boundary:
            return i
    return len(DISTANCE_BUCKETS)


@dataclass
class TrainingFrame:
    feature_schema: int = 2
    orderings: Dict[int, List[int]] = field(default_factory=dict)
    horse_index: Dict[int, int] = field(default_factory=dict)
    jockey_index: Dict[str, int] = field(default_factory=dict)
    sire_index: Dict[str, int] = field(default_factory=dict)
    track_dist_index: Dict[Tuple[int, int], int] = field(default_factory=dict)
    jockey_of_horse_in_race: List[int] = field(default_factory=list)
    sire_of_horse_in_race: List[int] = field(default_factory=list)
    track_dist_of_race: Dict[int, int] = field(default_factory=dict)
    agf_of_horse_in_race: List[float] = field(default_factory=list)
    kgs_of_horse_in_race: List[float] = field(default_factory=list)
    s20_of_horse_in_race: List[float] = field(default_factory=list)
    last6_of_horse_in_race: List[float] = field(default_factory=list)
    # Track-variant-adjusted seconds-per-meter, recency-mean over last N
    # prior runs (lower = faster).  NaN = no prior history (within-race
    # mean after z-score, so contributes 0 to PL score).
    speed_of_horse_in_race: List[float] = field(default_factory=list)
    # Workout sec-per-meter, recency-mean of last N prior workouts.
    # NaN = horse has no recorded workouts (cold-start).
    workout_of_horse_in_race: List[float] = field(default_factory=list)
    # Pace preference: mean pace-z-score across the horse's prior top-3
    # finishes (negative = closer type, positive = front-runner type).
    # NaN = no qualifying history.
    pace_of_horse_in_race: List[float] = field(default_factory=list)


def summarize_last_six(s: str | None) -> float:
    """Mean finish position over recorded races (lower=better).

    Returns NaN when no data — missing-aware within-race scaling acts
    as the within-race mean (no-op contribution to the PL score).
    """
    if not s:
        return float("nan")
    parsed = [p for p in parse_last_six(s) if p is not None]
    if not parsed:
        return float("nan")
    return float(sum(parsed)) / len(parsed)


def matrices_for_pymc(frame: TrainingFrame):
    """Build padded matrices for vectorized PL loglik.

    Returns a dict with:
      horse_idx_mat:   (R, K_max) int  — horse indices, 0 padded
      jockey_idx_mat:  (R, K_max) int  — jockey indices, 0 padded
      track_dist_arr:  (R,)       int
      agf_z_mat:       (R, K_max) float — within-race z-score, 0 padded
      valid_mask:      (R, K_max) bool
      horse_to_sire:   (n_horses,) int
    """
    import numpy as np

    race_ids = list(frame.orderings.keys())
    R = len(race_ids)
    K_max = max(len(frame.orderings[rid]) for rid in race_ids) if R else 0
    n_horses = len(frame.horse_index)

    horse_idx_mat = np.zeros((R, K_max), dtype="int64")
    jockey_idx_mat = np.zeros((R, K_max), dtype="int64")
    valid_mask = np.zeros((R, K_max), dtype="bool")
    agf_z_mat = np.zeros((R, K_max), dtype="float64")
    kgs_z_mat = np.zeros((R, K_max), dtype="float64")
    s20_z_mat = np.zeros((R, K_max), dtype="float64")
    last6_z_mat = np.zeros((R, K_max), dtype="float64")
    speed_z_mat = np.zeros((R, K_max), dtype="float64")
    workout_z_mat = np.zeros((R, K_max), dtype="float64")
    pace_z_mat = np.zeros((R, K_max), dtype="float64")
    track_dist_arr = np.zeros(R, dtype="int64")

    has_jockey = len(frame.jockey_of_horse_in_race) > 0
    has_agf = len(frame.agf_of_horse_in_race) > 0
    has_track_dist = len(frame.track_dist_of_race) > 0
    has_kgs = len(frame.kgs_of_horse_in_race) > 0
    has_s20 = len(frame.s20_of_horse_in_race) > 0
    has_last6 = len(frame.last6_of_horse_in_race) > 0
    has_speed = len(frame.speed_of_horse_in_race) > 0
    has_workout = len(frame.workout_of_horse_in_race) > 0
    has_pace = len(frame.pace_of_horse_in_race) > 0

    from ganyan.predictor.bayes.standardize import zscore_missing as _zscore

    flat_idx = 0
    for r, rid in enumerate(race_ids):
        order = frame.orderings[rid]
        n = len(order)
        horse_idx_mat[r, :n] = order
        if has_jockey:
            jockey_idx_mat[r, :n] = frame.jockey_of_horse_in_race[
                flat_idx : flat_idx + n
            ]
        if has_track_dist:
            track_dist_arr[r] = frame.track_dist_of_race[rid]
        if has_agf:
            agf_z_mat[r, :n] = _zscore(np.asarray(
                frame.agf_of_horse_in_race[flat_idx : flat_idx + n],
                dtype="float64",
            ))
        if has_kgs:
            kgs_z_mat[r, :n] = _zscore(np.asarray(
                frame.kgs_of_horse_in_race[flat_idx : flat_idx + n],
                dtype="float64",
            ))
        if has_s20:
            s20_z_mat[r, :n] = _zscore(np.asarray(
                frame.s20_of_horse_in_race[flat_idx : flat_idx + n],
                dtype="float64",
            ))
        if has_last6:
            last6_z_mat[r, :n] = _zscore(np.asarray(
                frame.last6_of_horse_in_race[flat_idx : flat_idx + n],
                dtype="float64",
            ))
        if has_speed:
            speed_z_mat[r, :n] = _zscore(np.asarray(
                frame.speed_of_horse_in_race[flat_idx : flat_idx + n],
                dtype="float64",
            ))
        if has_workout:
            workout_z_mat[r, :n] = _zscore(np.asarray(
                frame.workout_of_horse_in_race[flat_idx : flat_idx + n],
                dtype="float64",
            ))
        if has_pace:
            pace_z_mat[r, :n] = _zscore(np.asarray(
                frame.pace_of_horse_in_race[flat_idx : flat_idx + n],
                dtype="float64",
            ))
        valid_mask[r, :n] = True
        flat_idx += n

    horse_to_sire = np.zeros(n_horses, dtype="int64")
    seen: set[int] = set()
    flat_horses = [h for order in frame.orderings.values() for h in order]
    for h, s in zip(flat_horses, frame.sire_of_horse_in_race):
        if h not in seen:
            horse_to_sire[h] = s
            seen.add(h)

    return {
        "horse_idx_mat": horse_idx_mat,
        "jockey_idx_mat": jockey_idx_mat,
        "track_dist_arr": track_dist_arr,
        "agf_z_mat": agf_z_mat,
        "kgs_z_mat": kgs_z_mat,
        "s20_z_mat": s20_z_mat,
        "last6_z_mat": last6_z_mat,
        "speed_z_mat": speed_z_mat,
        "workout_z_mat": workout_z_mat,
        "pace_z_mat": pace_z_mat,
        "valid_mask": valid_mask,
        "horse_to_sire": horse_to_sire,
    }


def _intern(idx: Dict, key) -> int:
    if key not in idx:
        idx[key] = len(idx)
    return idx[key]


def build_training_frame(
    session: Session,
    from_date: date,
    to_date: date,
    min_field_size: int = 3,
    include_speed: bool = True,
    include_workouts: bool = True,
    include_pace: bool = True,
) -> TrainingFrame:
    """Build a per-race ordered frame for PL training.

    When ``include_speed`` is True (default), pre-computes track-variant
    adjusted seconds-per-meter per horse using the entire DB history up
    through ``to_date`` (variants computed from winners only) and feeds
    a recency-mean per entry as the ``speed_of_horse_in_race`` signal.

    When ``include_workouts`` is True, pre-computes recent-workouts
    sec-per-meter per horse from the ``tjk_workouts`` external_signals.
    Returns NaN (mapped to within-race mean after scaling) for horses without
    prior recorded workouts — useful since workout coverage was tiny
    as of 2026-04-30 (only 4 days of ingestion).
    """
    frame = TrainingFrame()
    _intern(frame.sire_index, "")

    from datetime import timedelta
    from ganyan.predictor.speed_figures import build_horse_speed_history, compute_track_variants
    from ganyan.predictor.pace import build_horse_pace_history, compute_pace_baseline
    from ganyan.predictor.workouts import build_horse_workout_history
    from ganyan.time import race_cutoff
    history_cache = {}

    races = session.execute(
        select(Race).where(
            Race.status == RaceStatus.resulted,
            Race.date >= from_date,
            Race.date <= to_date,
        ).order_by(Race.date, Race.race_number)
    ).scalars().all()

    for r in races:
        prior_day = r.date - timedelta(days=1)
        if prior_day not in history_cache:
            speeds = build_horse_speed_history(session, compute_track_variants(session, to_date=prior_day), to_date=prior_day) if include_speed else None
            paces = build_horse_pace_history(session, compute_pace_baseline(session, to_date=prior_day), to_date=prior_day) if include_pace else None
            history_cache[prior_day] = speeds, paces
        speed_history, pace_history = history_cache[prior_day]
        workout_history = build_horse_workout_history(session, to_date=r.date, as_of=race_cutoff(r)) if include_workouts else None
        finishers = [e for e in r.entries if not e.scratched]
        if (len(finishers) < min_field_size
                or any(e.finish_position is None for e in finishers)
                or not any(e.finish_position == 1 for e in finishers)):
            continue
        # Placing 0 marks a non-finisher: order it behind every finisher.
        finishers.sort(key=lambda e: (e.finish_position < 1, e.finish_position))
        horse_ids: List[int] = []
        jockey_ids: List[int] = []
        sire_ids: List[int] = []
        agfs: List[float] = []
        kgss: List[float] = []
        s20s: List[float] = []
        last6s: List[float] = []
        speeds: List[float] = []
        workouts: List[float] = []
        paces: List[float] = []
        for e in finishers:
            horse_ids.append(_intern(frame.horse_index, e.horse_id))
            jockey_ids.append(_intern(frame.jockey_index, e.jockey or ""))
            sire_name = (e.horse.sire or "") if e.horse else ""
            sire_ids.append(_intern(frame.sire_index, sire_name))
            agfs.append(float(e.agf) if e.agf is not None else float("nan"))
            kgss.append(float(e.kgs) if e.kgs is not None else float("nan"))
            s20s.append(float(e.s20) if e.s20 is not None else float("nan"))
            last6s.append(summarize_last_six(e.last_six))
            if speed_history is not None:
                from ganyan.predictor.speed_figures import horse_speed_score
                spm = horse_speed_score(speed_history, e.horse_id, r.date)
                speeds.append(spm if spm is not None else float("nan"))
            else:
                speeds.append(float("nan"))
            if workout_history is not None:
                from ganyan.predictor.workouts import horse_workout_score
                w = horse_workout_score(workout_history, e.horse_id, r.date, as_of=race_cutoff(r))
                workouts.append(w if w is not None else float("nan"))
            else:
                workouts.append(float("nan"))
            if pace_history is not None:
                from ganyan.predictor.pace import horse_pace_score
                p = horse_pace_score(pace_history, e.horse_id, r.date)
                paces.append(p if p is not None else float("nan"))
            else:
                paces.append(float("nan"))
        track_dist = (r.track_id, distance_bucket_for(r.distance_meters or 0))
        frame.track_dist_of_race[r.id] = _intern(frame.track_dist_index, track_dist)
        frame.orderings[r.id] = horse_ids
        frame.jockey_of_horse_in_race.extend(jockey_ids)
        frame.sire_of_horse_in_race.extend(sire_ids)
        frame.agf_of_horse_in_race.extend(agfs)
        frame.kgs_of_horse_in_race.extend(kgss)
        frame.s20_of_horse_in_race.extend(s20s)
        frame.last6_of_horse_in_race.extend(last6s)
        frame.speed_of_horse_in_race.extend(speeds)
        frame.workout_of_horse_in_race.extend(workouts)
        frame.pace_of_horse_in_race.extend(paces)

    return frame
