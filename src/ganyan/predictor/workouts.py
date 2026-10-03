"""Workout-derived features from the ``tjk_workouts`` external_signals
plugin.

Each ``ExternalSignal`` row with ``source_name='tjk_workouts'`` is one
timed split (200m / 400m / 600m / 800m / 1000m / 1200m / 1400m).  Bound
to a ``race_entry_id`` by the resolver (matched by horse name on the
target race date).

We summarise a horse's recent workout activity into a single per-entry
"workout score" (lower seconds-per-meter = faster training pace) so it
can be fed as a within-race z-score offset to the Bayes PL model — same
shape as the Beyer speed feature.

Coverage is the binding constraint as of 2026-04-30: tjk_workouts only
started ingesting 2026-04-26.  Until ingestion accumulates enough
training-window overlap, ``delta_workouts`` will train at ≈0 and this
feature contributes nothing.  See `project_bayes_predictor.md`.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from ganyan.db.models import ExternalSignal, Race, RaceEntry


WorkoutEntry = Tuple[date, float, datetime]  # workout date, seconds/meter, captured UTC


def build_horse_workout_history(
    session: Session,
    to_date: date | None = None, *, as_of=None,
) -> Dict[int, List[WorkoutEntry]]:
    """For each horse_id, sorted list of (workout_date, sec_per_meter).

    Joins external_signals(source_name='tjk_workouts') → race_entries to
    recover horse_id.  Each row already encodes a single timed split:
    ``value`` is workout seconds, ``payload['distance_m']`` is the split
    distance, ``payload['workout_date']`` is when the workout happened.

    Lower sec_per_meter = faster workout = healthier / sharper training.
    """
    q = (
        select(
            RaceEntry.horse_id,
            ExternalSignal.value,
            ExternalSignal.payload, ExternalSignal.captured_at,
        )
        .join(RaceEntry, RaceEntry.id == ExternalSignal.race_entry_id)
        .join(Race, Race.id == RaceEntry.race_id)
        .where(ExternalSignal.source_name == "tjk_workouts")
        .where(ExternalSignal.race_entry_id.is_not(None))
        .where(ExternalSignal.value.is_not(None))
    )
    if to_date is not None:
        q = q.where(Race.date <= to_date)

    if as_of is not None:
        q = q.where(ExternalSignal.captured_at <= as_of)
    history: Dict[int, List[WorkoutEntry]] = defaultdict(list)
    for horse_id, secs, payload, captured_at in session.execute(q):
        if not payload:
            continue
        distance_m = payload.get("distance_m")
        wdate_str = payload.get("workout_date")
        if not distance_m or distance_m <= 0 or secs is None or secs <= 0:
            continue
        try:
            wdate = date.fromisoformat(wdate_str) if wdate_str else None
        except (TypeError, ValueError):
            wdate = None
        if wdate is None:
            continue
        spm = float(secs) / float(distance_m)
        history[horse_id].append((wdate, spm, captured_at))

    for hid in history:
        history[hid].sort(key=lambda t: t[0])
    return history


def horse_workout_score(
    history: Dict[int, List[WorkoutEntry]],
    horse_id: int,
    as_of_date: date,
    n_recent: int = 3, *, as_of=None,
) -> float | None:
    """Mean sec/m over the last ``n_recent`` workouts strictly before
    ``as_of_date``.  None when no prior workouts.  Lower = faster.
    """
    runs = history.get(horse_id)
    if not runs:
        return None
    from datetime import date, datetimetime, time
    from ganyan.time import ISTANBUL
    from datetime import timezone
    cutoff = as_of or datetime.combine(as_of_date, time.min, ISTANBUL).astimezone(timezone.utc).replace(tzinfo=None)
    prior = [row[1] for row in runs if row[0] < as_of_date
             and len(row) >= 3 and row[2] <= cutoff]
    if not prior:
        return None
    take = prior[-n_recent:]
    return sum(take) / len(take)
