"""Regression test for compute_workout_signals' temporal anchor.

History (2026-05-02): days_since_workout was computed as
``date.today() - workout_date`` regardless of when the race actually
ran. For a 2025-06-15 race with a workout 5 days earlier, training
saw days_since ≈ 320 instead of 5. The model trained on garbage and
predicted on real values — train/predict divergence on this column.
"""
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ganyan.db.models import (
    Base,
    ExternalSignal,
    Horse,
    Race,
    RaceEntry,
    RaceStatus,
    Track,
)
from ganyan.predictor.features import compute_workout_signals


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()
    engine.dispose()


def _seed_entry_with_workout(s, *, race_date: date, workout_date: date):
    track = Track(name="İstanbul", city="İstanbul")
    s.add(track)
    s.flush()
    race = Race(
        track_id=track.id, date=race_date, race_number=1,
        distance_meters=1400, surface="çim", status=RaceStatus.scheduled,
    )
    s.add(race)
    s.flush()
    horse = Horse(name="TestHorse", age=4)
    s.add(horse)
    s.flush()
    entry = RaceEntry(
        race_id=race.id, horse_id=horse.id, gate_number=1,
        jockey="Test Jockey", weight_kg=57.0,
    )
    s.add(entry)
    s.flush()

    s.add(ExternalSignal(
        source_name="tjk_workouts",
        signal_type="workout_split",
        race_entry_id=entry.id,
        race_id=race.id,
        value=12.0,
        payload={
            "workout_date": workout_date.isoformat(),
            "distance_m": 1000,
        },
        captured_at=datetime(workout_date.year, workout_date.month, workout_date.day, 7, 0, 0),
    ))
    s.commit()
    return entry


def test_days_since_workout_anchored_to_race_date(session):
    """For a historical race, days_since must reflect race_date - workout_date,
    NOT today - workout_date.
    """
    entry = _seed_entry_with_workout(
        session,
        race_date=date(2025, 6, 15),
        workout_date=date(2025, 6, 10),   # 5 days before race
    )

    days_since, _, _ = compute_workout_signals(
        session, entry.id, race_distance_m=1400, race_date=date(2025, 6, 15),
    )

    assert days_since == 5.0, (
        f"days_since={days_since}, expected 5.0. "
        "Feature is anchored to today() instead of race_date — "
        "training data on historical races is being corrupted."
    )


def test_same_day_unverified_workout_is_excluded(session):
    """Live predict callers that don't yet pass race_date keep working."""
    today = date.today()
    entry = _seed_entry_with_workout(
        session,
        race_date=today,
        workout_date=today,           # same day so days_since = 0 either way
    )

    days_since_explicit, _, _ = compute_workout_signals(
        session, entry.id, race_distance_m=1400, race_date=today,
    )
    days_since_default, _, _ = compute_workout_signals(
        session, entry.id, race_distance_m=1400,
    )

    assert days_since_explicit is None and days_since_default is None
