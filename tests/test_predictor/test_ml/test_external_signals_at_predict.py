"""Regression tests for the predict-time external-signal pipeline.

History (2026-05-02): build_race_frame had been calling extract_features
without passing race_entry_id / race_id_for_signals, so every external-
signal feature (track_conditions, tipster_consensus, late_agf_drift,
workout_*, etc.) silently fell to NaN at inference even when the data
was in external_signals. The training path was correct, so the model
trained with the columns populated and predicted with them all null —
a persistent train/predict divergence.

These tests guard against that asymmetry creeping back in.
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


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()
    engine.dispose()


def _seed_minimal_race(s):
    track = Track(name="Bursa", city="Bursa")
    s.add(track)
    s.flush()
    race = Race(
        track_id=track.id,
        date=date(2026, 5, 2),
        race_number=1, post_time="14:00",
        distance_meters=1400,
        surface="çim",
        status=RaceStatus.scheduled,
    )
    s.add(race)
    s.flush()
    horses_entries = []
    for i in range(1, 5):
        h = Horse(name=f"H{i}", age=4)
        s.add(h)
        s.flush()
        e = RaceEntry(
            race_id=race.id,
            horse_id=h.id,
            gate_number=i,
            jockey=f"Jockey {i}",
            weight_kg=57.0,
        )
        s.add(e)
        s.flush()
        horses_entries.append((h, e))
    return race, horses_entries


def test_build_race_frame_propagates_tipster_consensus(session):
    """Tipster-consensus signal bound to race_entry_id must surface
    as a non-NaN tipster_consensus column for that entry.
    """
    from ganyan.predictor.ml.features import build_race_frame

    race, horses_entries = _seed_minimal_race(session)
    target_entry = horses_entries[0][1]

    session.add(ExternalSignal(
        source_name="yarisrehberi",
        signal_type="tipster_pick",
        race_id=race.id,
        race_entry_id=target_entry.id,
        value=1.0,
        payload={"source": "yarisrehberi", "rank": 1},
        captured_at=datetime(2026, 5, 2, 9, 0, 0),
    ))
    session.commit()

    frame = build_race_frame(session, race.id)

    target_row = frame[frame["horse_id"] == target_entry.horse_id]
    assert not target_row.empty
    assert target_row["tipster_consensus"].notna().all(), (
        "tipster_consensus is NaN at predict-time despite a populated "
        "signal bound to race_entry_id — extract_features is not "
        "receiving race_entry_id."
    )
