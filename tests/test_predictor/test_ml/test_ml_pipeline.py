"""End-to-end tests for the ML ranker pipeline.

Covers:
- Feature builder returns the expected shape and target.
- Trainer fits a small model, saves artefacts, reports sensible metrics.
- MLPredictor loads the trained model and returns well-formed predictions.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import (
    Base, Horse, Race, RaceEntry, RaceStatus, Track,
)
from ganyan.predictor.ml import (
    FEATURE_COLUMNS, MLPredictor, build_training_frame, train_ranker,
)
from ganyan.predictor.ml.features import build_race_frame
from ganyan.predictor.ml.predictor import load_latest_model


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _seed_race(
    session, *, race_date, race_number, track_name, entries, surface="kum",
    distance=1400,
):
    """Seed a resulted race with the given (name, agf, finish_pos) tuples."""
    track = session.query(Track).filter_by(name=track_name).first()
    if track is None:
        track = Track(name=track_name)
        session.add(track)
        session.flush()
    race = Race(
        track_id=track.id, date=race_date, race_number=race_number,
        distance_meters=distance, surface=surface, status=RaceStatus.resulted,
    )
    session.add(race)
    session.flush()
    for i, (name, agf, finish_pos) in enumerate(entries, start=1):
        horse = session.query(Horse).filter_by(name=name).first()
        if horse is None:
            horse = Horse(name=name, age=4, trainer=f"Tr{i}")
            session.add(horse)
            session.flush()
        session.add(RaceEntry(
            race_id=race.id, horse_id=horse.id,
            gate_number=i, jockey=f"J{i}",
            agf=agf, hp=80.0 + i, weight_kg=57.0,
            finish_position=finish_pos,
        ))
    session.flush()
    return race


def _seed_many(session, n_races: int = 25):
    """Create enough races that the trainer has a meaningful train/test split.

    Each race has 6 horses; the horse with the highest AGF wins ~70% of
    the time (deterministic pattern based on race number parity) so the
    model has real signal to learn.
    """
    horse_pool = [f"H{k}" for k in range(1, 13)]
    for r in range(n_races):
        # Sample six distinct horses for each race.
        offset = r % 7
        horses_this_race = horse_pool[offset:offset + 6]
        # AGF sums to 100.
        agfs = [40, 25, 15, 10, 6, 4]
        if r % 3 == 0:
            # Favourite wins.
            finish = [1, 2, 3, 4, 5, 6]
        elif r % 3 == 1:
            # Second favourite wins.
            finish = [2, 1, 3, 4, 5, 6]
        else:
            # Third wins (upset).
            finish = [3, 2, 1, 4, 5, 6]
        _seed_race(
            session,
            race_date=date(2026, 3, 1 + (r // 10)).replace(day=1 + (r % 10)),
            race_number=r + 1,
            track_name="TestTrack",
            entries=list(zip(horses_this_race, agfs, finish)),
        )
    session.commit()


def test_feature_columns_are_stable():
    """Adding or renaming columns should be a conscious decision."""
    expected_prefix = [
        "speed_figure", "form_cycle", "weight_delta", "rest_fitness",
        "class_indicator", "jockey_win_rate", "trainer_win_rate",
        "gate_bias", "surface_affinity", "agf_edge",
    ]
    assert FEATURE_COLUMNS[: len(expected_prefix)] == expected_prefix


def test_build_training_frame_shape(db_session):
    _seed_many(db_session, n_races=5)
    frame = build_training_frame(db_session)
    assert not frame.features.empty
    assert list(frame.features.columns) == FEATURE_COLUMNS
    # 5 races × 6 horses = 30 rows.
    assert len(frame.features) == 30
    assert len(frame.target) == 30
    assert frame.groups.nunique() == 5


def test_rank_score_target_is_inverse_of_finish_position(db_session):
    _seed_many(db_session, n_races=2)
    frame = build_training_frame(db_session)
    # Per race, winners must have the highest target.
    for _race_id, idx in frame.groups.groupby(frame.groups).groups.items():
        targets = frame.target.loc[list(idx)]
        assert targets.max() == 5  # field_size 6 - finish 1


def test_build_race_frame_for_inference(db_session):
    _seed_many(db_session, n_races=3)
    race = db_session.query(Race).first()
    df = build_race_frame(db_session, race.id)
    assert len(df) == 6
    assert "horse_id" in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_train_ranker_end_to_end(db_session, tmp_path: Path):
    _seed_many(db_session, n_races=30)
    result = train_ranker(
        db_session,
        holdout_fraction=0.2,
        num_boost_round=50,
        model_dir=tmp_path,
        model_name="test_ranker",
    )
    assert result.model_path.exists()
    assert result.metadata_path.exists()
    assert result.train_races > 0
    assert "top1_accuracy" in result.metrics
    # With the deterministic seed pattern, favourite wins 33% of the time
    # and the ranker should at least match that.
    assert result.metrics["top1_accuracy"] >= 25.0


def test_ml_predictor_round_trip(db_session, tmp_path: Path, monkeypatch):
    _seed_many(db_session, n_races=30)
    result = train_ranker(
        db_session,
        holdout_fraction=0.2,
        num_boost_round=50,
        model_dir=tmp_path,
        model_name="test_ranker",
    )
    # Point the loader at our temp dir.
    monkeypatch.setattr(
        "ganyan.predictor.ml.predictor.DEFAULT_MODEL_DIR", tmp_path,
    )
    monkeypatch.setattr(
        "ganyan.predictor.ml.predictor.DEFAULT_MODEL_BASENAME", "test_ranker",
    )

    loaded = load_latest_model()
    assert loaded.feature_columns == FEATURE_COLUMNS
    assert loaded.booster.num_feature() == len(FEATURE_COLUMNS)

    predictor = MLPredictor(db_session, model=loaded)
    race = db_session.query(Race).first()
    preds = predictor.predict(race.id)
    assert len(preds) == 6
    total_prob = sum(p.probability for p in preds)
    assert total_prob == pytest.approx(100.0, abs=1e-3)
    # Predictions sorted descending.
    probs = [p.probability for p in preds]
    assert probs == sorted(probs, reverse=True)


def test_ml_predictor_persists_audit_row(db_session, tmp_path: Path, monkeypatch):
    from ganyan.db.models import Prediction as PredictionRow

    _seed_many(db_session, n_races=30)
    train_ranker(
        db_session,
        holdout_fraction=0.2,
        num_boost_round=30,
        model_dir=tmp_path,
        model_name="test_ranker",
    )
    monkeypatch.setattr(
        "ganyan.predictor.ml.predictor.DEFAULT_MODEL_DIR", tmp_path,
    )
    monkeypatch.setattr(
        "ganyan.predictor.ml.predictor.DEFAULT_MODEL_BASENAME", "test_ranker",
    )

    predictor = MLPredictor(db_session)
    race = db_session.query(Race).first()
    before = db_session.query(PredictionRow).count()
    predictor.predict_and_save(race.id)
    db_session.commit()
    after = db_session.query(PredictionRow).count()
    assert after - before == 6
    versions = {row.model_version for row in db_session.query(PredictionRow).all()}
    assert any(v.startswith("lightgbm-lambdarank") for v in versions)
