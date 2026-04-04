import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Track, Race, Horse, RaceEntry, RaceStatus
from ganyan.predictor.bayesian import BayesianPredictor, Prediction


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def race_with_entries(db_session):
    """Create a race with 4 horses."""
    track = Track(name="İstanbul", city="İstanbul")
    db_session.add(track)
    db_session.flush()

    race = Race(
        track_id=track.id, date=date(2026, 4, 5), race_number=1,
        distance_meters=1400, surface="çim", status=RaceStatus.scheduled,
    )
    db_session.add(race)
    db_session.flush()

    horses_data = [
        ("Karayel", 4, 57.0, 85.5, 21, "1.30.45", "1 3 2 4 1 2", 12.5),
        ("Rüzgar", 3, 55.5, 78.0, 14, "1.31.20", "3 2 1 5 4 3", 10.8),
        ("Fırtına", 5, 59.0, 90.0, 35, "1.29.80", "2 1 1 2 3 1", 14.0),
        ("Yıldız", 4, 56.0, 82.0, 7, "1.32.00", "5 6 4 3 5 4", 9.5),
    ]

    for idx, (name, age, weight, hp, kgs, eid, last_six, s20) in enumerate(horses_data):
        horse = Horse(name=name, age=age)
        db_session.add(horse)
        db_session.flush()
        entry = RaceEntry(
            race_id=race.id, horse_id=horse.id,
            gate_number=idx + 1,
            jockey=f"Jokey {name}", weight_kg=weight, hp=hp, kgs=kgs,
            eid=eid, last_six=last_six, s20=s20,
        )
        db_session.add(entry)

    db_session.commit()
    return race.id


def test_predict_returns_predictions_for_all_horses(db_session, race_with_entries):
    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race_with_entries)
    assert len(predictions) == 4
    assert all(isinstance(p, Prediction) for p in predictions)


def test_probabilities_sum_to_100(db_session, race_with_entries):
    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race_with_entries)
    total = sum(p.probability for p in predictions)
    assert abs(total - 100.0) < 0.01


def test_predictions_are_sorted_by_probability(db_session, race_with_entries):
    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race_with_entries)
    probs = [p.probability for p in predictions]
    assert probs == sorted(probs, reverse=True)


def test_predictions_have_contributing_factors(db_session, race_with_entries):
    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race_with_entries)
    for p in predictions:
        assert isinstance(p.contributing_factors, dict)
        assert len(p.contributing_factors) > 0


def test_predictions_have_confidence(db_session, race_with_entries):
    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race_with_entries)
    for p in predictions:
        assert 0.0 <= p.confidence <= 1.0


def test_predict_empty_race(db_session):
    track = Track(name="Ankara", city="Ankara")
    db_session.add(track)
    db_session.flush()
    race = Race(
        track_id=track.id, date=date(2026, 4, 5), race_number=1,
        distance_meters=1200, surface="kum", status=RaceStatus.scheduled,
    )
    db_session.add(race)
    db_session.commit()

    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race.id)
    assert predictions == []
