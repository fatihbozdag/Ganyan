# tests/test_web/test_routes.py
import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ganyan.db.models import Base, Track, Race, Horse, RaceEntry, RaceStatus
from ganyan.web.app import create_app


@pytest.fixture
def app():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    # Seed test data
    with factory() as session:
        track = Track(name="İstanbul", city="İstanbul")
        session.add(track)
        session.flush()
        race = Race(
            track_id=track.id, date=date.today(), race_number=1,
            distance_meters=1400, surface="çim", status=RaceStatus.scheduled,
        )
        session.add(race)
        session.flush()
        horse = Horse(name="Karayel", age=4)
        session.add(horse)
        session.flush()
        entry = RaceEntry(
            race_id=race.id, horse_id=horse.id, gate_number=1,
            jockey="Ahmet Çelik", weight_kg=57.0, hp=85.5, kgs=21,
            eid="1.30.45", last_six="1 3 2 4 1 2",
        )
        session.add(entry)
        session.commit()

    flask_app = create_app(
        session_factory=factory,
        refresh_on_launch=False,
        enable_scheduler=False,
    )
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Ganyan" in response.data.decode()


def test_races_today(client):
    response = client.get(f"/races/{date.today().isoformat()}")
    assert response.status_code == 200


def test_races_json(client):
    response = client.get(
        f"/races/{date.today().isoformat()}",
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_predict_race(client):
    response = client.get("/races/1/predict")
    assert response.status_code == 200


def test_predict_race_json(client):
    response = client.get(
        "/races/1/predict",
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data


def test_predict_nonexistent_race(client):
    response = client.get("/races/999/predict")
    assert response.status_code == 404


def test_history_returns_200(client):
    response = client.get("/history")
    assert response.status_code == 200


def test_history_json(client):
    response = client.get(
        "/history",
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "summary" in data
    assert "evaluations" in data
    assert "races" in data
    assert isinstance(data["summary"]["total_races"], int)


@pytest.fixture
def app_with_results():
    """App with resulted race data and predictions."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    with factory() as session:
        track = Track(name="Ankara", city="Ankara")
        session.add(track)
        session.flush()
        race = Race(
            track_id=track.id, date=date.today(), race_number=1,
            distance_meters=1400, surface="cim", status=RaceStatus.resulted,
        )
        session.add(race)
        session.flush()
        h1 = Horse(name="Winner Horse", age=4)
        h2 = Horse(name="Loser Horse", age=5)
        session.add_all([h1, h2])
        session.flush()
        e1 = RaceEntry(
            race_id=race.id, horse_id=h1.id, gate_number=1,
            jockey="J1", weight_kg=57.0, hp=85.5, kgs=21,
            eid="1.30.45", last_six="1 3 2 4 1 2",
            finish_position=1, predicted_probability=60.0,
        )
        e2 = RaceEntry(
            race_id=race.id, horse_id=h2.id, gate_number=2,
            jockey="J2", weight_kg=56.0, hp=80.0, kgs=14,
            eid="1.31.20", last_six="3 2 1 5 4 3",
            finish_position=2, predicted_probability=40.0,
        )
        session.add_all([e1, e2])
        session.commit()

    flask_app = create_app(
        session_factory=factory,
        refresh_on_launch=False,
        enable_scheduler=False,
    )
    flask_app.config["TESTING"] = True
    return flask_app


def test_history_with_evaluations(app_with_results):
    client = app_with_results.test_client()
    response = client.get(
        "/history",
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["summary"]["total_races"] == 1
    assert data["summary"]["top1_accuracy"] == 100.0
    assert len(data["evaluations"]) == 1
    assert data["evaluations"][0]["winner_name"] == "Winner Horse"
