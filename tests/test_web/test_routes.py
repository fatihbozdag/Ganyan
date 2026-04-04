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

    flask_app = create_app(session_factory=factory)
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
