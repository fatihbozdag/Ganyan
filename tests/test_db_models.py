import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Track, Race, Horse, RaceEntry, ScrapeLog, RaceStatus, ScrapeStatus


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_create_track(db_session):
    track = Track(name="İstanbul", city="İstanbul", surface_types=["çim", "kum"])
    db_session.add(track)
    db_session.commit()
    assert track.id is not None
    assert track.name == "İstanbul"


def test_create_race_with_track(db_session):
    track = Track(name="Ankara", city="Ankara", surface_types=["sentetik"])
    db_session.add(track)
    db_session.flush()

    race = Race(
        track_id=track.id,
        date=date(2026, 4, 5),
        race_number=1,
        distance_meters=1200,
        surface="sentetik",
        status=RaceStatus.scheduled,
    )
    db_session.add(race)
    db_session.commit()
    assert race.id is not None
    assert race.track.name == "Ankara"


def test_create_horse_and_entry(db_session):
    track = Track(name="İzmir", city="İzmir", surface_types=["çim"])
    db_session.add(track)
    db_session.flush()

    race = Race(
        track_id=track.id,
        date=date(2026, 4, 5),
        race_number=2,
        distance_meters=1400,
        surface="çim",
        status=RaceStatus.scheduled,
    )
    db_session.add(race)
    db_session.flush()

    horse = Horse(name="Karayel", age=4, origin="TR")
    db_session.add(horse)
    db_session.flush()

    entry = RaceEntry(
        race_id=race.id,
        horse_id=horse.id,
        gate_number=3,
        jockey="Ahmet Çelik",
        weight_kg=57.0,
        hp=85.5,
        kgs=21,
        last_six="1 3 2 4 1 2",
    )
    db_session.add(entry)
    db_session.commit()

    assert entry.id is not None
    assert entry.horse.name == "Karayel"
    assert entry.race.race_number == 2
    assert entry.finish_position is None  # pre-race


def test_race_unique_constraint(db_session):
    track = Track(name="Bursa", city="Bursa", surface_types=["çim"])
    db_session.add(track)
    db_session.flush()

    race1 = Race(
        track_id=track.id, date=date(2026, 4, 5), race_number=1,
        distance_meters=1200, surface="çim", status=RaceStatus.scheduled,
    )
    race2 = Race(
        track_id=track.id, date=date(2026, 4, 5), race_number=1,
        distance_meters=1600, surface="çim", status=RaceStatus.scheduled,
    )
    db_session.add(race1)
    db_session.commit()
    db_session.add(race2)
    with pytest.raises(Exception):  # IntegrityError
        db_session.commit()


def test_scrape_log(db_session):
    log = ScrapeLog(
        date=date(2026, 4, 5),
        track="İstanbul",
        status=ScrapeStatus.success,
    )
    db_session.add(log)
    db_session.commit()
    assert log.id is not None
    assert log.scraped_at is not None
