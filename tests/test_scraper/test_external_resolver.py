# tests/test_scraper/test_external_resolver.py
"""Tests for external_signals binders."""
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
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()
    engine.dispose()


def _seed_race(s, *, race_date: date, jockey: str, track_name: str = "Bursa"):
    track = (
        s.query(Track).filter_by(name=track_name).one_or_none()
        or Track(name=track_name, city=track_name)
    )
    s.add(track)
    s.flush()
    race = Race(
        track_id=track.id, date=race_date, race_number=1,
        distance_meters=1400, surface="çim", status=RaceStatus.scheduled,
    )
    s.add(race)
    s.flush()
    horse = Horse(name=f"Horse_{race_date}_{jockey}", age=4)
    s.add(horse)
    s.flush()
    entry = RaceEntry(
        race_id=race.id, horse_id=horse.id, gate_number=1,
        jockey=jockey, weight_kg=57.0,
    )
    s.add(entry)
    s.flush()
    return race, entry


def test_discipline_binds_to_race_within_window(session):
    """Discipline signal must bind to entries on dates in [start_date, end_date]."""
    from ganyan.scraper.external.resolver import bind_discipline_to_entries

    race, entry = _seed_race(
        session, race_date=date(2026, 4, 28), jockey="ERTUĞRUL CANKILIÇ",
    )
    sig = ExternalSignal(
        source_name="tjk_discipline",
        signal_type="penalized_jockey",
        race_id=None,
        race_entry_id=None,
        value=1.0,
        payload={
            "jockey_name": "ERTUĞRUL CANKILIÇ",
            "start_date": "2026-04-19",
            "end_date": "2026-05-03",
            "for_date": "2026-04-26",
        },
        captured_at=datetime(2026, 4, 26, 1, 0, 0),
    )
    session.add(sig)
    session.commit()

    # Bind invocation date (today) is irrelevant — the race is in window.
    n = bind_discipline_to_entries(session, target_date=date(2026, 4, 30))
    session.commit()

    assert n >= 1
    bound = session.query(ExternalSignal).filter_by(id=sig.id).one()
    assert bound.race_entry_id == entry.id
    assert bound.race_id == race.id


def test_discipline_skips_race_outside_window(session):
    """Race date before/after the discipline window must not bind."""
    from ganyan.scraper.external.resolver import bind_discipline_to_entries

    # Race on 2026-04-15 — before the window starts.
    _seed_race(
        session, race_date=date(2026, 4, 15), jockey="ERTUĞRUL CANKILIÇ",
    )
    sig = ExternalSignal(
        source_name="tjk_discipline",
        signal_type="penalized_jockey",
        payload={
            "jockey_name": "ERTUĞRUL CANKILIÇ",
            "start_date": "2026-04-19",
            "end_date": "2026-05-03",
        },
        captured_at=datetime(2026, 4, 26, 1, 0, 0),
    )
    session.add(sig)
    session.commit()

    bind_discipline_to_entries(session, target_date=date(2026, 4, 30))
    session.commit()

    bound = session.query(ExternalSignal).filter_by(id=sig.id).one()
    assert bound.race_entry_id is None
