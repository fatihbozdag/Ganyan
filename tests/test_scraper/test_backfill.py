import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Track, Race, Horse, RaceEntry, ScrapeLog, RaceStatus, ScrapeStatus
from ganyan.scraper.parser import RawRaceCard, RawHorseEntry, parse_race_card
from ganyan.scraper.backfill import (
    store_race_card,
    update_race_results,
    get_or_create_track,
    get_or_create_horse,
    get_scraped_dates,
    log_scrape,
    BackfillManager,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _make_raw_card(track="İstanbul", race_num=1, horse_name="Karayel"):
    return RawRaceCard(
        track_name=track,
        date=date(2026, 4, 5),
        race_number=race_num,
        distance_meters=1400,
        surface="Çim",
        race_type="Handikap",
        horses=[
            RawHorseEntry(
                name=horse_name, age=4, origin="TR",
                gate_number=1, jockey="Ahmet Çelik", weight_kg=57.0,
                hp=85.5, kgs=21, s20=12.5, eid="1.30.45",
                gny=8.3, agf=5.2, last_six="1 3 2 4 1 2",
            ),
        ],
    )


# --- store_race_card ---

def test_store_race_card_creates_all_records(db_session):
    raw = _make_raw_card()
    parsed = parse_race_card(raw)
    store_race_card(db_session, parsed)
    db_session.commit()

    tracks = db_session.query(Track).all()
    assert len(tracks) == 1
    assert tracks[0].name == "İstanbul"

    races = db_session.query(Race).all()
    assert len(races) == 1
    assert races[0].race_number == 1

    horses = db_session.query(Horse).all()
    assert len(horses) == 1
    assert horses[0].name == "Karayel"

    entries = db_session.query(RaceEntry).all()
    assert len(entries) == 1
    assert float(entries[0].hp) == 85.5


def test_store_race_card_is_idempotent(db_session):
    raw = _make_raw_card()
    parsed = parse_race_card(raw)
    store_race_card(db_session, parsed)
    db_session.commit()
    store_race_card(db_session, parsed)
    db_session.commit()

    assert db_session.query(Track).count() == 1
    assert db_session.query(Race).count() == 1
    assert db_session.query(Horse).count() == 1
    assert db_session.query(RaceEntry).count() == 1


def test_store_race_card_reuses_existing_horse(db_session):
    raw1 = _make_raw_card(race_num=1, horse_name="Karayel")
    raw2 = _make_raw_card(race_num=2, horse_name="Karayel")
    store_race_card(db_session, parse_race_card(raw1))
    store_race_card(db_session, parse_race_card(raw2))
    db_session.commit()

    assert db_session.query(Horse).count() == 1
    assert db_session.query(RaceEntry).count() == 2


# --- get_or_create_track ---

def test_get_or_create_track_creates_new(db_session):
    track = get_or_create_track(db_session, "Ankara")
    db_session.flush()
    assert track.name == "Ankara"
    assert track.id is not None


def test_get_or_create_track_returns_existing(db_session):
    t1 = get_or_create_track(db_session, "İstanbul")
    db_session.flush()
    t2 = get_or_create_track(db_session, "İstanbul")
    assert t1.id == t2.id
    assert db_session.query(Track).count() == 1


# --- get_or_create_horse ---

def test_get_or_create_horse_creates_new(db_session):
    horse = get_or_create_horse(db_session, "Karayel", age=4, origin="TR")
    db_session.flush()
    assert horse.name == "Karayel"
    assert horse.age == 4


def test_get_or_create_horse_updates_mutable_fields(db_session):
    h1 = get_or_create_horse(db_session, "Karayel", age=4, trainer="Mehmet")
    db_session.flush()
    h2 = get_or_create_horse(db_session, "Karayel", age=5, trainer="Ali")
    db_session.flush()
    assert h1.id == h2.id
    assert h2.age == 5
    assert h2.trainer == "Ali"
    assert db_session.query(Horse).count() == 1


# --- update_race_results ---

def test_update_race_results_sets_positions(db_session):
    raw = _make_raw_card()
    parsed = parse_race_card(raw)
    store_race_card(db_session, parsed)
    db_session.commit()

    # Build a results card with finish data
    raw_result = RawRaceCard(
        track_name="İstanbul",
        date=date(2026, 4, 5),
        race_number=1,
        distance_meters=1400,
        surface="Çim",
        race_type="Handikap",
        horses=[
            RawHorseEntry(
                name="Karayel", age=4, origin="TR",
                gate_number=1, jockey="Ahmet Çelik", weight_kg=57.0,
                hp=85.5, finish_position=1, finish_time="1.30.45",
            ),
        ],
    )
    parsed_result = parse_race_card(raw_result)
    race = update_race_results(db_session, parsed_result)
    db_session.commit()

    assert race is not None
    assert race.status == RaceStatus.resulted
    entry = db_session.query(RaceEntry).first()
    assert entry.finish_position == 1
    assert entry.finish_time == "1.30.45"


def test_update_race_results_returns_none_for_missing_race(db_session):
    raw = _make_raw_card(track="Ankara", race_num=99)
    parsed = parse_race_card(raw)
    result = update_race_results(db_session, parsed)
    assert result is None


# --- get_scraped_dates ---

def test_get_scraped_dates(db_session):
    # Fully scraped day: ALL sentinel with success.
    db_session.add(ScrapeLog(date=date(2026, 4, 5), track="ALL", status=ScrapeStatus.success))
    db_session.commit()

    scraped = get_scraped_dates(db_session)
    assert date(2026, 4, 5) in scraped


def test_get_scraped_dates_excludes_partial_success(db_session):
    """A date with per-track success rows but no ALL completion marker
    is considered only partially scraped and should be retried."""
    db_session.add(ScrapeLog(date=date(2026, 4, 5), track="İstanbul", status=ScrapeStatus.success))
    # Missing ALL marker → date is NOT fully scraped.
    db_session.add(ScrapeLog(date=date(2026, 4, 6), track="ALL", status=ScrapeStatus.success))
    db_session.commit()

    scraped = get_scraped_dates(db_session)
    assert date(2026, 4, 5) not in scraped
    assert date(2026, 4, 6) in scraped


def test_get_scraped_dates_excludes_failed(db_session):
    db_session.add(ScrapeLog(date=date(2026, 4, 5), track="ALL", status=ScrapeStatus.failed))
    db_session.add(ScrapeLog(date=date(2026, 4, 6), track="ALL", status=ScrapeStatus.success))
    db_session.commit()

    scraped = get_scraped_dates(db_session)
    assert date(2026, 4, 5) not in scraped
    assert date(2026, 4, 6) in scraped


def test_log_scrape_captures_error_message(db_session):
    log_scrape(
        db_session,
        date(2026, 4, 5),
        "İstanbul",
        ScrapeStatus.failed,
        error_message="HTTP 503 Service Unavailable",
    )
    db_session.commit()
    row = db_session.query(ScrapeLog).first()
    assert row.error_message == "HTTP 503 Service Unavailable"


# --- log_scrape ---

def test_log_scrape(db_session):
    log_scrape(db_session, date(2026, 4, 5), "İstanbul", ScrapeStatus.success)
    db_session.commit()

    logs = db_session.query(ScrapeLog).all()
    assert len(logs) == 1
    assert logs[0].track == "İstanbul"
    assert logs[0].status == ScrapeStatus.success


# --- BackfillManager ---

@pytest.mark.asyncio
async def test_backfill_skips_scraped_dates(db_session):
    """BackfillManager should skip dates that already have an ALL-success marker."""
    # Mark 2026-04-05 as fully scraped via the ALL completion marker.
    db_session.add(ScrapeLog(date=date(2026, 4, 5), track="ALL", status=ScrapeStatus.success))
    db_session.commit()

    scraped_dates_called = []

    class FakeTJKClient:
        async def get_race_card(self, race_date):
            scraped_dates_called.append(race_date)
            return [_make_raw_card(race_num=1)]

        async def close(self):
            pass

    mgr = BackfillManager(db_session, FakeTJKClient())
    await mgr.backfill(from_date=date(2026, 4, 4), to_date=date(2026, 4, 5))

    # 2026-04-05 was already scraped so only 2026-04-04 should be fetched
    assert date(2026, 4, 5) not in scraped_dates_called
    assert date(2026, 4, 4) in scraped_dates_called


@pytest.mark.asyncio
async def test_backfill_processes_reverse_chronological(db_session):
    """BackfillManager should process dates from newest to oldest."""
    order = []

    class FakeTJKClient:
        async def get_race_card(self, race_date):
            order.append(race_date)
            return [_make_raw_card(race_num=1)]

        async def close(self):
            pass

    mgr = BackfillManager(db_session, FakeTJKClient())
    await mgr.backfill(from_date=date(2026, 4, 1), to_date=date(2026, 4, 3))

    assert order == [date(2026, 4, 3), date(2026, 4, 2), date(2026, 4, 1)]
