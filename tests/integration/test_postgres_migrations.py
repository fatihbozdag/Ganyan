"""Opt-in integration test against a dedicated *_test PostgreSQL database."""
import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session


def test_postgres_migration_roundtrip_and_identity(monkeypatch):
    url = os.environ.get("GANYAN_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set GANYAN_TEST_DATABASE_URL to an dedicated *_test PostgreSQL database")
    parsed = make_url(url)
    assert parsed.get_backend_name() == "postgresql" and parsed.database.endswith("_test")
    engine = create_engine(url)
    from uuid import uuid4
    schema = "audit_" + uuid4().hex
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    isolated_url = parsed.update_query_dict({"options": f"-csearch_path={schema}"}).render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", isolated_url)
    engine.dispose()
    engine = create_engine(isolated_url)
    assert not inspect(engine).get_table_names()
    config = Config("alembic.ini")
    command.upgrade(config, "head")
    from ganyan.db.models import Horse, Track, Race, RaceEntry, RaceStatus
    from datetime import date
    with Session(engine) as session:
        track = Track(name="Migration test")
        a, b = Horse(name="Same", tjk_at_id=101), Horse(name="Same", tjk_at_id=102)
        session.add_all([track, a, b]); session.flush()
        race = Race(track_id=track.id, date=date(2026, 1, 1), race_number=1,
                    status=RaceStatus.scheduled, conditions_history=[{"surface": "kum"}])
        session.add(race); session.flush()
        session.add(RaceEntry(race_id=race.id, horse_id=a.id, gate_number=3, start_gate=5,
                              age_at_race=4, trainer_at_race="T"))
        session.commit()
        session.expire_all()
        entry = session.query(RaceEntry).one()
        assert (entry.gate_number, entry.start_gate, entry.age_at_race) == (3, 5, 4)
        assert session.query(Horse).count() == 2
        session.delete(entry); session.flush()
        session.delete(race); session.flush()
        session.delete(a); session.delete(b); session.delete(track)
        session.commit()
    command.downgrade(config, "-1")
    assert "start_gate" not in {c["name"] for c in inspect(engine).get_columns("race_entries")}
    command.upgrade(config, "head")
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar() == "f3a4b5c6d7e8"
    engine.dispose()
    with create_engine(url).begin() as connection:
        connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
