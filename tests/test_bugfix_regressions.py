"""Regressions for crashes introduced in the 2026-09-18 audit commit.

- workouts.horse_workout_score imported a non-existent ``datetimetime``.
- cli/main.py and web/routes.py used ``timedelta`` without importing it
  (``ganyan advice`` crashed; ``/ops/health`` raised; the web advice page
  silently disabled the Bayes gate via a swallowed NameError).
- scheduler repredict aborted the whole loop on the first failing race.
"""
from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ganyan.db.models import Base, Horse, Race, RaceEntry, RaceStatus, Track
from ganyan.predictor.workouts import horse_workout_score


def test_horse_workout_score_with_history():
    history = {
        7: [
            (date(2025, 5, 1), 0.070, datetime(2025, 5, 1, 9, 0)),
            (date(2025, 5, 8), 0.068, datetime(2025, 5, 8, 9, 0)),
            (date(2025, 6, 2), 0.060, datetime(2025, 6, 2, 9, 0)),  # after race
        ]
    }
    score = horse_workout_score(history, 7, date(2025, 5, 20))
    assert score == pytest.approx((0.070 + 0.068) / 2)
    assert horse_workout_score(history, 99, date(2025, 5, 20)) is None


def test_web_and_cli_modules_define_timedelta():
    import ganyan.cli.main as cli_main
    import ganyan.web.routes as routes

    assert hasattr(cli_main, "timedelta")
    assert hasattr(routes, "timedelta")


def test_ops_health_does_not_crash():
    from ganyan.web.app import create_app

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    app = create_app(
        session_factory=sessionmaker(bind=engine),
        refresh_on_launch=False,
        enable_scheduler=False,
    )
    app.config["TESTING"] = True
    resp = app.test_client().get("/ops/health")
    assert resp.status_code in (200, 503)
    assert "status" in resp.get_json()
    engine.dispose()


def test_repredict_continues_after_failed_race_then_raises(monkeypatch):
    import ganyan.db
    import ganyan.predictor.ml.ensemble as ensemble_mod
    import ganyan.predictor.picks as picks_mod
    from ganyan import scheduler
    from ganyan.config import Settings

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    today = date(2026, 9, 20)
    with factory() as s:
        track = Track(name="İstanbul", city="İstanbul")
        s.add(track)
        s.flush()
        race_ids = []
        for n in (1, 2, 3):
            race = Race(track_id=track.id, date=today, race_number=n,
                        distance_meters=1400, surface="çim",
                        status=RaceStatus.scheduled)
            s.add(race)
            s.flush()
            race_ids.append(race.id)
            for g in (1, 2, 3):
                horse = Horse(name=f"H{n}{g}", age=4)
                s.add(horse)
                s.flush()
                s.add(RaceEntry(race_id=race.id, horse_id=horse.id,
                                gate_number=g, jockey="J", weight_kg=57.0))
        s.commit()

    attempted = []

    class StubPredictor:
        def __init__(self, session):
            pass

        def predict_and_save(self, race_id):
            attempted.append(race_id)
            if race_id == race_ids[0]:
                raise ValueError("boom")

    monkeypatch.setattr(ganyan.db, "get_session", lambda url: factory())
    monkeypatch.setattr(ensemble_mod, "EnsemblePredictor", StubPredictor)
    monkeypatch.setattr(picks_mod, "generate_picks_for_race",
                        lambda session, race_id, refresh: [])
    monkeypatch.setattr(scheduler, "race_today", lambda: today)
    monkeypatch.setattr(scheduler, "is_upcoming",
                        lambda race, margin_minutes: True)

    with pytest.raises(RuntimeError, match="1 race"):
        scheduler._job_repredict_upcoming(Settings(database_url="sqlite://"))
    assert sorted(attempted) == sorted(race_ids)
    engine.dispose()
