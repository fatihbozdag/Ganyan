"""Offline regressions for the September audit's measurement and safety defects."""
from datetime import date, datetime, timedelta
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ganyan.db.models import (Base, Horse, RaceStatus, Pick,
                             AgfSnapshot, MultiRacePool, MultiRacePick)
from ganyan.scraper.parser import ParsedHorseEntry, ParsedRaceCard
from ganyan.scraper.backfill import store_race_card
from ganyan.predictor.ml.features import TrainingFrame, build_race_frame, build_training_frame
from ganyan.predictor.ml.trainer import _evaluate_ranker
from ganyan.predictor.ml.artifacts import (approved_paths, artifact_identity, candidate_directory,
                                         pipeline_digest, promote)
from ganyan.predictor.ml.gate import assert_min_window, validate_window, evaluate_gate
from ganyan.time import race_cutoff


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
    engine.dispose()


def seed(session, *, future=False):
    day = date.today() + timedelta(days=1) if future else date(2026, 1, 1)
    race = store_race_card(session, ParsedRaceCard(
        track_name="Audit", date=day, race_number=1, surface="kum",
        distance_meters=1200, post_time="14:00",
        horses=[ParsedHorseEntry(name=f"H{i}", tjk_at_id=100+i, gate_number=i,
                                 agf=50-i*5) for i in range(1, 5)]))
    session.flush()
    if not future:
        race.status = RaceStatus.resulted
        for i, entry in enumerate(race.entries, 1):
            entry.finish_position = i
    return race


def test_name_cache_cannot_override_authoritative_horse_id(session):
    session.add(Horse(name="H1", tjk_at_id=999))
    session.flush()
    race = seed(session)
    assert race.entries[0].horse.tjk_at_id == 101
    assert session.query(Horse).filter_by(tjk_at_id=999).one().name == "H1"


def test_race_condition_refresh_has_history(session):
    race = seed(session)
    store_race_card(session, ParsedRaceCard(track_name="Audit", date=race.date,
                    race_number=1, surface="çim", distance_meters=1600, post_time="15:00"))
    assert (race.surface, race.distance_meters, race.post_time) == ("çim", 1600, "15:00")
    assert race.conditions_history


def test_late_snapshot_cannot_change_historical_drift(session):
    from ganyan.predictor.features import compute_late_agf_drift
    race = seed(session)
    entry = race.entries[0]
    for hour, value in [(9, 10), (10, 20), (12, 90)]:
        session.add(AgfSnapshot(race_entry_id=entry.id, agf=value,
                               taken_at=datetime(2026, 1, 1, hour)))
    session.flush()
    assert compute_late_agf_drift(session, entry.id) == 10
    assert compute_late_agf_drift(session, entry.id, as_of=datetime(2026, 1, 1, 9, 30)) is None


def test_historical_scrape_does_not_manufacture_pre_race_snapshot(session):
    seed(session)
    assert session.query(AgfSnapshot).count() == 0


def test_sparse_fields_train_and_serve_match_and_scratches_are_excluded(session):
    race = seed(session)
    race.entries[0].hp = 80
    race.entries[-1].scratched = True
    race.entries[-1].hp = 200
    train = build_training_frame(session, min_field_size=3)
    serve = build_race_frame(session, race.id)
    pd.testing.assert_frame_equal(train.features.reset_index(drop=True),
                                  serve.drop(columns="horse_id").astype(float).reset_index(drop=True))
    assert train.features.class_indicator.isna().all()
    assert (serve.field_size == 3).all()
    assert serve.gate_number.isna().all()  # Program number is not a physical draw.


def test_plase_losses_settle_once_pool_is_confirmed(session):
    from ganyan.predictor.picks import grade_race
    race = seed(session)
    race.entries[0].plase_payout_tl = 2
    pick = Pick(race_id=race.id, strategy="plase_top1", combination=[race.entries[-1].horse_id],
                stake_tl=100, ticket_count=1)
    session.add(pick)
    session.flush()
    grade_race(session, race.id)
    assert pick.graded and pick.hit is False
    assert float(pick.net_tl) == -100


def test_exotic_ledger_and_backtest_use_same_ticket_units(session):
    from ganyan.predictor.picks import grade_race
    from ganyan.predictor.exotic_evaluate import evaluate_pool
    race = seed(session)
    race.uclu_payout_tl = 10
    pick = Pick(race_id=race.id, strategy="uclu_top1",
                combination=[e.horse_id for e in race.entries[:3]], stake_tl=100, ticket_count=1)
    session.add(pick)
    session.flush()
    grade_race(session, race.id)
    fake = SimpleNamespace(predict=lambda rid: [SimpleNamespace(horse_id=e.horse_id, probability=p)
                           for e, p in zip(race.entries, [60, 30, 9, 1])])
    result = evaluate_pool(session, "uclu", 1, predictor_factory=lambda s: fake)
    assert float(pick.payout_tl) == result.total_payout_tl == 500
    race.uclu_payout_tl = None
    result = evaluate_pool(session, "uclu", 1, predictor_factory=lambda s: fake)
    assert result.races == 0 and result.total_stake_tl == 0


def test_exotic_charges_only_generated_tickets(session):
    from ganyan.predictor.exotic_evaluate import evaluate_pool
    race = seed(session)
    race.ganyan_payout_tl = 10
    fake = SimpleNamespace(predict=lambda rid: [SimpleNamespace(horse_id=e.horse_id, probability=25)
                                               for e in race.entries])
    result = evaluate_pool(session, "ganyan", 20, predictor_factory=lambda s: fake)
    assert result.total_stake_tl == 400


def test_constant_scores_are_chance_even_with_winner_first():
    frame = TrainingFrame(features=pd.DataFrame({"x": [0.] * 8}),
        target=pd.Series([3, 2, 1, 0] * 2), ev_target=pd.Series([0.] * 8),
        finish_time_target=pd.Series([0.] * 8), groups=pd.Series([1] * 4 + [2] * 4),
        race_dates=pd.Series([date(2026, 1, 1)] * 8))
    metrics = _evaluate_ranker(SimpleNamespace(predict=lambda x: np.zeros(len(x))), frame)
    assert metrics["top1_accuracy"] == 25
    assert metrics["top3_accuracy"] == 75


@pytest.mark.parametrize("n,days", [(1, 365), (1500, 1), (1499, 365)])
def test_gate_checks_scored_count_and_observed_date_span(n, days):
    with pytest.raises(ValueError):
        assert_min_window(date(2024, 1, 1), date(2024, 1, 1)+timedelta(days=days-1), n)


@pytest.mark.parametrize("metadata", [{}, {"data_from_date": "2025-01-01", "data_to_date": "2026-01-01"}])
def test_gate_rejects_missing_provenance_and_overlap(metadata):
    with pytest.raises(ValueError):
        validate_window(metadata, date(2025, 6, 1), date(2026, 6, 1))


def test_gate_does_not_count_unscored_requested_window(session):
    seed(session)
    model = SimpleNamespace(metadata={"feature_schema": 2, "pipeline_sha256": pipeline_digest(), "data_from_date": "2023-01-01",
                                      "data_to_date": "2024-01-01"},
        feature_columns=["agf_raw"], softmax_temperature=1,
        booster=SimpleNamespace(predict=lambda x: np.zeros(len(x))))
    with pytest.raises(ValueError, match="1500 scored"):
        evaluate_gate(session, model, model, date(2025, 1, 1), date(2026, 6, 1))


def test_candidates_cannot_overwrite_live_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("GANYAN_MODEL_DIR", str(tmp_path))
    assert candidate_directory() == tmp_path / "candidates"
    with pytest.raises(ValueError):
        candidate_directory(tmp_path)


def test_promotion_requires_exact_gate_and_preserves_live_weights(tmp_path):
    root = tmp_path
    (root/"live.txt").write_text("old")
    (root/"live.meta.json").write_text("{}")
    live = artifact_identity(root/"live.txt", root/"live.meta.json")
    (root/"active.json").write_text(json.dumps({"version": 1, "heads": {"lightgbm_ranker":
        {"model": "live.txt", "metadata": "live.meta.json", **live}}}))
    candidate = root/"candidate"
    candidate.with_suffix(".txt").write_text("new")
    candidate.with_suffix(".meta.json").write_text("{}")
    gate = {"version": 2, "swap": True, "mode": "forward", "n": 1500,
        "coverage_days": 365, "delta_pp": 1.1, "mcnemar_p": 0.01, "pipeline_sha256": pipeline_digest(),
        "live_artifact": live, "candidate_artifact": artifact_identity(
            candidate.with_suffix(".txt"), candidate.with_suffix(".meta.json"))}
    report = root/"gate.json"
    report.write_text(json.dumps({**gate, "n": 1}))
    with pytest.raises(ValueError):
        promote(candidate, report, root=root)
    report.write_text(json.dumps(gate))
    release = promote(candidate, report, root=root)
    assert approved_paths("lightgbm_ranker", root)["model"] == release/"model.txt"
    assert (root/"live.txt").read_text() == "old"
    # The same stale report cannot authorize a second deployment.
    with pytest.raises(ValueError):
        promote(candidate, report, root=root)


def test_prediction_records_survive_mutable_slot_refresh(session):
    from ganyan.predictor.records import save_predictions, recorded_entries
    from ganyan.predictor.bayesian import Prediction
    race = seed(session, future=True)
    preds = [Prediction(e.horse_id, e.horse.name, p, 0.8, {})
             for e, p in zip(race.entries, [40, 30, 20, 10])]
    save_predictions(session, race.id, preds, "regression")
    session.flush()
    for e in race.entries:
        e.predicted_probability = 99
    assert [e.predicted_probability for e in recorded_entries(session, race.id)] == [40, 30, 20, 10]
    race.status = RaceStatus.resulted
    with pytest.raises(ValueError):
        save_predictions(session, race.id, preds, "regression")


def test_missing_pool_payout_does_not_permanently_settle_coupon(session):
    from ganyan.predictor.multi_race_picks import grade_pick
    race = seed(session)
    pool = MultiRacePool(date=race.date, track_id=race.track_id, pool_type="6li", pool_index=1,
        start_race_no=1, end_race_no=6, winning_combo="1/1/1/1/1/1")
    pick = MultiRacePick(date=race.date, track_id=race.track_id, pool_type="6li", pool_index=1,
        strategy="test", start_race_no=1, end_race_no=6, kept_horses_per_leg=[[1]]*6,
        total_tickets=1, ticket_unit_tl=1, stake_tl=1)
    session.add_all([pool, pick]); session.flush()
    assert grade_pick(session, pick) is None and not pick.graded
    pool.payout_tl = 1000
    assert grade_pick(session, pick) is True and float(pick.payout_tl) == 1000


def test_scheduler_failure_propagates(monkeypatch):
    from ganyan.scheduler import _job_agf_snapshot
    from ganyan.config import Settings
    def fail(coro):
        coro.close()
        raise RuntimeError("synthetic network failure")
    monkeypatch.setattr("ganyan.scheduler.asyncio.run", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        _job_agf_snapshot(Settings())


def test_mutating_routes_require_csrf_or_authenticated_api(session):
    from ganyan.web.app import create_app
    app = create_app(session_factory=sessionmaker(bind=session.bind), refresh_on_launch=False,
                     enable_scheduler=False)
    with app.test_client() as client:
        assert client.post("/predict/today", headers={"Origin": "https://evil.example"}).status_code == 403
        assert client.post("/predict/today").status_code == 403
        assert client.post("/predict/today", environ_overrides={"REMOTE_ADDR": "192.0.2.1"}).status_code == 401


def test_health_cannot_be_green_without_results_and_predictions():
    from ganyan.web.routes import _compute_health
    assert _compute_health(date.today(), None, None, 0)["status"] != "ok"


def test_missing_bayesian_values_are_neutral_not_slowest():
    from ganyan.predictor.bayes.standardize import zscore_missing
    values = zscore_missing(np.array([10, 20, np.nan]))
    np.testing.assert_allclose(values, [-1, 1, 0])


def test_malformed_halt_flag_fails_closed(tmp_path, monkeypatch):
    from ganyan.predictor.halt_flag import is_halted, set_halt
    path = tmp_path/"halt.json"
    monkeypatch.setenv("GANYAN_HALT_FLAG_PATH", str(path))
    path.write_text("[]")
    assert is_halted()
    set_halt("new", "test")
    assert path.read_text() == "[]"


def test_finish_time_inference_prefers_faster_horse(session):
    from ganyan.predictor.ml.predictor import LoadedModel, MLPredictor
    race = seed(session)
    model = LoadedModel(SimpleNamespace(predict=lambda x: np.array([70., 60., 80., 90.])),
                        ["agf_raw"], "time", metadata={"objective": "finish_time"})
    assert MLPredictor(session, model).predict(race.id)[0].horse_id == race.entries[1].horse_id


def test_post_race_external_signal_cannot_enter_feature(session):
    from ganyan.db.models import ExternalSignal
    from ganyan.predictor.features import compute_tipster_consensus
    race = seed(session)
    entry = race.entries[0]
    session.add(ExternalSignal(source_name="yarisrehberi", signal_type="tipster_pick",
        race_id=race.id, race_entry_id=entry.id, captured_at=race_cutoff(race)+timedelta(seconds=1),
        payload={"ticket_timestamp": "late"}))
    session.flush()
    assert compute_tipster_consensus(session, entry.id) is None


def test_external_plugin_failure_is_not_a_successful_empty_scrape(session, monkeypatch):
    from ganyan.scraper.external.resolver import fetch_and_resolve
    from ganyan.scraper.external import REGISTRY
    class Broken:
        def fetch_for_date(self, session, target):
            raise RuntimeError("plugin failed")
    monkeypatch.setitem(REGISTRY, "broken", Broken)
    with pytest.raises(RuntimeError, match="plugin failed"):
        fetch_and_resolve(session, date.today(), sources=["broken"])


def test_unapproved_candidates_are_not_loaded(tmp_path, monkeypatch):
    from ganyan.predictor.ml.ensemble import load_all_models
    (tmp_path/"active.json").write_text(json.dumps({"version": 1, "heads": {
        "approved": {"model": "approved.txt", "metadata": "approved.meta.json",
                     "model_sha256": "bad", "metadata_sha256": "bad"}}}))
    (tmp_path/"approved.txt").write_text("model")
    (tmp_path/"candidate.meta.json").write_text("{}")
    with pytest.raises(ValueError, match="Approved artifact changed"):
        load_all_models(tmp_path)


def test_example_configuration_disables_implicit_background_work():
    from ganyan.config import Settings
    settings = Settings(_env_file=".env.example")
    assert settings.ganyan_skip_scheduler and settings.ganyan_skip_launch_refresh
    assert settings.flask_host == "127.0.0.1"


def test_bayesian_training_keeps_unknown_jockey_but_excludes_partial_results(session):
    from ganyan.predictor.bayes.data import build_training_frame as bayes_frame
    race = seed(session)
    race.entries[0].jockey = None
    frame = bayes_frame(session, race.date, race.date, include_speed=False,
                        include_workouts=False, include_pace=False)
    assert len(frame.orderings[race.id]) == 4
    race.entries[0].finish_position = None
    frame = bayes_frame(session, race.date, race.date, include_speed=False,
                        include_workouts=False, include_pace=False)
    assert not frame.orderings
