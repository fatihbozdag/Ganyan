"""Ensemble forward gate, manifest assembly and shared paired metrics."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Race
from ganyan.predictor.ml import gate
from ganyan.predictor.ml.ensemble import load_all_models
from ganyan.predictor.ml.linear_ranker import train_conditional_logit
from ganyan.predictor.ml.trainer import train_ranker
from tests.test_predictor.test_ml.test_ml_pipeline import _seed_many


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        _seed_many(s, n_races=40)
        yield s
    engine.dispose()


@pytest.fixture
def small_window(monkeypatch):
    monkeypatch.setattr(gate, "MIN_WINDOW_DAYS", 1)
    monkeypatch.setattr(gate, "MIN_RACE_COUNT", 1)


@pytest.mark.parametrize("b,c,expected", [(0, 6, 0.03125), (0, 0, 1.0), (3, 3, 1.0), (1, 9, 0.021484375)])
def test_mcnemar_exact_known_values(b, c, expected):
    assert gate.mcnemar_exact(b, c) == pytest.approx(expected)


def _ranked(entries, first):
    order = sorted(entries, key=lambda e: (e.horse_id != first, e.horse_id))
    return [SimpleNamespace(horse_id=e.horse_id, probability=1.0) for e in order]


def test_paired_top1_counts_discordant_pairs(session, small_window):
    races = session.query(Race).all()
    start, end = min(r.date for r in races), max(r.date for r in races)
    by_race = {r.id: r for r in races}

    def winner_first(race_id):
        entries = by_race[race_id].entries
        return _ranked(entries, next(e.horse_id for e in entries if e.finish_position == 1))

    def last_first(race_id):
        entries = by_race[race_id].entries
        return _ranked(entries, next(e.horse_id for e in entries if e.finish_position == 6))

    result = gate.paired_top1(session, last_first, winner_first, start, end)
    assert result["n"] == len(races)
    assert result["live_top1_pct"] == 0 and result["candidate_top1_pct"] == 100
    assert result["candidate_top3_pct"] == 100
    assert result["delta_pp"] == 100
    assert result["mcnemar_p"] == pytest.approx(gate.mcnemar_exact(0, len(races)))
    assert result["agf_baseline"]["n"] == len(races)


def _train_set(session, root, to_date):
    train_ranker(session, to_date=to_date, holdout_fraction=0.2, num_boost_round=20,
                 model_dir=root, model_name="lightgbm_ranker")
    train_conditional_logit(session, to_date=to_date, epochs=5, model_dir=root,
                            model_name="linear_conditional_logit")
    return gate.write_manifest(root)


def test_ensemble_gate_end_to_end(session, small_window, tmp_path):
    dates = sorted({r.date for r in session.query(Race).all()})
    cutoff = dates[len(dates) // 2]
    live_root, cand_root = tmp_path / "live", tmp_path / "cand"
    manifest = json.loads(_train_set(session, live_root, cutoff).read_text())
    assert set(manifest["heads"]) == {"lightgbm_ranker", "linear_conditional_logit"}
    assert manifest["heads"]["linear_conditional_logit"]["model"].endswith(".npz")
    _train_set(session, cand_root, cutoff)

    live, cand = load_all_models(live_root), load_all_models(cand_root)
    later = [d for d in dates if d > cutoff]
    result = gate.evaluate_ensemble_gate(session, live, cand, later[0], later[-1])
    assert result["n"] > 0
    assert result["swap"] is False  # ensemble gate never authorizes promotion
    assert set(result["candidate_artifact"]) == {"lightgbm_ranker", "linear_conditional_logit"}

    # Forward window overlapping training data is refused.
    with pytest.raises(ValueError, match="overlaps"):
        gate.evaluate_ensemble_gate(session, live, cand, dates[0], dates[-1])


def test_ensemble_gate_rejects_head_without_provenance(session, small_window, tmp_path):
    dates = sorted({r.date for r in session.query(Race).all()})
    root = tmp_path / "set"
    _train_set(session, root, dates[len(dates) // 2])
    models = load_all_models(root)
    models[0].metadata.pop("data_to_date")
    with pytest.raises(ValueError, match="provenance"):
        gate.evaluate_ensemble_gate(session, models, load_all_models(root), dates[-2], dates[-1])


def test_write_manifest_refuses_production_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GANYAN_MODEL_DIR", str(tmp_path))
    with pytest.raises(ValueError):
        gate.write_manifest(tmp_path)
