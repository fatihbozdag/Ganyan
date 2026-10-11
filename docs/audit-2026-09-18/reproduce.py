"""Offline defect reproductions; no live DB, model writes, or network.

Run: .venv/bin/python docs/audit-2026-09-18/reproduce.py
Assertions describe defects at audit time, NOT desired regression behavior.
"""
from datetime import date, datetime
from unittest.mock import patch
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ganyan.db.models import (
    Base, Horse, Race, RaceEntry, RaceStatus, Track, Pick, AgfSnapshot,
    MultiRacePool,
)
from ganyan.scraper.parser import ParsedHorseEntry, ParsedRaceCard
from ganyan.scraper.backfill import store_race_card
from ganyan.predictor.features import compute_late_agf_drift
from ganyan.predictor.ml.features import TrainingFrame, build_race_frame, build_training_frame
from ganyan.predictor.ml.trainer import _evaluate_ranker
from ganyan.predictor.multi_race_picks import (
    CouponDraft, persist_coupon, grade_pick, grade_all_pending_multi,
)
from ganyan.predictor.picks import grade_race, strategy_summary
from ganyan.web.app import create_app
from ganyan.web.routes import _compute_health

engine = create_engine("sqlite:///:memory:")
Base.metadata.create_all(engine)
factory = sessionmaker(bind=engine)
evidence = {}
day = date(2026, 1, 1)

with Session(engine) as s:
    # A cached name bypasses the authoritative AtId identity check.
    s.add(Horse(name="Shared name", tjk_at_id=111))
    s.flush()
    race = store_race_card(s, ParsedRaceCard(
        track_name="Audit", date=day, race_number=1,
        surface="kum", distance_meters=1200, post_time="14:00",
        horses=[ParsedHorseEntry(name="Shared name", tjk_at_id=222)],
    ))
    s.flush()
    actual_id = race.entries[0].horse.tjk_at_id
    assert actual_id == 111
    evidence["horse_identity"] = {"incoming_at_id": 222, "stored_at_id": actual_id}

    # Fresh cards do not update changed race-level surface/post time.
    store_race_card(s, ParsedRaceCard(
        track_name="Audit", date=day, race_number=1,
        surface="çim", distance_meters=1600, post_time="15:00",
    ))
    assert (race.surface, race.distance_meters, race.post_time) == ("kum", 1200, "14:00")
    evidence["race_refresh"] = [race.surface, race.distance_meters, race.post_time]

    # A snapshot collected after the race changes the historical feature.
    entry = race.entries[0]
    s.add(AgfSnapshot(race_entry_id=entry.id, agf=10, taken_at=datetime(2026, 1, 1, 10)))
    s.flush()
    before = compute_late_agf_drift(s, entry.id)
    s.add(AgfSnapshot(race_entry_id=entry.id, agf=90, taken_at=datetime(2026, 2, 1)))
    s.flush()
    after = compute_late_agf_drift(s, entry.id)
    assert before is None and after == 80
    evidence["post_race_snapshot"] = {"before": before, "after": after}

    # A confirmed Plase pool still leaves a losing pick financially unsettled.
    race.status = RaceStatus.resulted
    entry.finish_position = 1
    entry.plase_payout_tl = 2
    for pos in [2, 3]:
        h = Horse(name=f"H{pos}")
        s.add(h)
        s.flush()
        s.add(RaceEntry(race_id=race.id, horse_id=h.id, finish_position=pos,
                        plase_payout_tl=2 if pos == 2 else None))
        if pos == 3:
            loser_id = h.id
    s.flush()
    s.expire(race, ["entries"])
    pick = Pick(race_id=race.id, strategy="plase_top1", combination=[loser_id],
                stake_tl=100, ticket_count=1)
    s.add(pick)
    s.flush()
    grade_race(s, race.id)
    assert pick.graded and pick.hit is False and pick.net_tl is None
    evidence["plase_loss"] = {"hit": pick.hit, "net_tl": pick.net_tl,
                               "summary": strategy_summary(s)}

    # Exotic backtesting still omits the unit-price divisor used by the ledger.
    from ganyan.predictor.exotic_evaluate import evaluate_pool
    ordered = sorted(race.entries, key=lambda e: e.finish_position)
    race.uclu_payout_tl = 10
    triple = Pick(race_id=race.id, strategy="uclu_top1",
                  combination=[e.horse_id for e in ordered], stake_tl=100, ticket_count=1)
    s.add(triple)
    s.flush()
    grade_race(s, race.id)
    fake = SimpleNamespace(predict=lambda rid: [
        SimpleNamespace(horse_id=e.horse_id, probability=p)
        for e, p in zip(ordered, [60, 30, 10])
    ])
    backtest = evaluate_pool(s, "uclu", 1, predictor_factory=lambda session: fake)
    assert triple.payout_tl == 500 and backtest.total_payout_tl == 1000
    evidence["exotic_payout_units"] = {"ledger_payout": triple.payout_tl,
                                        "backtest_payout": backtest.total_payout_tl}

    # Training and inference disagree on sparse field-average coverage.
    entry.hp = 80
    entry.agf = 30
    train = build_training_frame(s)
    serve = build_race_frame(s, race.id)
    assert train.features.class_indicator.isna().all()
    assert serve.loc[serve.horse_id == entry.horse_id, "class_indicator"].notna().all()
    evidence["train_serve_coverage"] = {"train": "NaN", "serve": float(serve.iloc[0].class_indicator)}

    # A winning combination can be settled at zero before payout arrives.
    draft = CouponDraft([[1]] * 6, [0.6] * 6, 1)
    multi = persist_coupon(s, day, "Audit", 1, draft)
    pool = MultiRacePool(date=day, track_id=race.track_id, pool_type="6li",
                         pool_index=1, winning_combo="1/1/1/1/1/1", payout_tl=None)
    s.add(pool)
    s.flush()
    grade_pick(s, multi)
    assert multi.hit and multi.graded and multi.payout_tl == 0 and multi.net_tl == -1
    pool.payout_tl = 1000
    s.flush()
    assert grade_all_pending_multi(s) == 0 and multi.payout_tl == 0
    evidence["multi_early_settlement"] = {"hit": multi.hit, "payout": multi.payout_tl,
                                           "net": multi.net_tl}

    # Re-persisting a different window overwrites the settled coupon.
    old_id = multi.id
    replacement = persist_coupon(s, day, "Audit", 3, CouponDraft([[2]] * 6, [0.6] * 6, 1))
    assert replacement.id == old_id and not replacement.graded and replacement.hit is None
    evidence["multi_overwrite"] = {"same_id": True, "start_race": replacement.start_race_no,
                                    "graded": replacement.graded}

# Label-sorted training rows let a constant-score model appear perfect.
frame = TrainingFrame(
    features=pd.DataFrame({"x": [0.] * 8}),
    target=pd.Series([3, 2, 1, 0] * 2),
    ev_target=pd.Series([0.] * 8), finish_time_target=pd.Series([0.] * 8),
    groups=pd.Series([1] * 4 + [2] * 4), race_dates=pd.Series([day] * 8),
)
class ConstantModel:
    def predict(self, features):
        return np.zeros(len(features))
metrics = _evaluate_ranker(ConstantModel(), frame)
assert metrics["top1_accuracy"] == 100
evidence["constant_model_accuracy"] = metrics

# Health accepts yesterday's program, no results, no predictions.
health = _compute_health(date.today(), None, None, 0)
assert health["status"] == "ok"
evidence["health_without_results_or_predictions"] = health

# The POST endpoint accepts an unauthenticated cross-origin request.
app = create_app(session_factory=factory, refresh_on_launch=False, enable_scheduler=False)
app.config["TESTING"] = True
with app.test_client() as client:
    response = client.post("/predict/today", headers={"Origin": "https://untrusted.example",
                                                     "Accept": "application/json"})
    assert response.status_code == 200
    evidence["unauthenticated_post"] = {"status": response.status_code, "body": response.get_json()}

# A scheduler scrape failure is swallowed and returns normally.
from ganyan.scheduler import _job_agf_snapshot
from ganyan.config import Settings
def fail_run(coro):
    coro.close()
    raise RuntimeError("synthetic network failure")
with patch("ganyan.scheduler.asyncio.run", side_effect=fail_run):
    result = _job_agf_snapshot(Settings())
assert result is None
evidence["scheduler_failure_returns_successfully"] = True

# The OOS gate can approve after scoring only one of 1,500 queried races.
import importlib.util
import contextlib
import io
from types import SimpleNamespace
from unittest.mock import MagicMock
spec = importlib.util.spec_from_file_location("audit_gate", "logs/oos_model_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
db = MagicMock()
initial = MagicMock()
initial.fetchall.return_value = [(i,) for i in range(1500)]
winner = MagicMock()
winner.all.return_value = [(1,)]
agf = MagicMock()
agf.fetchone.return_value = (1,)
empty = MagicMock()
empty.all.return_value = []
db.execute.side_effect = [initial, winner, agf] + [empty] * 1499
live_model = SimpleNamespace(feature_columns=["x"], metadata={"from_date": "2025-01-01"})
candidate = SimpleNamespace(feature_columns=["x"], metadata={"from_date": "2025-01-01"})
def fake_predictor(session, model):
    return SimpleNamespace(predict=lambda rid: [SimpleNamespace(horse_id=1 if model is candidate else 2)])
with patch.object(gate, "get_session", return_value=db), \
     patch.object(gate, "load_named", side_effect=[live_model, candidate]), \
     patch.object(gate, "MLPredictor", side_effect=fake_predictor), \
     patch("sys.argv", ["oos_model_gate.py", "--candidate", "fake"]), \
     patch.object(gate.Path, "write_text") as write, contextlib.redirect_stdout(io.StringIO()):
    gate.main()
    gate_result = json.loads(write.call_args.args[0])
assert gate_result["n"] == 1 and gate_result["swap"] is True
evidence["oos_gate_one_scored_race"] = gate_result
print(json.dumps(evidence, indent=2, default=str))
