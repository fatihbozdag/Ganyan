"""Integrity smoke tests: guarantee predictions don't leak race outcomes.

Leakage would be a silent bug where the model's inference-time inputs
include a field only knowable *after* the race ran (finish_position,
finish_time, race-specific payout).  The entire +ROI claim collapses
if predictions have ever seen result data for the target race.

Two complementary tests:
  1. Static check of FEATURE_COLUMNS against a post-race blacklist.
  2. Dynamic invariance check: clearing the target race's finish data
     must leave predictions bit-for-bit unchanged.

Written after the 2026-04-23 session where the user asked "is it maybe
predicting after seeing winning horses?"  No leakage was found, but
these tests institutionalise the check.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from sqlalchemy import create_engine
from ganyan.db.models import Base
from tests.test_predictor.test_ml.test_ml_pipeline import _seed_many
from ganyan.db.models import Race, RaceEntry, RaceStatus


def test_feature_columns_exclude_post_race_fields() -> None:
    """Static check: FEATURE_COLUMNS must not list any post-race field name.

    These fields are only knowable AFTER the race runs, so their presence
    in the feature matrix would be an obvious leakage path.
    """
    from ganyan.predictor.ml.features import FEATURE_COLUMNS

    FORBIDDEN = {
        "finish_position",
        "finish_time",
        "performance_score",  # derived from finish_position
        "ganyan_payout_tl",
        "ikili_payout_tl",
        "sirali_ikili_payout_tl",
        "uclu_payout_tl",
        "dortlu_payout_tl",
        "pace_l800_leader_s",   # measured during the race
        "pace_l800_runner_up_s",
    }
    leaked = set(FEATURE_COLUMNS) & FORBIDDEN
    assert not leaked, (
        f"Feature columns contain post-race fields: {sorted(leaked)}. "
        "These fields are only knowable after the race has been run, "
        "so including them in training/inference data is an outright "
        "leakage bug.  Remove them from FEATURE_COLUMNS."
    )


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _seed_many(session, n_races=5)
        yield session
    engine.dispose()


@pytest.fixture
def resulted_race_id(db_session: Session) -> int:
    """Pick any resulted race with finish data for the invariance check."""
    race = (
        db_session.query(Race)
        .filter(Race.status == RaceStatus.resulted)
        .order_by(Race.date.desc(), Race.id.desc())
        .first()
    )
    if race is None:
        pytest.skip("No resulted races in DB — cannot run leakage check.")
    # Ensure at least one entry has finish_position so the test is meaningful.
    fp_count = (
        db_session.query(RaceEntry)
        .filter(
            RaceEntry.race_id == race.id,
            RaceEntry.finish_position.isnot(None),
        )
        .count()
    )
    if fp_count == 0:
        pytest.skip(
            f"Race {race.id} marked resulted but has no finish_position rows.",
        )
    return race.id


def test_predict_invariant_to_target_race_finish_data(
    db_session: Session, resulted_race_id: int,
) -> None:
    """Clearing the target race's own finish data must not change predictions.

    This catches leakage where the predictor/features builder
    accidentally reads finish_position / finish_time / any post-race
    field from the RaceEntry of the race being predicted.

    Procedure:
      1. Predict the target race — capture probabilities (A).
      2. Null out finish_position, finish_time on the target race's
         entries (in-transaction, no commit).
      3. Expire session cache so next predict re-reads from DB.
      4. Predict again — capture probabilities (B).
      5. Assert A == B to within floating-point tolerance.
      6. Rollback so the DB is unchanged.

    A difference of more than 1e-9 per horse is a leakage signal.
    """
    try:
        from ganyan.predictor.ml import MLPredictor
        from ganyan.predictor.ml.predictor import load_latest_model
    except ImportError:
        pytest.skip("ML predictor not available — skipping leakage check.")

    from types import SimpleNamespace
    import numpy as np
    from ganyan.predictor.ml.predictor import LoadedModel
    from ganyan.predictor.ml.features import FEATURE_COLUMNS
    # A deterministic feature-sensitive scorer, independent of production files.
    booster = SimpleNamespace(predict=lambda x: np.nan_to_num(x.to_numpy()).sum(axis=1))
    predictor = MLPredictor(db_session, LoadedModel(booster, FEATURE_COLUMNS, "test"))

    # Baseline — with finish data in place.
    preds_a = predictor.predict(resulted_race_id)
    if not preds_a:
        pytest.skip(f"Race {resulted_race_id} has no predictions possible.")
    probs_a = {p.horse_id: p.probability for p in preds_a}

    # Null out every post-race field on the target race — entry-level
    # (finish_position, finish_time, performance_score) AND race-level
    # (pace_l800_*, exotic payouts).  If predict() is truly invariant
    # to the target race's own outcome then probabilities must be
    # bit-for-bit identical before/after.
    entries = (
        db_session.query(RaceEntry)
        .filter(RaceEntry.race_id == resulted_race_id)
        .all()
    )
    race = db_session.get(Race, resulted_race_id)
    saved_entry = [
        (e.id, e.finish_position, e.finish_time, e.performance_score)
        for e in entries
    ]
    saved_race = {
        f: getattr(race, f)
        for f in (
            "pace_l800_leader_s", "pace_l800_runner_up_s",
            "ganyan_payout_tl", "ikili_payout_tl",
            "sirali_ikili_payout_tl", "uclu_payout_tl", "dortlu_payout_tl",
        )
    }
    try:
        for e in entries:
            e.finish_position = None
            e.finish_time = None
            e.performance_score = None
        for f in saved_race:
            setattr(race, f, None)
        db_session.flush()
        db_session.expire_all()  # force re-read from DB on next predict

        preds_b = predictor.predict(resulted_race_id)
        probs_b = {p.horse_id: p.probability for p in preds_b}

        # Compare probability by horse.
        assert set(probs_a.keys()) == set(probs_b.keys()), (
            "Different horses in predictions before/after clearing finish. "
            "Feature builder may be dropping rows based on finish data."
        )

        tolerance = 1e-9
        diffs = []
        for hid in probs_a:
            delta = abs(probs_a[hid] - probs_b[hid])
            if delta > tolerance:
                diffs.append((hid, probs_a[hid], probs_b[hid], delta))
        assert not diffs, (
            "LEAKAGE DETECTED — predictions changed when target race's own "
            "finish data was cleared.  The predictor must only use pre-race "
            "features.  Differences:\n  "
            + "\n  ".join(
                f"horse_id={hid}: {a:.6f} -> {b:.6f} (Δ={d:.2e})"
                for (hid, a, b, d) in diffs[:10]
            )
        )
    finally:
        # Restore post-race data so we don't leave the DB in a weird state.
        for (eid, fp, ft, ps) in saved_entry:
            entry = db_session.get(RaceEntry, eid)
            if entry is not None:
                entry.finish_position = fp
                entry.finish_time = ft
                entry.performance_score = ps
        for f, v in saved_race.items():
            setattr(race, f, v)
        db_session.flush()
        db_session.rollback()


def test_build_race_frame_excludes_target_finish(db_session: Session) -> None:
    """Feature frame for a race must not carry that race's finish_position.

    Defensive check on ``build_race_frame`` output: the DataFrame must
    not contain a column named after any post-race field.  Covers the
    case where a future feature-engineering step accidentally adds one.
    """
    try:
        from ganyan.predictor.ml.features import build_race_frame
    except ImportError:
        pytest.skip("ML features module not available.")

    race = (
        db_session.query(Race)
        .filter(Race.status == RaceStatus.resulted)
        .order_by(Race.date.desc())
        .first()
    )
    if race is None:
        pytest.skip("No resulted races in DB.")

    frame = build_race_frame(db_session, race.id)
    if frame.empty:
        pytest.skip(f"Race {race.id} produces empty frame (no features).")

    POST_RACE_COLUMNS = {
        "finish_position",
        "finish_time",
        "performance_score",
        "ganyan_payout_tl",
        "ikili_payout_tl",
        "sirali_ikili_payout_tl",
        "uclu_payout_tl",
        "dortlu_payout_tl",
        "pace_l800_leader_s",
        "pace_l800_runner_up_s",
    }
    leaked_cols = set(frame.columns) & POST_RACE_COLUMNS
    assert not leaked_cols, (
        f"build_race_frame output contains post-race columns: "
        f"{sorted(leaked_cols)}.  These should never appear in inference data."
    )
