"""Tests for the prediction evaluation module."""

import math

import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Track, Race, Horse, RaceEntry, RaceStatus
from ganyan.predictor.bayesian import BayesianPredictor
from ganyan.predictor.evaluate import (
    RaceEvaluation,
    EvaluationSummary,
    evaluate_race,
    evaluate_all,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _create_track(session, name="Istanbul"):
    track = Track(name=name, city=name)
    session.add(track)
    session.flush()
    return track


def _create_race(session, track, *, race_number=1, status=RaceStatus.resulted):
    race = Race(
        track_id=track.id,
        date=date(2026, 4, 5),
        race_number=race_number,
        distance_meters=1400,
        surface="cim",
        status=status,
    )
    session.add(race)
    session.flush()
    return race


def _add_entry(
    session, race, horse_name, *,
    finish_position=None, predicted_probability=None,
    weight_kg=57.0, hp=85.0, kgs=21, eid="1.30.45", last_six="1 3 2 4 1 2",
):
    horse = Horse(name=horse_name, age=4)
    session.add(horse)
    session.flush()
    entry = RaceEntry(
        race_id=race.id,
        horse_id=horse.id,
        gate_number=1,
        jockey=f"Jockey {horse_name}",
        weight_kg=weight_kg,
        hp=hp,
        kgs=kgs,
        eid=eid,
        last_six=last_six,
        finish_position=finish_position,
        predicted_probability=predicted_probability,
    )
    session.add(entry)
    session.flush()
    return entry


# -----------------------------------------------------------------------
# predict_and_save
# -----------------------------------------------------------------------


class TestPredictAndSave:
    def test_saves_probabilities_to_entries(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track, status=RaceStatus.scheduled)
        _add_entry(db_session, race, "Horse A", hp=90.0)
        _add_entry(db_session, race, "Horse B", hp=80.0)
        _add_entry(db_session, race, "Horse C", hp=70.0)
        db_session.commit()

        predictor = BayesianPredictor(db_session)
        predictions = predictor.predict_and_save(race.id)
        db_session.commit()

        assert len(predictions) == 3
        entries = (
            db_session.query(RaceEntry)
            .filter(RaceEntry.race_id == race.id)
            .all()
        )
        for entry in entries:
            assert entry.predicted_probability is not None
            assert float(entry.predicted_probability) > 0

    def test_probabilities_match_predictions(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track, status=RaceStatus.scheduled)
        e1 = _add_entry(db_session, race, "Horse X", hp=95.0)
        e2 = _add_entry(db_session, race, "Horse Y", hp=75.0)
        db_session.commit()

        predictor = BayesianPredictor(db_session)
        predictions = predictor.predict_and_save(race.id)
        db_session.commit()

        pred_map = {p.horse_id: p.probability for p in predictions}
        for entry in [e1, e2]:
            db_session.refresh(entry)
            assert abs(float(entry.predicted_probability) - pred_map[entry.horse_id]) < 0.01

    def test_no_entries_returns_empty(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track, status=RaceStatus.scheduled)
        db_session.commit()

        predictor = BayesianPredictor(db_session)
        predictions = predictor.predict_and_save(race.id)
        assert predictions == []


# -----------------------------------------------------------------------
# evaluate_race
# -----------------------------------------------------------------------


class TestEvaluateRace:
    def test_race_with_predictions_and_results(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track)
        _add_entry(
            db_session, race, "Winner",
            finish_position=1, predicted_probability=40.0,
        )
        _add_entry(
            db_session, race, "Second",
            finish_position=2, predicted_probability=35.0,
        )
        _add_entry(
            db_session, race, "Third",
            finish_position=3, predicted_probability=25.0,
        )
        db_session.commit()

        ev = evaluate_race(db_session, race.id)
        assert ev is not None
        assert ev.race_id == race.id
        assert ev.winner_name == "Winner"
        assert ev.winner_predicted_prob == 40.0
        assert ev.winner_predicted_rank == 1
        assert ev.top1_correct is True
        assert ev.top3_correct is True
        assert ev.num_horses == 3

    def test_winner_not_top_pick(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track)
        # Winner had lowest predicted probability
        _add_entry(
            db_session, race, "Surprise",
            finish_position=1, predicted_probability=10.0,
        )
        _add_entry(
            db_session, race, "Favorite",
            finish_position=3, predicted_probability=50.0,
        )
        _add_entry(
            db_session, race, "Contender",
            finish_position=2, predicted_probability=40.0,
        )
        db_session.commit()

        ev = evaluate_race(db_session, race.id)
        assert ev is not None
        assert ev.winner_predicted_rank == 3
        assert ev.top1_correct is False
        assert ev.top3_correct is True

    def test_winner_outside_top3(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track)
        _add_entry(db_session, race, "Longshot", finish_position=1, predicted_probability=5.0)
        _add_entry(db_session, race, "H2", finish_position=2, predicted_probability=35.0)
        _add_entry(db_session, race, "H3", finish_position=3, predicted_probability=30.0)
        _add_entry(db_session, race, "H4", finish_position=4, predicted_probability=20.0)
        _add_entry(db_session, race, "H5", finish_position=5, predicted_probability=10.0)
        db_session.commit()

        ev = evaluate_race(db_session, race.id)
        assert ev is not None
        assert ev.winner_predicted_rank == 5
        assert ev.top1_correct is False
        assert ev.top3_correct is False

    def test_no_predictions_returns_none(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track)
        _add_entry(db_session, race, "NoPredict", finish_position=1)
        db_session.commit()

        ev = evaluate_race(db_session, race.id)
        assert ev is None

    def test_no_results_returns_none(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track, status=RaceStatus.scheduled)
        _add_entry(db_session, race, "Pending", predicted_probability=50.0)
        db_session.commit()

        ev = evaluate_race(db_session, race.id)
        assert ev is None

    def test_no_winner_returns_none(self, db_session):
        """Race resulted but no entry has finish_position=1."""
        track = _create_track(db_session)
        race = _create_race(db_session, track)
        _add_entry(
            db_session, race, "DNF",
            finish_position=None, predicted_probability=50.0,
        )
        db_session.commit()

        ev = evaluate_race(db_session, race.id)
        assert ev is None

    def test_nonexistent_race_returns_none(self, db_session):
        ev = evaluate_race(db_session, 9999)
        assert ev is None

    def test_winner_without_prediction_in_field(self, db_session):
        """Winner exists but their predicted_probability is None, while others have predictions."""
        track = _create_track(db_session)
        race = _create_race(db_session, track)
        _add_entry(
            db_session, race, "UnpredWinner",
            finish_position=1, predicted_probability=None,
        )
        _add_entry(
            db_session, race, "Predicted",
            finish_position=2, predicted_probability=60.0,
        )
        db_session.commit()

        ev = evaluate_race(db_session, race.id)
        assert ev is not None
        assert ev.winner_predicted_prob is None
        assert ev.winner_predicted_rank is None
        assert ev.top1_correct is False
        assert ev.top3_correct is False


# -----------------------------------------------------------------------
# evaluate_all
# -----------------------------------------------------------------------


class TestEvaluateAll:
    def test_multiple_races(self, db_session):
        track = _create_track(db_session)
        # Race 1: top pick wins
        race1 = _create_race(db_session, track, race_number=1)
        _add_entry(db_session, race1, "R1H1", finish_position=1, predicted_probability=50.0)
        _add_entry(db_session, race1, "R1H2", finish_position=2, predicted_probability=30.0)
        _add_entry(db_session, race1, "R1H3", finish_position=3, predicted_probability=20.0)

        # Race 2: top pick loses
        race2 = _create_race(db_session, track, race_number=2)
        _add_entry(db_session, race2, "R2H1", finish_position=1, predicted_probability=20.0)
        _add_entry(db_session, race2, "R2H2", finish_position=2, predicted_probability=50.0)
        _add_entry(db_session, race2, "R2H3", finish_position=3, predicted_probability=30.0)
        db_session.commit()

        summary, evaluations = evaluate_all(db_session)
        assert summary.total_races == 2
        assert len(evaluations) == 2
        assert summary.top1_accuracy == 50.0  # 1 of 2
        assert summary.top3_accuracy == 100.0  # both in top 3

    def test_no_resulted_races(self, db_session):
        track = _create_track(db_session)
        _create_race(db_session, track, status=RaceStatus.scheduled)
        db_session.commit()

        summary, evaluations = evaluate_all(db_session)
        assert summary.total_races == 0
        assert evaluations == []
        assert summary.top1_accuracy == 0.0
        assert summary.log_loss == 0.0
        assert summary.roi_simulation == 0.0

    def test_empty_db(self, db_session):
        summary, evaluations = evaluate_all(db_session)
        assert summary.total_races == 0
        assert evaluations == []

    def test_log_loss_calculation(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track, race_number=1)
        _add_entry(db_session, race, "H1", finish_position=1, predicted_probability=40.0)
        _add_entry(db_session, race, "H2", finish_position=2, predicted_probability=60.0)
        db_session.commit()

        summary, _ = evaluate_all(db_session)
        expected_log_loss = -math.log(40.0 / 100.0)
        assert abs(summary.log_loss - expected_log_loss) < 0.0001

    def test_roi_top_pick_wins(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track, race_number=1)
        _add_entry(db_session, race, "Fav", finish_position=1, predicted_probability=50.0)
        _add_entry(db_session, race, "Other", finish_position=2, predicted_probability=50.0)
        db_session.commit()

        summary, _ = evaluate_all(db_session)
        # Bet 100, payout = 10000/50 = 200, ROI = (200-100)/100 = 1.0
        assert abs(summary.roi_simulation - 1.0) < 0.01

    def test_roi_top_pick_loses(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track, race_number=1)
        _add_entry(db_session, race, "Upset", finish_position=1, predicted_probability=10.0)
        _add_entry(db_session, race, "Fav", finish_position=2, predicted_probability=90.0)
        db_session.commit()

        summary, _ = evaluate_all(db_session)
        # Bet 100, payout = 0, ROI = (0-100)/100 = -1.0
        assert abs(summary.roi_simulation - (-1.0)) < 0.01

    def test_avg_winner_rank(self, db_session):
        track = _create_track(db_session)
        # Race 1: winner ranked 1st
        race1 = _create_race(db_session, track, race_number=1)
        _add_entry(db_session, race1, "W1", finish_position=1, predicted_probability=60.0)
        _add_entry(db_session, race1, "L1", finish_position=2, predicted_probability=40.0)

        # Race 2: winner ranked 3rd
        race2 = _create_race(db_session, track, race_number=2)
        _add_entry(db_session, race2, "W2", finish_position=1, predicted_probability=10.0)
        _add_entry(db_session, race2, "X2", finish_position=2, predicted_probability=50.0)
        _add_entry(db_session, race2, "Y2", finish_position=3, predicted_probability=40.0)
        db_session.commit()

        summary, _ = evaluate_all(db_session)
        assert abs(summary.avg_winner_rank - 2.0) < 0.01  # (1+3)/2

    def test_skips_races_without_predictions(self, db_session):
        track = _create_track(db_session)
        # Race with predictions
        race1 = _create_race(db_session, track, race_number=1)
        _add_entry(db_session, race1, "PH1", finish_position=1, predicted_probability=50.0)
        _add_entry(db_session, race1, "PH2", finish_position=2, predicted_probability=50.0)

        # Race without predictions
        race2 = _create_race(db_session, track, race_number=2)
        _add_entry(db_session, race2, "NPH1", finish_position=1)
        _add_entry(db_session, race2, "NPH2", finish_position=2)
        db_session.commit()

        summary, evaluations = evaluate_all(db_session)
        assert summary.total_races == 1
        assert len(evaluations) == 1


# -----------------------------------------------------------------------
# Temporal holdout + calibration + baselines
# -----------------------------------------------------------------------


class TestNewMetrics:
    def test_cutoff_filters_older_races(self, db_session):
        track = _create_track(db_session)
        old = Race(
            track_id=track.id, date=date(2026, 1, 1), race_number=1,
            distance_meters=1400, surface="Çim", status=RaceStatus.resulted,
        )
        new = Race(
            track_id=track.id, date=date(2026, 4, 1), race_number=2,
            distance_meters=1400, surface="Çim", status=RaceStatus.resulted,
        )
        db_session.add_all([old, new])
        db_session.flush()
        _add_entry(db_session, old, "OldWinner", finish_position=1, predicted_probability=60.0)
        _add_entry(db_session, old, "OldLoser", finish_position=2, predicted_probability=40.0)
        _add_entry(db_session, new, "NewWinner", finish_position=1, predicted_probability=55.0)
        _add_entry(db_session, new, "NewLoser", finish_position=2, predicted_probability=45.0)
        db_session.commit()

        summary, evaluations = evaluate_all(db_session, cutoff_date=date(2026, 3, 1))
        assert summary.total_races == 1
        assert summary.cutoff_date == date(2026, 3, 1)
        assert len(evaluations) == 1
        assert evaluations[0].date == date(2026, 4, 1)

    def test_random_baseline_top1(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track, race_number=1)
        _add_entry(db_session, race, "A", finish_position=1, predicted_probability=25.0)
        _add_entry(db_session, race, "B", finish_position=2, predicted_probability=25.0)
        _add_entry(db_session, race, "C", finish_position=3, predicted_probability=25.0)
        _add_entry(db_session, race, "D", finish_position=4, predicted_probability=25.0)
        db_session.commit()

        summary, _ = evaluate_all(db_session)
        # 1 / 4 horses = 25% expected random top-1.
        assert abs(summary.random_baseline_top1 - 25.0) < 0.01

    def test_brier_score_perfect_confident_win(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track, race_number=1)
        _add_entry(db_session, race, "A", finish_position=1, predicted_probability=100.0)
        _add_entry(db_session, race, "B", finish_position=2, predicted_probability=0.0)
        db_session.commit()

        summary, _ = evaluate_all(db_session)
        # Perfect confident prediction → brier score ≈ 0.
        assert summary.brier_score == pytest.approx(0.0, abs=1e-6)

    def test_calibration_buckets_present(self, db_session):
        track = _create_track(db_session)
        race = _create_race(db_session, track, race_number=1)
        _add_entry(db_session, race, "A", finish_position=1, predicted_probability=70.0)
        _add_entry(db_session, race, "B", finish_position=2, predicted_probability=20.0)
        _add_entry(db_session, race, "C", finish_position=3, predicted_probability=10.0)
        db_session.commit()

        summary, _ = evaluate_all(db_session, num_calibration_bins=10)
        assert len(summary.calibration) >= 1
        # Counts should sum to the number of predicted entries.
        total = sum(b.count for b in summary.calibration)
        assert total == 3
