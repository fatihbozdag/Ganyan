"""Tests for rolling-PnL halt canary (premortem failure mode 02)."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Pick, Race, Track
from ganyan.predictor import rolling_pnl_halt


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        track = Track(name="Test")
        s.add(track)
        s.flush()
        for i in range(60):
            r = Race(
                track_id=track.id,
                date=datetime.now(timezone.utc).date() - timedelta(days=i % 30),
                race_number=i + 1,
                distance_meters=1400,
            )
            s.add(r)
        s.flush()
        yield s
    engine.dispose()


def _add_pick(session, strategy, stake, payout, hit, days_ago, race_idx=None):
    """race_idx defaults to a per-strategy counter so the (race_id, strategy)
    uniqueness constraint isn't tripped when many picks land on the same
    days_ago bucket."""
    races = session.query(Race).all()
    if race_idx is None:
        race_idx = (
            session.query(Pick).filter(Pick.strategy == strategy).count()
        )
    race = races[race_idx % len(races)]
    p = Pick(
        race_id=race.id,
        strategy=strategy,
        combination=[1, 2, 3],
        stake_tl=stake,
        graded=True,
        hit=hit,
        payout_tl=payout if hit else 0,
        net_tl=(payout - stake) if hit else -stake,
        generated_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )
    session.add(p)
    session.flush()


def test_no_picks_returns_none(session):
    result = rolling_pnl_halt.compute(session, strategy="uclu_box6")
    assert result is None


def test_below_min_n_returns_none(session):
    for i in range(20):
        _add_pick(session, "uclu_box6", stake=10, payout=0, hit=False, days_ago=i)
    result = rolling_pnl_halt.compute(session, strategy="uclu_box6", min_n=40)
    assert result is None


def test_above_min_n_with_negative_roi_below_threshold_returns_reason(session):
    for i in range(50):
        _add_pick(session, "uclu_box6", stake=10, payout=0, hit=False, days_ago=i % 30)
    result = rolling_pnl_halt.compute(
        session, strategy="uclu_box6", min_n=40, halt_threshold=-0.10, lookback_days=30,
    )
    assert result is not None
    assert "uclu_box6" in result
    assert "ROI" in result


def test_above_min_n_with_positive_roi_returns_none(session):
    for i in range(40):
        _add_pick(session, "uclu_box6", stake=10, payout=12, hit=True, days_ago=i % 30)
    result = rolling_pnl_halt.compute(
        session, strategy="uclu_box6", min_n=40, halt_threshold=-0.10, lookback_days=30,
    )
    assert result is None


def test_only_counts_within_lookback_window(session):
    for i in range(50):
        _add_pick(session, "uclu_box6", stake=10, payout=0, hit=False, days_ago=60 + i)
    result = rolling_pnl_halt.compute(
        session, strategy="uclu_box6", min_n=40, halt_threshold=-0.10, lookback_days=30,
    )
    assert result is None


def test_only_counts_graded_picks(session):
    races = session.query(Race).all()
    for i in range(50):
        race = races[i % len(races)]
        p = Pick(
            race_id=race.id, strategy="uclu_box6", combination=[1, 2, 3], stake_tl=10,
            graded=False, generated_at=datetime.now(timezone.utc) - timedelta(days=i % 30),
        )
        session.add(p)
    session.flush()
    result = rolling_pnl_halt.compute(
        session, strategy="uclu_box6", min_n=40, halt_threshold=-0.10, lookback_days=30,
    )
    assert result is None


def test_check_all_strategies_returns_dict(session):
    for i in range(50):
        _add_pick(session, "uclu_box6", stake=10, payout=0, hit=False, days_ago=i % 30)
        _add_pick(session, "sirali_ikili_top1", stake=10, payout=12, hit=True, days_ago=i % 30)
    results = rolling_pnl_halt.check_all_strategies(
        session, strategies=("uclu_box6", "sirali_ikili_top1"),
        min_n=40, halt_threshold=-0.10, lookback_days=30,
    )
    assert results["uclu_box6"] is not None
    assert results["sirali_ikili_top1"] is None
