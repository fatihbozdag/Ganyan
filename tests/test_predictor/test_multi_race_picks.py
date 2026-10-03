"""Tests for the 6'lı/5'lı/7'lı multi-race coupon generator + grader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import (
    Base, Horse, MultiRacePick, MultiRacePool, Race, RaceEntry, RaceStatus,
    Track,
)
from ganyan.predictor.multi_race_picks import (
    CouponDraft, _leg_width, _parse_winning_combo, _product, generate_coupon,
    grade_all_pending_multi, grade_pick, multi_strategy_summary, persist_coupon,
)


# ---------------------------------------------------------------------------
# Pure-function tests — no DB, no mocking
# ---------------------------------------------------------------------------


def test_leg_width_lock_tier():
    """0.50+ top-1 prob → lock width=1."""
    assert _leg_width(0.50) == 1
    assert _leg_width(0.75) == 1
    assert _leg_width(0.99) == 1


def test_leg_width_medium_tiers():
    assert _leg_width(0.49) == 2
    assert _leg_width(0.35) == 2
    assert _leg_width(0.34) == 3
    assert _leg_width(0.25) == 3
    assert _leg_width(0.24) == 4
    assert _leg_width(0.18) == 4


def test_leg_width_spread_tier():
    """Below 0.18 the model is barely above uniform — spread to 5."""
    assert _leg_width(0.17) == 5
    assert _leg_width(0.10) == 5
    assert _leg_width(0.0) == 5


def test_leg_width_monotonic():
    """Width should never increase with conviction."""
    probs = [0.0, 0.05, 0.10, 0.18, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75]
    widths = [_leg_width(p) for p in probs]
    assert widths == sorted(widths, reverse=True)


def test_parse_winning_combo_simple():
    assert _parse_winning_combo("1/4/2/1/3/4") == [[1], [4], [2], [1], [3], [4]]


def test_parse_winning_combo_dead_heat():
    """Commas inside a leg = dead-heat alternatives (any winner counts)."""
    assert _parse_winning_combo("1/4/2/1/3,6,7,10/4,11") == [
        [1], [4], [2], [1], [3, 6, 7, 10], [4, 11],
    ]


def test_parse_winning_combo_5li():
    assert _parse_winning_combo("3/1,12/4/2/5") == [[3], [1, 12], [4], [2], [5]]


def test_parse_winning_combo_7li():
    combo = "1/2/3/4/5/6/7"
    parsed = _parse_winning_combo(combo)
    assert len(parsed) == 7
    assert parsed[0] == [1]
    assert parsed[6] == [7]


def test_product():
    assert _product([[1], [1, 2], [1, 2, 3]]) == 6
    assert _product([[1]] * 6) == 1
    assert _product([[1, 2, 3], [1, 2], [1, 2, 3, 4], [1], [1, 2], [1]]) == 48


# ---------------------------------------------------------------------------
# DB-backed tests with fake predictor
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@dataclass
class _FakePred:
    horse_id: int
    mean_probability: float  # already in 0-100 percent like the real one


class _FakePredictor:
    """Stub for EnsemblePredictor — returns canned predictions per race."""

    def __init__(self, by_race: dict[int, list[_FakePred]]):
        self.by_race = by_race

    def predict(self, race_id: int):
        return self.by_race.get(race_id, [])


def _build_program(
    session, target: date, track_name: str, n_races: int, n_horses: int,
):
    """Create a track + N races with M horses each, returning (track, races)."""
    track = Track(name=track_name, city=track_name, surface_types=["çim"])
    session.add(track)
    session.flush()

    races = []
    horses = []
    for h_idx in range(n_horses):
        horse = Horse(name=f"Horse{h_idx+1}", age=4, origin="TR")
        session.add(horse)
    session.flush()
    horses = session.query(Horse).all()

    for r_idx in range(n_races):
        race = Race(
            track_id=track.id,
            date=target,
            race_number=r_idx + 1,
            distance_meters=1400,
            surface="çim",
            status=RaceStatus.scheduled,
        )
        session.add(race)
        session.flush()
        for h_idx, horse in enumerate(horses):
            entry = RaceEntry(
                race_id=race.id,
                horse_id=horse.id,
                gate_number=h_idx + 1,
            )
            session.add(entry)
        races.append(race)
    session.flush()
    return track, races, horses


def test_generate_coupon_lock_legs_only(db_session):
    """Six high-conviction (>=0.50) legs all lock to width=1, total=1."""
    target = date(2026, 5, 3)
    track, races, horses = _build_program(db_session, target, "Ankara", 6, 8)
    by_race = {
        race.id: [
            _FakePred(horse_id=horses[0].id, mean_probability=60.0),
            _FakePred(horse_id=horses[1].id, mean_probability=20.0),
            _FakePred(horse_id=horses[2].id, mean_probability=10.0),
            _FakePred(horse_id=horses[3].id, mean_probability=5.0),
        ]
        for race in races
    }
    pred = _FakePredictor(by_race)

    draft = generate_coupon(
        db_session, target, "Ankara", 1, pool_type="6li", predictor=pred,
    )
    assert draft.total_tickets == 1
    assert all(len(leg) == 1 for leg in draft.kept_horses_per_leg)
    # All legs picked horse 0, gate=1.
    assert draft.kept_horses_per_leg == [[1]] * 6
    # Top-1 prob 0.60 is way above the 0.50 lock threshold.
    assert all(c >= 0.50 for c in draft.conviction_per_leg)


def test_generate_coupon_mixed_tiers_under_budget(db_session):
    """Mixed conviction → mixed widths; product stays under cap so no shrink."""
    target = date(2026, 5, 3)
    track, races, horses = _build_program(db_session, target, "Bursa", 6, 11)

    # Convictions: [60, 40, 30, 20, 15, 60] → widths [1, 2, 3, 4, 5, 1] = 120
    convictions = [60.0, 40.0, 30.0, 20.0, 15.0, 60.0]
    by_race = {}
    for race, conv in zip(races, convictions):
        # Build a descending probability list.
        probs = [conv] + [
            (100.0 - conv) / (len(horses) - 1) for _ in horses[1:]
        ]
        by_race[race.id] = [
            _FakePred(horse_id=h.id, mean_probability=p)
            for h, p in zip(horses, probs)
        ]
    pred = _FakePredictor(by_race)

    draft = generate_coupon(
        db_session, target, "Bursa", 1, pool_type="6li",
        max_tickets=512, predictor=pred,
    )
    widths = [len(leg) for leg in draft.kept_horses_per_leg]
    assert widths == [1, 2, 3, 4, 5, 1]
    assert draft.total_tickets == 1 * 2 * 3 * 4 * 5 * 1


def test_generate_coupon_budget_cap_shrinks_widest_leg(db_session):
    """Budget cap forces the widest+lowest-conviction leg to shrink."""
    target = date(2026, 5, 3)
    track, races, horses = _build_program(db_session, target, "İzmir", 6, 11)

    # Six all-spread legs (conviction <0.18) → 5×5×5×5×5×5 = 15,625 unconstrained.
    convictions = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    by_race = {}
    for race, conv in zip(races, convictions):
        probs = [conv] + [
            (100.0 - conv) / (len(horses) - 1) for _ in horses[1:]
        ]
        by_race[race.id] = [
            _FakePred(horse_id=h.id, mean_probability=p)
            for h, p in zip(horses, probs)
        ]
    pred = _FakePredictor(by_race)

    draft = generate_coupon(
        db_session, target, "İzmir", 1, pool_type="6li",
        max_tickets=64, predictor=pred,
    )
    assert draft.total_tickets <= 64
    # All legs should still have at least 1 horse.
    assert all(len(leg) >= 1 for leg in draft.kept_horses_per_leg)


def test_generate_coupon_unknown_track_raises(db_session):
    target = date(2026, 5, 3)
    pred = _FakePredictor({})
    with pytest.raises(ValueError, match="unknown track"):
        generate_coupon(
            db_session, target, "AtlantisOnTheSea", 1,
            pool_type="6li", predictor=pred,
        )


def test_generate_coupon_missing_races_raises(db_session):
    target = date(2026, 5, 3)
    track, races, horses = _build_program(db_session, target, "Adana", 3, 8)
    pred = _FakePredictor({})
    with pytest.raises(ValueError, match="expected 6 races"):
        generate_coupon(
            db_session, target, "Adana", 1, pool_type="6li", predictor=pred,
        )


# ---------------------------------------------------------------------------
# persist_coupon — idempotent upsert
# ---------------------------------------------------------------------------


def test_persist_coupon_is_immutable(db_session):
    from datetime import timedelta
    target = date.today() + timedelta(days=1)
    track, races, horses = _build_program(db_session, target, "Kocaeli", 6, 10)
    for race in races:
        race.post_time = "14:00"
    draft = CouponDraft([[1]] * 6, [0.6] * 6, 1)
    pick = persist_coupon(db_session, target, "Kocaeli", 1, draft, pool_index=1)
    assert persist_coupon(db_session, target, "Kocaeli", 1, draft, pool_index=1).id == pick.id
    pick.graded, pick.hit, pick.payout_tl = True, True, 100
    with pytest.raises(ValueError, match="immutable"):
        persist_coupon(db_session, target, "Kocaeli", 1,
                       CouponDraft([[2]] * 6, [0.6] * 6, 1), pool_index=1)
    assert pick.graded and pick.payout_tl == 100


# ---------------------------------------------------------------------------
# grade_pick — boolean AND across legs, dead-heat handling
# ---------------------------------------------------------------------------


def _make_pick(
    db_session, *, target: date, track: Track,
    kept: list[list[int]], stake_tl: float = 6.0,
    ticket_unit_tl: float = 1.0,
) -> MultiRacePick:
    pick = MultiRacePick(
        date=target, track_id=track.id,
        pool_type="6li", pool_index=1, strategy="asymmetric_v1",
        start_race_no=1, end_race_no=6,
        kept_horses_per_leg=kept,
        total_tickets=len(kept) and 1 or 0,  # filled below
        ticket_unit_tl=ticket_unit_tl, stake_tl=stake_tl,
        conviction_per_leg=[0.5] * 6,
    )
    pick.total_tickets = 1
    for leg in kept:
        pick.total_tickets *= len(leg)
    db_session.add(pick)
    db_session.flush()
    return pick


def test_grade_pick_all_legs_hit(db_session):
    target = date(2026, 5, 3)
    track = Track(name="Şanlıurfa", city="Şanlıurfa", surface_types=["kum"])
    db_session.add(track)
    db_session.flush()

    pool = MultiRacePool(
        date=target, track_id=track.id, pool_type="6li", pool_index=1, start_race_no=1, end_race_no=6,
        winning_combo="1/4/2/1/3/7", payout_tl=1000.0,
    )
    db_session.add(pool)
    db_session.flush()

    pick = _make_pick(
        db_session, target=target, track=track,
        kept=[[1], [4, 5], [2], [1, 8], [3], [7]],
    )
    result = grade_pick(db_session, pick)
    assert result is True
    assert pick.hit is True
    assert pick.payout_tl == pytest.approx(1000.0)
    assert pick.net_tl == pytest.approx(1000.0 - float(pick.stake_tl))
    assert pick.graded is True
    assert pick.graded_at is not None


def test_grade_pick_one_leg_missed(db_session):
    target = date(2026, 5, 3)
    track = Track(name="Diyarbakır", city="Diyarbakır", surface_types=["kum"])
    db_session.add(track)
    db_session.flush()

    pool = MultiRacePool(
        date=target, track_id=track.id, pool_type="6li", pool_index=1, start_race_no=1, end_race_no=6,
        winning_combo="1/4/2/1/3/7", payout_tl=1000.0,
    )
    db_session.add(pool)
    db_session.flush()

    pick = _make_pick(
        db_session, target=target, track=track,
        kept=[[1], [4, 5], [2], [1, 8], [3], [9]],  # leg 6 misses (kept 9, won 7)
    )
    result = grade_pick(db_session, pick)
    assert result is False
    assert pick.hit is False
    assert pick.payout_tl == pytest.approx(0.0)
    assert pick.net_tl == pytest.approx(-float(pick.stake_tl))


def test_grade_pick_dead_heat_doubles_payout(db_session):
    """If we kept BOTH dead-heat alternates in a leg, payout multiplies."""
    target = date(2026, 5, 3)
    track = Track(name="İstanbul", city="İstanbul", surface_types=["çim"])
    db_session.add(track)
    db_session.flush()

    # Leg 5 has FOUR dead-heat winners; leg 6 has TWO.
    pool = MultiRacePool(
        date=target, track_id=track.id, pool_type="6li", pool_index=1, start_race_no=1, end_race_no=6,
        winning_combo="1/4/2/1/3,6,7,10/4,11", payout_tl=52842.87,
    )
    db_session.add(pool)
    db_session.flush()

    # Kept 2 of 4 in leg 5 (3, 6) and 1 of 2 in leg 6 (4) → multiplier 2*1 = 2.
    pick = _make_pick(
        db_session, target=target, track=track,
        kept=[[1], [4], [2], [1], [3, 6], [4]],
    )
    result = grade_pick(db_session, pick)
    assert result is True
    assert pick.payout_tl == pytest.approx(52842.87 * 2)


def test_grade_pick_no_pool_returns_none(db_session):
    target = date(2026, 5, 3)
    track = Track(name="Antalya", city="Antalya", surface_types=["kum"])
    db_session.add(track)
    db_session.flush()
    pick = _make_pick(
        db_session, target=target, track=track,
        kept=[[1], [2], [3], [4], [5], [6]],
    )
    assert grade_pick(db_session, pick) is None
    assert pick.graded is False


def test_grade_pick_unresulted_pool_returns_none(db_session):
    target = date(2026, 5, 3)
    track = Track(name="Elazığ", city="Elazığ", surface_types=["kum"])
    db_session.add(track)
    db_session.flush()
    pool = MultiRacePool(
        date=target, track_id=track.id, pool_type="6li", pool_index=1, start_race_no=1, end_race_no=6,
        winning_combo=None, payout_tl=None,
    )
    db_session.add(pool)
    db_session.flush()

    pick = _make_pick(
        db_session, target=target, track=track,
        kept=[[1], [2], [3], [4], [5], [6]],
    )
    assert grade_pick(db_session, pick) is None


def test_grade_pick_ticket_unit_scales_payout(db_session):
    """Payout = pool.payout_tl * pick.ticket_unit_tl * n_winning_combos."""
    target = date(2026, 5, 3)
    track = Track(name="İzmir", city="İzmir", surface_types=["kum"])
    db_session.add(track)
    db_session.flush()
    pool = MultiRacePool(
        date=target, track_id=track.id, pool_type="6li", pool_index=1, start_race_no=1, end_race_no=6,
        winning_combo="1/2/3/4/5/6", payout_tl=100.0,
    )
    db_session.add(pool)
    db_session.flush()

    pick = _make_pick(
        db_session, target=target, track=track,
        kept=[[1], [2], [3], [4], [5], [6]],
        ticket_unit_tl=2.5, stake_tl=2.5,
    )
    grade_pick(db_session, pick)
    assert pick.payout_tl == pytest.approx(250.0)  # 100 * 2.5 * 1


# ---------------------------------------------------------------------------
# grade_all_pending_multi + multi_strategy_summary — batch grading + ROI agg
# ---------------------------------------------------------------------------


def test_grade_all_pending_multi_only_grades_resulted_pools(db_session):
    target = date(2026, 5, 3)
    track = Track(name="Bursa", city="Bursa", surface_types=["çim"])
    db_session.add(track)
    db_session.flush()

    # Pool A: resulted.
    db_session.add(MultiRacePool(
        date=target, track_id=track.id, pool_type="6li", pool_index=1, start_race_no=1, end_race_no=6,
        winning_combo="1/2/3/4/5/6", payout_tl=500.0,
    ))
    # Pool B: same date but pool_index=2, NOT resulted.
    db_session.add(MultiRacePool(
        date=target, track_id=track.id, pool_type="6li", pool_index=2,
        winning_combo=None, payout_tl=None,
    ))
    db_session.flush()

    # Pick A (matches pool A): hits.
    pick_a = MultiRacePick(
        date=target, track_id=track.id, pool_type="6li", pool_index=1,
        strategy="asymmetric_v1", start_race_no=1, end_race_no=6,
        kept_horses_per_leg=[[1], [2], [3], [4], [5], [6]],
        total_tickets=1, ticket_unit_tl=1.0, stake_tl=1.0,
        conviction_per_leg=[0.6] * 6,
    )
    # Pick B (matches pool B): pool not resulted yet, should stay ungraded.
    pick_b = MultiRacePick(
        date=target, track_id=track.id, pool_type="6li", pool_index=2,
        strategy="asymmetric_v1", start_race_no=4, end_race_no=9,
        kept_horses_per_leg=[[1], [2], [3], [4], [5], [6]],
        total_tickets=1, ticket_unit_tl=1.0, stake_tl=1.0,
        conviction_per_leg=[0.6] * 6,
    )
    db_session.add(pick_a)
    db_session.add(pick_b)
    db_session.flush()

    n = grade_all_pending_multi(db_session)
    assert n == 1
    db_session.refresh(pick_a)
    db_session.refresh(pick_b)
    assert pick_a.graded is True
    assert pick_a.hit is True
    assert pick_b.graded is False


def test_multi_strategy_summary_aggregates_roi(db_session):
    target = date(2026, 5, 3)
    track = Track(name="Adana", city="Adana", surface_types=["kum"])
    db_session.add(track)
    db_session.flush()

    # Two graded picks: one hit (paid 1000), one miss.
    db_session.add(MultiRacePick(
        date=target, track_id=track.id, pool_type="6li", pool_index=1,
        strategy="asymmetric_v1", start_race_no=1, end_race_no=6,
        kept_horses_per_leg=[[1]] * 6,
        total_tickets=1, ticket_unit_tl=1.0, stake_tl=1.0,
        conviction_per_leg=[0.6] * 6,
        graded=True, hit=True, payout_tl=1000.0, net_tl=999.0,
    ))
    db_session.add(MultiRacePick(
        date=target, track_id=track.id, pool_type="6li", pool_index=2,
        strategy="asymmetric_v1", start_race_no=4, end_race_no=9,
        kept_horses_per_leg=[[1]] * 6,
        total_tickets=1, ticket_unit_tl=1.0, stake_tl=1.0,
        conviction_per_leg=[0.6] * 6,
        graded=True, hit=False, payout_tl=0.0, net_tl=-1.0,
    ))
    # Plus one ungraded — should be skipped.
    db_session.add(MultiRacePick(
        date=target, track_id=track.id, pool_type="6li", pool_index=3,
        strategy="asymmetric_v1", start_race_no=7, end_race_no=12,
        kept_horses_per_leg=[[1]] * 6,
        total_tickets=1, ticket_unit_tl=1.0, stake_tl=1.0,
        conviction_per_leg=[0.6] * 6,
        graded=False,
    ))
    db_session.flush()

    summary = multi_strategy_summary(db_session)
    assert "asymmetric_v1" in summary
    row = summary["asymmetric_v1"]
    assert row["n"] == 2
    assert row["hits"] == 1
    assert row["stake_tl"] == pytest.approx(2.0)
    assert row["payout_tl"] == pytest.approx(1000.0)
    assert row["net_tl"] == pytest.approx(998.0)
    assert row["hit_rate_pct"] == pytest.approx(50.0)
    assert row["roi_pct"] == pytest.approx(998.0 / 2.0 * 100)
