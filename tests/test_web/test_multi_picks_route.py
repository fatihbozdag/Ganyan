"""Route-level tests for /multi-picks: published-window collapse vs fallback.

Tests that:
- A MultiRacePool row with start_race_no set collapses the 6'lı section
  to exactly one card (the published window).
- A missing or NULL-window row keeps the existing multi-window fallback
  (every valid mathematical window is rendered).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ganyan.db.models import (
    Base,
    Horse,
    MultiRacePool,
    Race,
    RaceEntry,
    RaceStatus,
    Track,
)
from ganyan.predictor.multi_race_picks import CouponDraft
from ganyan.web.app import create_app


def _stub_draft(n_legs: int) -> CouponDraft:
    """Minimal CouponDraft that won't trip any template logic."""
    return CouponDraft(
        kept_horses_per_leg=[[1]] * n_legs,
        conviction_per_leg=[0.40] * n_legs,
        total_tickets=1,
    )


def _build_app(today: date, pool_rows: list[dict] | None = None):
    """Return a test Flask app with a 9-race program for ``today``.

    ``pool_rows`` is an optional list of dicts accepted by MultiRacePool
    (pool_type, pool_index, start_race_no, end_race_no) that are seeded
    before the session is committed.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    with factory() as session:
        track = Track(name="Test Track", city="Test")
        session.add(track)
        session.flush()

        for r in range(1, 10):  # 9-race program
            race = Race(
                track_id=track.id,
                date=today,
                race_number=r,
                distance_meters=1400,
                surface="Kum",
                status=RaceStatus.scheduled,
            )
            session.add(race)
            session.flush()
            h = Horse(name=f"Horse{r}", age=4)
            session.add(h)
            session.flush()
            session.add(RaceEntry(
                race_id=race.id,
                horse_id=h.id,
                gate_number=1,
                weight_kg=57.0,
                hp=80.0,
                kgs=14,
            ))

        for row in (pool_rows or []):
            session.add(MultiRacePool(
                date=today,
                track_id=track.id,
                **row,
            ))

        session.commit()

    app = create_app(
        session_factory=factory,
        refresh_on_launch=False,
        enable_scheduler=False,
    )
    app.config["TESTING"] = True
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiPicksWindowCollapse:
    """Published window collapses /multi-picks to one card per pool type."""

    @patch("ganyan.predictor.multi_race_picks.generate_coupon")
    def test_captured_6li_window_yields_one_coupon(self, mock_gen):
        """A 6'lı pool with published window R4-R9 renders exactly 1 coupon."""
        today = date.today()
        mock_gen.side_effect = lambda *a, **kw: _stub_draft(6)

        app = _build_app(today, pool_rows=[{
            "pool_type": "6li",
            "pool_index": 1,
            "start_race_no": 4,
            "end_race_no": 9,
            "winning_combo": "1/2/3/4/5/6",
            "payout_tl": 45230.75,
        }])

        with app.test_client() as client:
            resp = client.get(
                f"/multi-picks?date={today.isoformat()}",
                headers={"Accept": "application/json"},
            )
        assert resp.status_code == 200
        data = resp.get_json()

        # Filter to 6'lı coupons for the test track
        sixli_coupons = [
            c for c in data["coupons"]
            if c["pool_type"] == "6li" and c["track"] == "Test Track"
        ]
        assert len(sixli_coupons) == 1, (
            f"Expected 1 coupon, got {len(sixli_coupons)}: {sixli_coupons}"
        )
        coupon = sixli_coupons[0]
        assert coupon["start_race_no"] == 4
        assert coupon["end_race_no"] == 9
        assert coupon["window_source"] == "tjk"

    @patch("ganyan.predictor.multi_race_picks.generate_coupon")
    def test_captured_7li_window_yields_one_coupon(self, mock_gen):
        """A 7'lı pool with published window R2-R8 renders exactly 1 coupon."""
        today = date.today()
        mock_gen.side_effect = lambda *a, **kw: _stub_draft(7)

        app = _build_app(today, pool_rows=[{
            "pool_type": "7li",
            "pool_index": 1,
            "start_race_no": 2,
            "end_race_no": 8,
            "winning_combo": "1/2/3/4/5/6/7",
            "payout_tl": 250000.00,
        }])

        with app.test_client() as client:
            resp = client.get(
                f"/multi-picks?date={today.isoformat()}",
                headers={"Accept": "application/json"},
            )
        assert resp.status_code == 200
        data = resp.get_json()

        sevli_coupons = [
            c for c in data["coupons"]
            if c["pool_type"] == "7li" and c["track"] == "Test Track"
        ]
        assert len(sevli_coupons) == 1
        coupon = sevli_coupons[0]
        assert coupon["start_race_no"] == 2
        assert coupon["end_race_no"] == 8
        assert coupon["window_source"] == "tjk"


class TestMultiPicksWindowFallback:
    """NULL-window pool keeps the existing multi-window fallback expansion."""

    @patch("ganyan.predictor.multi_race_picks.generate_coupon")
    def test_no_pool_row_expands_all_6li_windows(self, mock_gen):
        """Without any MultiRacePool row, a 9-race program renders 4 6'lı windows."""
        today = date.today()
        mock_gen.side_effect = lambda *a, **kw: _stub_draft(6)

        app = _build_app(today, pool_rows=None)

        with app.test_client() as client:
            resp = client.get(
                f"/multi-picks?date={today.isoformat()}",
                headers={"Accept": "application/json"},
            )
        assert resp.status_code == 200
        data = resp.get_json()

        sixli_coupons = [
            c for c in data["coupons"]
            if c["pool_type"] == "6li" and c["track"] == "Test Track"
        ]
        # 9-race program: R1-R6, R2-R7, R3-R8, R4-R9 → 4 windows
        assert len(sixli_coupons) == 4
        for c in sixli_coupons:
            assert c["window_source"] == "inferred"

    @patch("ganyan.predictor.multi_race_picks.generate_coupon")
    def test_null_window_pool_row_expands_all_6li_windows(self, mock_gen):
        """A MultiRacePool row with NULL start_race_no still triggers fallback."""
        today = date.today()
        mock_gen.side_effect = lambda *a, **kw: _stub_draft(6)

        app = _build_app(today, pool_rows=[{
            "pool_type": "6li",
            "pool_index": 1,
            "start_race_no": None,
            "end_race_no": None,
            "winning_combo": "1/2/3/4/5/6",
            "payout_tl": 45000.00,
        }])

        with app.test_client() as client:
            resp = client.get(
                f"/multi-picks?date={today.isoformat()}",
                headers={"Accept": "application/json"},
            )
        assert resp.status_code == 200
        data = resp.get_json()

        sixli_coupons = [
            c for c in data["coupons"]
            if c["pool_type"] == "6li" and c["track"] == "Test Track"
        ]
        # NULL window → fallback: 9-race program has 4 valid 6'lı windows
        assert len(sixli_coupons) == 4
        for c in sixli_coupons:
            assert c["window_source"] == "inferred"

    @patch("ganyan.predictor.multi_race_picks.generate_coupon")
    def test_no_pool_row_expands_all_7li_windows(self, mock_gen):
        """Without any pool row, a 9-race program renders 3 7'lı windows."""
        today = date.today()
        mock_gen.side_effect = lambda *a, **kw: _stub_draft(7)

        app = _build_app(today, pool_rows=None)

        with app.test_client() as client:
            resp = client.get(
                f"/multi-picks?date={today.isoformat()}",
                headers={"Accept": "application/json"},
            )
        assert resp.status_code == 200
        data = resp.get_json()

        sevli_coupons = [
            c for c in data["coupons"]
            if c["pool_type"] == "7li" and c["track"] == "Test Track"
        ]
        # 9-race program: R1-R7, R2-R8, R3-R9 → 3 windows
        assert len(sevli_coupons) == 3
        for c in sevli_coupons:
            assert c["window_source"] == "inferred"
