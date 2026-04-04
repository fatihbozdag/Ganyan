"""Flask blueprint with all web routes for Ganyan."""

from __future__ import annotations

from datetime import date, datetime

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    render_template,
    request,
)
from sqlalchemy.orm import Session

from ganyan.db.models import Race, RaceStatus

bp = Blueprint("main", __name__)


def _get_session() -> Session:
    """Obtain a new database session from the app-level factory."""
    factory = current_app.config["SESSION_FACTORY"]
    return factory()


def _wants_json() -> bool:
    """Return True when the client prefers JSON over HTML."""
    best = request.accept_mimetypes.best_match(
        ["application/json", "text/html"]
    )
    return best == "application/json"


# ---------------------------------------------------------------------------
# GET / — Dashboard
# ---------------------------------------------------------------------------


@bp.route("/")
def index():
    session = _get_session()
    try:
        today_races = (
            session.query(Race)
            .filter(Race.date == date.today())
            .order_by(Race.race_number)
            .all()
        )
        recent_races = (
            session.query(Race)
            .filter(Race.status == RaceStatus.resulted)
            .order_by(Race.date.desc(), Race.race_number.desc())
            .limit(10)
            .all()
        )
        return render_template(
            "index.html",
            today_races=today_races,
            recent_races=recent_races,
        )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# GET /races/<date> — Races for a given date
# ---------------------------------------------------------------------------


@bp.route("/races/<race_date>")
def races_by_date(race_date: str):
    try:
        target_date = datetime.strptime(race_date, "%Y-%m-%d").date()
    except ValueError:
        abort(400)

    session = _get_session()
    try:
        race_list = (
            session.query(Race)
            .filter(Race.date == target_date)
            .order_by(Race.race_number)
            .all()
        )

        if _wants_json():
            return jsonify(
                [
                    {
                        "id": r.id,
                        "track": r.track.name if r.track else None,
                        "race_number": r.race_number,
                        "distance_meters": r.distance_meters,
                        "surface": r.surface,
                        "entry_count": len(r.entries),
                        "status": r.status.value,
                    }
                    for r in race_list
                ]
            )

        return render_template(
            "races.html",
            races=race_list,
            race_date=target_date,
        )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# GET /races/<race_id>/predict — Prediction results
# ---------------------------------------------------------------------------


@bp.route("/races/<int:race_id>/predict")
def predict_race(race_id: int):
    session = _get_session()
    try:
        race = session.get(Race, race_id)
        if race is None:
            if _wants_json():
                return jsonify({"error": "Race not found"}), 404
            abort(404)

        from ganyan.predictor import BayesianPredictor

        predictor = BayesianPredictor(session)
        predictions = predictor.predict(race_id)

        if _wants_json():
            return jsonify(
                {
                    "race_id": race_id,
                    "predictions": [
                        {
                            "horse_id": p.horse_id,
                            "horse_name": p.horse_name,
                            "probability": round(p.probability, 2),
                            "confidence": round(p.confidence, 2),
                            "contributing_factors": p.contributing_factors,
                        }
                        for p in predictions
                    ],
                }
            )

        return render_template(
            "predict.html",
            race=race,
            predictions=predictions,
        )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# GET /history — Past resulted races
# ---------------------------------------------------------------------------


@bp.route("/history")
def history():
    session = _get_session()
    try:
        resulted_races = (
            session.query(Race)
            .filter(Race.status == RaceStatus.resulted)
            .order_by(Race.date.desc(), Race.race_number.desc())
            .limit(50)
            .all()
        )

        if _wants_json():
            return jsonify(
                [
                    {
                        "id": r.id,
                        "track": r.track.name if r.track else None,
                        "date": r.date.isoformat(),
                        "race_number": r.race_number,
                        "status": r.status.value,
                    }
                    for r in resulted_races
                ]
            )

        return render_template("history.html", races=resulted_races)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# POST /scrape/today — Trigger scrape
# ---------------------------------------------------------------------------


@bp.route("/scrape/today", methods=["POST"])
def scrape_today():
    import asyncio

    from ganyan.config import get_settings
    from ganyan.db.models import ScrapeStatus
    from ganyan.scraper import TJKClient, parse_race_card
    from ganyan.scraper.backfill import log_scrape, store_race_card

    settings = get_settings()
    session = _get_session()

    try:

        async def _do_scrape():
            async with TJKClient(
                base_url=settings.tjk_base_url, delay=settings.scrape_delay
            ) as client:
                raw_cards = await client.get_race_card(date.today())
                return raw_cards

        raw_cards = asyncio.run(_do_scrape())

        if not raw_cards:
            msg = "Bugün için yarış kartı bulunamadı."
            if _wants_json():
                return jsonify({"message": msg, "count": 0})
            return render_template("index.html", today_races=[], recent_races=[], message=msg)

        for raw in raw_cards:
            parsed = parse_race_card(raw)
            store_race_card(session, parsed)
            log_scrape(session, date.today(), parsed.track_name, ScrapeStatus.success)
        session.commit()

        msg = f"{len(raw_cards)} yarış kartı kaydedildi."
        if _wants_json():
            return jsonify({"message": msg, "count": len(raw_cards)})

        # Reload today's races for the template
        today_races = (
            session.query(Race)
            .filter(Race.date == date.today())
            .order_by(Race.race_number)
            .all()
        )
        return render_template(
            "index.html",
            today_races=today_races,
            recent_races=[],
            message=msg,
        )

    except Exception as exc:
        session.rollback()
        msg = f"Hata: {exc}"
        if _wants_json():
            return jsonify({"error": msg}), 500
        return render_template(
            "index.html",
            today_races=[],
            recent_races=[],
            message=msg,
        ), 500
    finally:
        session.close()
