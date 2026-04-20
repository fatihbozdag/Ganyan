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
    """Return True when the client explicitly prefers JSON over HTML."""
    best = request.accept_mimetypes.best_match(
        ["text/html", "application/json"]
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
    from ganyan.predictor.evaluate import evaluate_all

    session = _get_session()
    try:
        summary, evaluations = evaluate_all(session)

        # Also fetch the full race list for any races without predictions.
        resulted_races = (
            session.query(Race)
            .filter(Race.status == RaceStatus.resulted)
            .order_by(Race.date.desc(), Race.race_number.desc())
            .limit(50)
            .all()
        )

        if _wants_json():
            return jsonify(
                {
                    "summary": {
                        "total_races": summary.total_races,
                        "top1_accuracy": round(summary.top1_accuracy, 2),
                        "top3_accuracy": round(summary.top3_accuracy, 2),
                        "avg_winner_rank": round(summary.avg_winner_rank, 2),
                        "avg_winner_probability": round(
                            summary.avg_winner_probability, 2
                        ),
                        "log_loss": round(summary.log_loss, 4),
                        "roi_simulation": round(summary.roi_simulation, 4),
                    },
                    "evaluations": [
                        {
                            "race_id": ev.race_id,
                            "track": ev.track,
                            "date": ev.date.isoformat(),
                            "race_number": ev.race_number,
                            "num_horses": ev.num_horses,
                            "winner_name": ev.winner_name,
                            "winner_predicted_prob": (
                                round(ev.winner_predicted_prob, 2)
                                if ev.winner_predicted_prob is not None
                                else None
                            ),
                            "winner_predicted_rank": ev.winner_predicted_rank,
                            "top1_correct": ev.top1_correct,
                            "top3_correct": ev.top3_correct,
                        }
                        for ev in evaluations
                    ],
                    "races": [
                        {
                            "id": r.id,
                            "track": r.track.name if r.track else None,
                            "date": r.date.isoformat(),
                            "race_number": r.race_number,
                            "status": r.status.value,
                        }
                        for r in resulted_races
                    ],
                }
            )

        return render_template(
            "history.html",
            races=resulted_races,
            summary=summary,
            evaluations=evaluations,
        )
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

    except Exception as exc:  # noqa: BLE001
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


# ---------------------------------------------------------------------------
# POST /predict/today — Predict all today's races and save
# ---------------------------------------------------------------------------


@bp.route("/predict/today", methods=["POST"])
def predict_today():
    from ganyan.predictor import BayesianPredictor

    session = _get_session()
    try:
        today_races = (
            session.query(Race)
            .filter(Race.date == date.today())
            .order_by(Race.race_number)
            .all()
        )

        if not today_races:
            msg = "Bugün için yarış bulunamadı."
            if _wants_json():
                return jsonify({"message": msg, "count": 0})
            return render_template(
                "index.html", today_races=[], recent_races=[], message=msg,
            )

        predictor = BayesianPredictor(session)
        count = 0
        for race in today_races:
            predictor.predict_and_save(race.id)
            count += 1
        session.commit()

        msg = f"{count} yarış için tahmin kaydedildi."
        if _wants_json():
            return jsonify({"message": msg, "count": count})

        # Reload for template
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

    except Exception as exc:  # noqa: BLE001
        session.rollback()
        msg = f"Hata: {exc}"
        if _wants_json():
            return jsonify({"error": msg}), 500
        return render_template(
            "index.html", today_races=[], recent_races=[], message=msg,
        ), 500
    finally:
        session.close()


# ---------------------------------------------------------------------------
# POST /scrape/history — Load historical data via KosuSorgulama
# ---------------------------------------------------------------------------


@bp.route("/scrape/history", methods=["POST"])
def scrape_history():
    import asyncio

    from ganyan.config import get_settings
    from ganyan.scraper import TJKClient
    from ganyan.scraper.backfill import BackfillManager

    settings = get_settings()
    session = _get_session()

    from_str = request.form.get("from_date", "")
    to_str = request.form.get("to_date", "")

    if not from_str or not to_str:
        msg = "Baslangic ve bitis tarihi gerekli."
        if _wants_json():
            return jsonify({"error": msg}), 400
        return render_template(
            "index.html", today_races=[], recent_races=[], message=msg,
        ), 400

    try:
        from_date = datetime.strptime(from_str, "%Y-%m-%d").date()
        to_date = datetime.strptime(to_str, "%Y-%m-%d").date()
    except ValueError:
        msg = "Tarih formati hatali. YYYY-MM-DD olmali."
        if _wants_json():
            return jsonify({"error": msg}), 400
        return render_template(
            "index.html", today_races=[], recent_races=[], message=msg,
        ), 400

    try:

        async def _do_history():
            async with TJKClient(
                base_url=settings.tjk_base_url, delay=settings.scrape_delay
            ) as client:
                manager = BackfillManager(session, client)
                return await manager.backfill_historical(from_date, to_date)

        count = asyncio.run(_do_history())

        msg = f"{count} gecmis yaris kaydi yuklendi ({from_date} -> {to_date})."
        if _wants_json():
            return jsonify({"message": msg, "count": count})

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
            message=msg,
        )

    except Exception as exc:  # noqa: BLE001
        session.rollback()
        msg = f"Hata: {exc}"
        if _wants_json():
            return jsonify({"error": msg}), 500
        return render_template(
            "index.html", today_races=[], recent_races=[], message=msg,
        ), 500
    finally:
        session.close()


# ---------------------------------------------------------------------------
# POST /scrape/results — Fetch today's results from TJK
# ---------------------------------------------------------------------------


@bp.route("/scrape/results", methods=["POST"])
def scrape_results():
    import asyncio

    from ganyan.config import get_settings
    from ganyan.scraper import TJKClient, parse_race_card
    from ganyan.scraper.backfill import update_race_results

    settings = get_settings()
    session = _get_session()

    try:

        async def _do_scrape():
            async with TJKClient(
                base_url=settings.tjk_base_url, delay=settings.scrape_delay
            ) as client:
                return await client.get_race_results(date.today())

        raw_results = asyncio.run(_do_scrape())

        if not raw_results:
            msg = "Bugün için sonuç bulunamadı."
            if _wants_json():
                return jsonify({"message": msg, "count": 0})
            return render_template(
                "index.html", today_races=[], recent_races=[], message=msg,
            )

        updated = 0
        for raw in raw_results:
            parsed = parse_race_card(raw)
            result = update_race_results(session, parsed)
            if result:
                updated += 1
        session.commit()

        msg = f"{updated} yarış sonucu güncellendi."
        if _wants_json():
            return jsonify({"message": msg, "count": updated})

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
            message=msg,
        )

    except Exception as exc:  # noqa: BLE001
        session.rollback()
        msg = f"Hata: {exc}"
        if _wants_json():
            return jsonify({"error": msg}), 500
        return render_template(
            "index.html", today_races=[], recent_races=[], message=msg,
        ), 500
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Ops dashboard + health endpoint
# ---------------------------------------------------------------------------

_FRESHNESS_BUDGET = {
    # Hours before a data-freshness signal turns "stale" on the dashboard.
    "today_card": 24,        # today's race card should have been scraped
    "last_results": 12,      # most recent result should be <12h old
    "last_prediction": 24,   # at least one prediction written in 24h
}


@bp.route("/ops")
def ops_dashboard():
    """Show recent scheduled-job runs + data-freshness health."""
    from ganyan.db.models import JobRun, Prediction, Race, RaceEntry
    from sqlalchemy import desc, func

    session = _get_session()
    try:
        recent_runs = (
            session.query(JobRun)
            .order_by(desc(JobRun.started_at))
            .limit(50)
            .all()
        )
        # Aggregate last run per job_id.
        by_job: dict[str, JobRun] = {}
        for r in recent_runs:
            if r.job_id not in by_job:
                by_job[r.job_id] = r

        last_scrape = session.query(func.max(Race.date)).scalar()
        last_result_date = (
            session.query(func.max(Race.date))
            .join(RaceEntry)
            .filter(RaceEntry.finish_position.isnot(None))
            .scalar()
        )
        last_prediction_at = (
            session.query(func.max(Prediction.predicted_at)).scalar()
        )
        failure_count_24h = (
            session.query(func.count(JobRun.id))
            .filter(
                JobRun.status == "failed",
                JobRun.started_at >= datetime.utcnow().replace(
                    hour=0, minute=0, second=0, microsecond=0,
                ),
            )
            .scalar()
        ) or 0

        health = _compute_health(
            last_scrape, last_result_date, last_prediction_at, failure_count_24h,
        )

        if _wants_json():
            return jsonify({
                "health": health,
                "last_scrape": last_scrape.isoformat() if last_scrape else None,
                "last_result_date": (
                    last_result_date.isoformat() if last_result_date else None
                ),
                "last_prediction_at": (
                    last_prediction_at.isoformat() if last_prediction_at else None
                ),
                "failure_count_24h": failure_count_24h,
                "jobs": [
                    {
                        "job_id": jid,
                        "status": r.status,
                        "started_at": r.started_at.isoformat(),
                        "duration_ms": r.duration_ms,
                        "error_message": r.error_message,
                    }
                    for jid, r in sorted(by_job.items())
                ],
                "recent_runs": [
                    {
                        "job_id": r.job_id,
                        "status": r.status,
                        "started_at": r.started_at.isoformat(),
                        "duration_ms": r.duration_ms,
                        "error_message": r.error_message,
                    }
                    for r in recent_runs
                ],
            })

        return render_template(
            "ops.html",
            health=health,
            by_job=by_job,
            recent_runs=recent_runs,
            last_scrape=last_scrape,
            last_result_date=last_result_date,
            last_prediction_at=last_prediction_at,
            failure_count_24h=failure_count_24h,
        )
    finally:
        session.close()


@bp.route("/ops/health")
def ops_health():
    """Lightweight JSON endpoint for external monitors (cron pings, UptimeRobot)."""
    from ganyan.db.models import JobRun, Race, RaceEntry
    from sqlalchemy import func

    session = _get_session()
    try:
        last_scrape = session.query(func.max(Race.date)).scalar()
        last_result = (
            session.query(func.max(Race.date))
            .join(RaceEntry)
            .filter(RaceEntry.finish_position.isnot(None))
            .scalar()
        )
        failures = (
            session.query(func.count(JobRun.id))
            .filter(JobRun.status == "failed")
            .filter(JobRun.started_at >= datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0,
            ))
            .scalar()
        ) or 0
        health = _compute_health(last_scrape, last_result, None, failures)
        status_code = 200 if health["status"] == "ok" else 503
        return jsonify({
            "status": health["status"],
            "reasons": health["reasons"],
            "failures_24h": failures,
            "last_scrape": last_scrape.isoformat() if last_scrape else None,
            "last_result_date": last_result.isoformat() if last_result else None,
        }), status_code
    finally:
        session.close()


def _compute_health(
    last_scrape, last_result_date, last_prediction_at, failure_count_24h,
) -> dict:
    """Build a small status payload: ok / warn / fail + reasons."""
    today = date.today()
    reasons: list[str] = []

    if last_scrape is None or (today - last_scrape).days > 1:
        reasons.append("no race scraped today or yesterday")
    if last_result_date is not None and (today - last_result_date).days > 1:
        reasons.append("no race results pulled in 48h")
    if failure_count_24h > 0:
        reasons.append(f"{failure_count_24h} scheduled-job failure(s) today")

    if not reasons:
        status = "ok"
    elif failure_count_24h > 2:
        status = "fail"
    else:
        status = "warn"

    return {"status": status, "reasons": reasons}


# ---------------------------------------------------------------------------
# Live betting sheet — picks + actuals + rolling daily P&L
# ---------------------------------------------------------------------------


@bp.route("/live")
def live_sheet():
    """One page per day: our picks, actuals, hit/miss, rolling P&L.

    Auto-refreshes every 30s so the picks appear before the race and
    fill in outcomes as results come in.
    """
    from ganyan.db.models import Race, RaceEntry, RaceStatus
    from ganyan.predictor.exotics import (
        ganyan_probabilities, ikili_probabilities,
        sirali_ikili_probabilities, uclu_probabilities,
    )

    target_str = request.args.get("date")
    try:
        target = (
            datetime.strptime(target_str, "%Y-%m-%d").date()
            if target_str else date.today()
        )
    except ValueError:
        target = date.today()

    session = _get_session()
    try:
        races = (
            session.query(Race)
            .filter(Race.date == target)
            .order_by(Race.post_time.asc().nullslast(), Race.race_number.asc())
            .all()
        )

        rows: list[dict] = []
        # Pool → (races_staked, hits, stake, payout)
        tally = {p: [0, 0, 0.0, 0.0] for p in ("ganyan", "ikili", "sirali_ikili", "uclu")}
        STAKE = 100.0

        for race in races:
            entries = list(race.entries)
            name_for = {e.horse_id: (e.horse.name if e.horse else "?") for e in entries}
            agf_rank_by_id = {}
            agf_ranked = sorted(
                [e for e in entries if e.agf is not None],
                key=lambda e: float(e.agf), reverse=True,
            )
            for i, e in enumerate(agf_ranked):
                agf_rank_by_id[e.horse_id] = i + 1

            # Win probabilities from stored predicted_probability.
            win_probs = {
                e.horse_id: float(e.predicted_probability) / 100.0
                for e in entries if e.predicted_probability is not None
            }
            if not win_probs:
                rows.append({
                    "race": race, "pending": True, "picks": {}, "actual": None,
                    "agf_rank_by_id": agf_rank_by_id, "name_for": name_for,
                })
                continue

            picks = {
                "ganyan": ganyan_probabilities(win_probs)[:1],
                "ikili": ikili_probabilities(win_probs)[:1] if len(win_probs) >= 2 else [],
                "sirali_ikili": sirali_ikili_probabilities(win_probs)[:1] if len(win_probs) >= 2 else [],
                "uclu": uclu_probabilities(win_probs)[:1] if len(win_probs) >= 3 else [],
            }

            winners = sorted(
                [e for e in entries if e.finish_position in (1, 2, 3)],
                key=lambda e: e.finish_position,
            )
            is_finished = race.status == RaceStatus.resulted and len(winners) >= 1
            actual_ids = tuple(e.horse_id for e in winners) if is_finished else None

            # Hit + payout per pool.
            results: dict[str, dict] = {}
            for pool, combos in picks.items():
                if not combos:
                    results[pool] = {"combo": None, "hit": None, "payout": None}
                    continue
                our = combos[0]
                hit: bool | None = None
                if is_finished:
                    if pool == "ganyan" and len(winners) >= 1:
                        hit = our.horses[0] == actual_ids[0]
                    elif pool == "ikili" and len(winners) >= 2:
                        hit = set(our.horses) == set(actual_ids[:2])
                    elif pool == "sirali_ikili" and len(winners) >= 2:
                        hit = our.horses == actual_ids[:2]
                    elif pool == "uclu" and len(winners) >= 3:
                        hit = our.horses == actual_ids[:3]
                payout_col = f"{pool}_payout_tl"
                payout_tl = getattr(race, payout_col, None)
                results[pool] = {
                    "combo": our,
                    "horses": [name_for.get(h, "?") for h in our.horses],
                    "prob_pct": our.probability * 100.0,
                    "hit": hit,
                    "payout": float(payout_tl) if payout_tl is not None else None,
                }

                # Feed the daily tally only when we have a payout (so rows
                # where TJK didn't offer that pool don't drag the denominator).
                if is_finished and payout_tl is not None:
                    tally[pool][0] += 1        # races staked
                    tally[pool][2] += STAKE    # stake
                    if hit:
                        tally[pool][1] += 1
                        tally[pool][3] += float(payout_tl) * STAKE

            rows.append({
                "race": race,
                "pending": not is_finished,
                "picks": results,
                "actual": [
                    name_for.get(e.horse_id, "?") for e in winners[:3]
                ] if is_finished else None,
                "agf_rank_by_id": agf_rank_by_id,
                "name_for": name_for,
            })

        tally_display = {}
        for pool, (n, hits, stake, payout) in tally.items():
            net = payout - stake
            roi_pct = (net / stake) * 100.0 if stake > 0 else None
            tally_display[pool] = {
                "races": n, "hits": hits, "stake": stake,
                "payout": payout, "net": net, "roi_pct": roi_pct,
            }

        return render_template(
            "live.html",
            rows=rows,
            tally=tally_display,
            target=target,
            now=datetime.now(),
        )
    finally:
        session.close()
