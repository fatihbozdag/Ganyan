"""Flask blueprint with all web routes for Ganyan."""

from __future__ import annotations

from datetime import date, datetime

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from sqlalchemy.orm import Session, joinedload

from ganyan.db.models import Race, RaceStatus

bp = Blueprint("main", __name__)

# Module-level cache for the Bayes posterior. Loading the .nc file costs
# ~0.5-1s and parses ~100MB of arviz state — caching across requests keeps
# /advice responsive. Keyed by absolute path so swapping --bayes-posterior
# works without a process restart.
_BAYES_CACHE: dict[str, tuple[object, object]] = {}


def _load_bayes_cached(base_path):
    from pathlib import Path

    key = str(Path(base_path).resolve())
    cached = _BAYES_CACHE.get(key)
    if cached is not None:
        return cached
    nc = Path(base_path).with_suffix(".nc")
    idx = Path(base_path).with_suffix(".indices.json")
    if not (nc.exists() and idx.exists()):
        return None
    from ganyan.predictor.bayes.trainer import load_posterior
    try:
        loaded = load_posterior(Path(base_path))
    except Exception:  # noqa: BLE001
        return None
    _BAYES_CACHE[key] = loaded
    return loaded


@bp.app_context_processor
def _inject_daemon_health():
    """Make daemon health visible in the navbar on every page.

    Cheap because the underlying query is one indexed lookup against
    ``job_runs``; runs once per request and Flask handles caching at
    the response level.  Returns an empty dict on any failure so the
    UI degrades gracefully if ``job_runs`` doesn't exist yet (e.g.
    fresh DB).
    """
    from ganyan.db.models import JobRun

    try:
        session = _get_session()
    except Exception:  # noqa: BLE001
        return {}
    try:
        last = (
            session.query(JobRun)
            .filter(JobRun.job_id == "agf_snapshot",
                    JobRun.status == "success")
            .order_by(JobRun.finished_at.desc()).first()
        )
        if last is None or last.finished_at is None:
            return {"daemon_health": {"color": "secondary", "label": "daemon off"}}
        age_m = int((datetime.now() - last.finished_at).total_seconds() / 60)
        if age_m <= 35:
            color = "success"
        elif age_m <= 90:
            color = "warning"
        else:
            color = "danger"
        return {
            "daemon_health": {"color": color, "label": f"snapshot {age_m}m"},
        }
    except Exception:  # noqa: BLE001
        return {}
    finally:
        session.close()


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
    """Daily dashboard.

    Aggregates the bits an operator looks for *first* on a race day:
    daemon health (last AGF snapshot age, last results-poll), today's
    race counts by status, and the next race + tonight's bet roster.
    Race list goes below — useful but not the headline.
    """
    from sqlalchemy import func

    from ganyan.config import get_settings
    from ganyan.db.models import (
        AgfSnapshot, JobRun, Pick, RaceEntry,
    )

    settings = get_settings()
    session = _get_session()
    today = date.today()
    try:
        today_races = (
            session.query(Race)
            .options(joinedload(Race.track), joinedload(Race.entries))
            .filter(Race.date == today)
            .order_by(Race.post_time.nulls_last(), Race.race_number)
            .all()
        )
        recent_races = (
            session.query(Race)
            .options(joinedload(Race.track))
            .filter(Race.status == RaceStatus.resulted)
            .order_by(Race.date.desc(), Race.race_number.desc())
            .limit(10)
            .all()
        )

        # Daemon health: pull the most recent runs of agf_snapshot,
        # results_poll, morning_card jobs.  Show last-success age in
        # minutes; ≤30 min green, ≤90 min amber, older red.
        last_snapshot_run = (
            session.query(JobRun)
            .filter(JobRun.job_id == "agf_snapshot",
                    JobRun.status == "success")
            .order_by(JobRun.finished_at.desc()).first()
        )
        last_results_run = (
            session.query(JobRun)
            .filter(JobRun.job_id == "results_poll",
                    JobRun.status == "success")
            .order_by(JobRun.finished_at.desc()).first()
        )

        def _age_min(run):
            if run is None or run.finished_at is None:
                return None
            return int((datetime.now() - run.finished_at).total_seconds() / 60)

        snapshot_age = _age_min(last_snapshot_run)
        results_age = _age_min(last_results_run)

        # Daemon-overall status colour: green if snapshot age ≤ 35min
        # (cron is every 30, so anything under 35 is fresh), amber if
        # ≤90, red beyond.  Off entirely if no runs ever.
        if snapshot_age is None:
            daemon = {"color": "secondary", "label": "daemon: unknown"}
        elif snapshot_age <= 35:
            daemon = {"color": "success", "label": f"snapshot {snapshot_age}m"}
        elif snapshot_age <= 90:
            daemon = {"color": "warning", "label": f"snapshot {snapshot_age}m"}
        else:
            daemon = {"color": "danger", "label": f"snapshot {snapshot_age}m old"}

        # Race-count breakdown today.
        status_counts = {
            "scheduled": 0, "resulted": 0, "cancelled": 0,
        }
        for r in today_races:
            status_counts[r.status.value] = status_counts.get(r.status.value, 0) + 1

        # Snapshot count today (all entries × samples).
        snapshot_total = (
            session.query(func.count(AgfSnapshot.id))
            .join(RaceEntry, RaceEntry.id == AgfSnapshot.race_entry_id)
            .join(Race, Race.id == RaceEntry.race_id)
            .filter(Race.date == today).scalar()
        ) or 0
        snapshot_entries = (
            session.query(func.count(func.distinct(AgfSnapshot.race_entry_id)))
            .join(RaceEntry, RaceEntry.id == AgfSnapshot.race_entry_id)
            .join(Race, Race.id == RaceEntry.race_id)
            .filter(Race.date == today).scalar()
        ) or 0
        avg_snaps = (
            round(snapshot_total / snapshot_entries, 1)
            if snapshot_entries else 0.0
        )

        # Today's pick ledger summary: how many bets active, graded, hit.
        today_picks_q = (
            session.query(Pick)
            .join(Race, Race.id == Pick.race_id)
            .filter(Race.date == today)
        )
        n_picks = today_picks_q.count()
        n_graded = today_picks_q.filter(Pick.graded.is_(True)).count()
        n_hit = today_picks_q.filter(Pick.hit.is_(True)).count()

        # Next race upcoming (post_time > now), if any.
        next_race = next(
            (r for r in today_races
             if r.status == RaceStatus.scheduled and r.post_time),
            None,
        )

        return render_template(
            "index.html",
            today_races=today_races,
            recent_races=recent_races,
            daemon_health=daemon,
            today=today,
            snapshot_age=snapshot_age,
            results_age=results_age,
            status_counts=status_counts,
            snapshot_total=snapshot_total,
            snapshot_entries=snapshot_entries,
            avg_snaps=avg_snaps,
            n_picks=n_picks,
            n_graded=n_graded,
            n_hit=n_hit,
            next_race=next_race,
            show_backfill=settings.show_backfill_ui,
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

        from ganyan.predictor.ml import MLPredictor
        from ganyan.predictor.ml.ensemble import EnsemblePredictor

        predictor = MLPredictor(session)
        predictions = predictor.predict(race_id)

        # Ensemble convergence + per-head breakdown.  Best-effort —
        # falls back gracefully when no models are loaded yet.
        ensemble_by_horse: dict[int, dict] = {}
        try:
            ens_predictor = EnsemblePredictor(session)
            ens_preds = ens_predictor.predict(race_id)
            n_heads = len(ens_predictor.models)
            for ep in ens_preds:
                ensemble_by_horse[ep.horse_id] = {
                    "convergence_top1": ep.convergence_top1,
                    "convergence_top3": ep.convergence_top3,
                    "n_heads": n_heads,
                    "mean_probability": round(ep.mean_probability, 2),
                    "disagreement": round(ep.disagreement, 2),
                    "by_model": ep.by_model,
                }
        except FileNotFoundError:
            pass

        # AGF drift per entry — latest snapshot vs earliest, if ≥2.
        from ganyan.db.models import AgfSnapshot, RaceEntry
        drift_by_horse: dict[int, dict] = {}
        entries = (
            session.query(RaceEntry).filter(RaceEntry.race_id == race_id).all()
        )
        for e in entries:
            snaps = (
                session.query(AgfSnapshot.agf, AgfSnapshot.taken_at)
                .filter(AgfSnapshot.race_entry_id == e.id)
                .order_by(AgfSnapshot.taken_at.asc())
                .all()
            )
            if len(snaps) < 2:
                continue
            first = float(snaps[0][0])
            last = float(snaps[-1][0])
            drift_by_horse[e.horse_id] = {
                "first": first,
                "last": last,
                "delta": round(last - first, 2),
                "n_samples": len(snaps),
            }

        from ganyan.predictor.bet_recommendations import (
            compute_bet_recommendations,
        )
        win_probs = {
            p.horse_id: max(p.probability, 0.0) / 100.0 for p in predictions
        }
        horse_names = {p.horse_id: p.horse_name for p in predictions}
        gates = {
            e.horse_id: int(e.gate_number) for e in entries
            if e.gate_number is not None
        }
        bet_picks = [
            r.to_dict() for r in
            compute_bet_recommendations(win_probs, horse_names, gates)
        ]
        recommendations = bet_picks  # legacy template key

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
                            "ensemble": ensemble_by_horse.get(p.horse_id),
                            "agf_drift": drift_by_horse.get(p.horse_id),
                        }
                        for p in predictions
                    ],
                    "recommendations": recommendations,
                }
            )

        return render_template(
            "predict.html",
            race=race,
            predictions=predictions,
            recommendations=recommendations,
            ensemble_by_horse=ensemble_by_horse,
            drift_by_horse=drift_by_horse,
        )
    finally:
        session.close()


# Typical Turkish parimutuel takeouts (rough, varies by pool/track).
_TAKEOUT = {
    "ganyan": 0.18,
    "ikili": 0.22,
    "sirali_ikili": 0.22,
    "uclu": 0.25,
}


def _build_bet_recommendations(
    race, predictions, ikili_fn, sirali_fn, uclu_fn,
) -> list[dict]:
    """Build the per-race betting suggestions shown on /races/<id>/predict.

    Only surfaces strategies that are **backtest-positive or plausibly
    break-even** at our current signal quality:

      - Üçlü top-1: the +150% edge we measured
      - Üçlü box-6: +112% variance-smoothed alternative
      - Sıralı İkili top-1: near-breakeven (-2.7% backtest, low stake)

    We report the model's combination probability + the backtest's
    long-run ROI rather than a per-race EV.  Per-race EV derived from
    market-implied payouts is misleading for these strategies: the
    whole reason Üçlü top-1 is profitable is that the market mis-prices
    the favorite-ordered trifecta, so market-derived EV systematically
    understates the real edge.
    """
    if not predictions:
        return []
    # Our model's win probabilities (0..1), normalised.
    mp = {p.horse_id: max(p.probability, 0) / 100.0 for p in predictions}
    total = sum(mp.values())
    if total > 0:
        mp = {h: v / total for h, v in mp.items()}

    name_for = {p.horse_id: p.horse_name for p in predictions}

    recs: list[dict] = []

    # --- Üçlü top-1 (the edge) ---
    our_uclu = uclu_fn(mp)[:1]
    if our_uclu and len(predictions) >= 3:
        our_top = our_uclu[0]
        recs.append({
            "title": "Üçlü — Top 1 (ana strateji)",
            "subtitle": "Backtest: ~5% hit rate, +150% ROI long-run",
            "horses": [name_for.get(h, "?") for h in our_top.horses],
            "separator": "→",
            "tickets": 1,
            "stake_tl": 100.0,
            "model_prob_pct": our_top.probability * 100.0,
            "expected_long_run_roi_pct": 150.0,
            "warning": (
                "Yüksek varyans: ~%40 ihtimalle kart boyunca hiç vurmayabilir."
            ),
            "positive": True,
        })

        # --- Üçlü box-6 (same horses, all 6 orderings) ---
        top3_set = set(our_top.horses)
        prob_any_order = sum(
            c.probability for c in uclu_fn(mp) if set(c.horses) == top3_set
        )
        recs.append({
            "title": "Üçlü — Kutu 6 (düşük varyans)",
            "subtitle": "Backtest: ~16% hit rate, +112% ROI long-run",
            "horses": [name_for.get(h, "?") for h in our_top.horses],
            "separator": "box",
            "tickets": 6,
            "stake_tl": 600.0,
            "model_prob_pct": prob_any_order * 100.0,
            "expected_long_run_roi_pct": 112.0,
            "warning": "Hit oranı 3× daha yüksek ama bilet maliyeti de 6× daha fazla.",
            "positive": True,
        })

    # --- Sıralı İkili top-1 (break-even indicator) ---
    our_si = sirali_fn(mp)[:1] if len(predictions) >= 2 else []
    if our_si:
        top = our_si[0]
        recs.append({
            "title": "Sıralı İkili — Top 1",
            "subtitle": "Backtest: -2.7% ROI (yaklaşık başabaş)",
            "horses": [name_for.get(h, "?") for h in top.horses],
            "separator": "→",
            "tickets": 1,
            "stake_tl": 100.0,
            "model_prob_pct": top.probability * 100.0,
            "expected_long_run_roi_pct": -2.7,
            "warning": "Kâr beklentisi sıfır civarı — ısınma bahsi olarak düşünülebilir.",
            "positive": False,
        })

    return recs


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
    from ganyan.predictor.ml import MLPredictor

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

        predictor = MLPredictor(session)
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
    if not settings.show_backfill_ui:
        return jsonify({"error": "Geçmiş veri yükleme arayüzü kapalı."}), 404

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
            if _wants_json():
                return jsonify(
                    {"message": "Bugün için sonuç bulunamadı.", "count": 0},
                )
            return redirect(url_for("main.index"))

        updated = 0
        for raw in raw_results:
            parsed = parse_race_card(raw)
            result = update_race_results(session, parsed)
            if result:
                updated += 1
        session.commit()

        if _wants_json():
            return jsonify(
                {
                    "message": f"{updated} yarış sonucu güncellendi.",
                    "count": updated,
                },
            )
        return redirect(url_for("main.index"))

    except Exception as exc:  # noqa: BLE001
        session.rollback()
        msg = f"Hata: {exc}"
        if _wants_json():
            return jsonify({"error": msg}), 500
        return redirect(url_for("main.index")), 302
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
    from ganyan.db.models import Race, RaceStatus
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
        # Per-pool birim (minimum bilet stake).  TJK publishes payouts as
        # "what one bilet at the pool's birim wins" — divide by birim
        # so payout_tl × stake / birim gives the actual TL.  Mirrors
        # BIRIM_TL_BY_STRATEGY in predictor/picks.py.  Üçlü birim = 2 TL,
        # others = 1 TL — without this divisor the /live tally double-
        # counts üçlü winnings vs the graded ledger.
        POOL_BIRIM_TL = {"ganyan": 1.0, "ikili": 1.0, "sirali_ikili": 1.0, "uclu": 2.0}

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
                        birim = POOL_BIRIM_TL.get(pool, 1.0)
                        tally[pool][3] += float(payout_tl) * STAKE / birim

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


# ---------------------------------------------------------------------------
# /picks — strategy-level bet ledger with running ROI
# ---------------------------------------------------------------------------


@bp.route("/picks")
def picks_dashboard():
    """Cumulative + per-strategy + recent picks view.

    Reads the picks table (which the scheduler writes & grades) so the
    user can verify the long-run ROI numbers claimed by the backtest
    with the system's own live track record.
    """
    from ganyan.db.models import Pick, Race
    from ganyan.predictor.picks import strategy_summary
    from sqlalchemy import desc

    session = _get_session()
    try:
        summary = strategy_summary(session)

        # Show the 60 most recent races with all their strategies
        # grouped together.  Ordering by generated_at alone made the
        # newest backfill dominate the list (all rows of one strategy
        # shared the same generated_at).  Pull the top race_ids by
        # race date first, then fetch every pick for those races.
        recent_race_ids = [
            rid for (rid,) in
            session.query(Race.id)
            .join(Pick, Pick.race_id == Race.id)
            .group_by(Race.id, Race.date, Race.race_number)
            .order_by(desc(Race.date), desc(Race.race_number))
            .limit(60)
            .all()
        ]
        recent_picks = (
            session.query(Pick)
            .filter(Pick.race_id.in_(recent_race_ids))
            .all()
        ) if recent_race_ids else []
        # Sort picks so rows for the same race cluster and strategies
        # within a race come out in a stable order.
        _strategy_order = {
            "ganyan_top1": 0, "plase_top1": 1, "sirali_ikili_top1": 2,
            "uclu_top1": 3, "uclu_box6": 4,
        }
        recent_picks.sort(
            key=lambda p: (
                -(p.race_id),  # newer race ids first
                _strategy_order.get(p.strategy, 99),
            )
        )

        # Pre-fetch race + track data we'll need so the template can't
        # trigger lazy-loads outside the session.
        race_ids = {p.race_id for p in recent_picks}
        races = {
            r.id: r for r in
            session.query(Race).filter(Race.id.in_(race_ids)).all()
        } if race_ids else {}

        if _wants_json():
            return jsonify({
                "summary": summary,
                "recent_picks": [
                    {
                        "id": p.id,
                        "race_id": p.race_id,
                        "strategy": p.strategy,
                        "combination_names": p.combination_names,
                        "stake_tl": float(p.stake_tl),
                        "model_prob_pct": (
                            float(p.model_prob_pct) if p.model_prob_pct is not None else None
                        ),
                        "generated_at": p.generated_at.isoformat(),
                        "graded": p.graded,
                        "hit": p.hit,
                        "payout_tl": float(p.payout_tl) if p.payout_tl is not None else None,
                        "net_tl": float(p.net_tl) if p.net_tl is not None else None,
                    }
                    for p in recent_picks
                ],
            })

        return render_template(
            "picks.html",
            summary=summary,
            recent_picks=recent_picks,
            races=races,
        )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# /advice — forward-looking bet advisor (companion to /picks ledger)
# ---------------------------------------------------------------------------


@bp.route("/advice")
def advice_dashboard():
    """Today's (or ?date=) recommended bets with Kelly-sized stakes.

    Reads the picks the scheduler already generated and presents them
    in a "here's what to bet tonight" format.  Distinct from /picks
    which is historical P&L.
    """
    from datetime import datetime as _dt
    from ganyan.db.models import Pick, Race, RaceStatus
    from ganyan.predictor.kelly import (
        strategy_edge_stats, suggested_stake_tl, kelly_fraction,
    )
    from ganyan.predictor.exotics import (
        ganyan_probabilities, sirali_ikili_probabilities,
        uclu_probabilities,
    )
    from ganyan.predictor.trip_wire import compute_trip_wire, is_anomalous, is_halt
    from ganyan.predictor import halt_flag, rolling_pnl_halt, uniformity_guard
    from sqlalchemy.orm import joinedload

    # uclu_top1 dropped 2026-05-02: 0 hits on n=16 gated picks, ROI −100%.
    # uclu_box6 dropped 2026-05-05: rolling 30d ROI −32.5% on n=895 with 7d
    # slope at −44.4%; structurally fragile (Üçlü-K3 set match ~15%).
    BETTING_STRATEGIES = ("sirali_ikili_top1",)
    STRATEGY_ORDER = {s: i for i, s in enumerate(BETTING_STRATEGIES)}

    date_str = request.args.get("date")
    bankroll = float(request.args.get("bankroll", 10000))
    kelly_mult = float(request.args.get("kelly", 0.25))
    # Default OFF since 2026-05-05: filter helps uclu_box6 (+14.8pp) but
    # hurts sirali_ikili_top1 (−10.4pp); uclu_box6 retired, only the hurt
    # remains. ?cohort_filter=1 to re-enable.
    cohort_filter = request.args.get("cohort_filter", "0") not in ("0", "false", "")
    # Bayes skip-gate (mirrors `ganyan advice` CLI). Default ON.
    bayes_skip = request.args.get("bayes_skip", "1") not in ("0", "false", "")
    bayes_min_prob = float(request.args.get("bayes_min_prob", 0.35))
    bayes_min_lo5 = float(request.args.get("bayes_min_lo5", 0.20))
    bayes_posterior_path = request.args.get(
        "bayes_posterior", "models/bayes_pl_v3",
    )
    # Trip-wire (avg top-1 prob z-score vs 90d baseline). ?bypass=1 disables.
    trip_wire_sigma = float(request.args.get("trip_wire_sigma", 2.0))
    trip_wire_bypass = request.args.get("bypass", "0") in ("1", "true")
    try:
        target_date = (
            _dt.strptime(date_str, "%Y-%m-%d").date() if date_str else date.today()
        )
    except ValueError:
        target_date = date.today()

    def _cohort_skip_reason(race) -> str | None:
        rt = (race.race_type or "")
        if "Maiden /Dişi" in rt or "Maiden Dişi" in rt:
            return "Maiden /Dişi (-46% ROI)"
        if len(race.entries) >= 13:
            return f"field {len(race.entries)}≥13 (-22% ROI)"
        track_name = race.track.name if race.track else ""
        if track_name in {"Şanlıurfa", "Bursa"}:
            return f"track {track_name}"
        return None

    session = _get_session()

    # Halt-flag canary checks. If any fires, the flag is set and rendering
    # downgrades to informational (no stake_tl / kelly_tl). First writer wins.
    halt_state = halt_flag.is_halted()
    if halt_state is None:
        # In-process canary: rolling-PnL halt per active strategy.
        pnl_results = rolling_pnl_halt.check_all_strategies(
            session, BETTING_STRATEGIES,
        )
        for _strat, _reason in pnl_results.items():
            if _reason:
                halt_flag.set_halt(reason=_reason, source="rolling_pnl_halt")
                halt_state = halt_flag.is_halted()
                break

    # Lazy-load Bayes posterior + per-horse history (speed/workout/pace).
    # All three histories are session-scoped (one query each) and reused
    # across every race in this request.
    bayes_loaded = None
    bayes_speed_history = None
    bayes_workout_history = None
    bayes_pace_history = None
    bayes_load_error = None
    if bayes_skip:
        bayes_loaded = _load_bayes_cached(bayes_posterior_path)
        if bayes_loaded is None:
            bayes_load_error = (
                f"posterior not found at {bayes_posterior_path}.nc / "
                f".indices.json — train via `uv run python logs/bayes_train.py`"
            )
        else:
            try:
                from ganyan.predictor.speed_figures import (
                    build_horse_speed_history, compute_track_variants,
                )
                from ganyan.predictor.workouts import build_horse_workout_history
                from ganyan.predictor.pace import (
                    build_horse_pace_history, compute_pace_baseline,
                )
                variants = compute_track_variants(session, to_date=target_date)
                bayes_speed_history = build_horse_speed_history(
                    session, variants, to_date=target_date,
                )
                bayes_workout_history = build_horse_workout_history(
                    session, to_date=target_date,
                )
                pace_baseline = compute_pace_baseline(session, to_date=target_date)
                bayes_pace_history = build_horse_pace_history(
                    session, pace_baseline, to_date=target_date,
                )
            except Exception as e:  # noqa: BLE001
                bayes_load_error = f"history build failed: {e}"
                bayes_loaded = None

    def _bayes_top_pred(race):
        if bayes_loaded is None:
            return None
        from ganyan.predictor.bayes.predictor import predict_from_posterior
        from ganyan.predictor.speed_figures import horse_speed_score
        from ganyan.predictor.workouts import horse_workout_score
        from ganyan.predictor.pace import horse_pace_score
        entries = list(race.entries)
        if len(entries) < 3:
            return None
        race_in = {
            "horse_ids": [e.horse_id for e in entries],
            "horse_names": [
                e.horse.name if e.horse else "?" for e in entries
            ],
            "jockeys": [e.jockey or "" for e in entries],
            "sires": [
                (e.horse.sire or "") if e.horse else "" for e in entries
            ],
            "track_id": race.track_id,
            "distance_meters": race.distance_meters or 0,
            "agfs": [
                float(e.agf) if e.agf is not None else 0.0 for e in entries
            ],
            "kgss": [
                float(e.kgs) if e.kgs is not None else 0.0 for e in entries
            ],
            "s20s": [
                float(e.s20) if e.s20 is not None else 0.0 for e in entries
            ],
            "last_sixes": [e.last_six or "" for e in entries],
            "speeds": [
                horse_speed_score(bayes_speed_history, e.horse_id, race.date) or 0.0
                for e in entries
            ],
            "workouts": [
                horse_workout_score(bayes_workout_history, e.horse_id, race.date) or 0.0
                for e in entries
            ],
            "paces": [
                horse_pace_score(bayes_pace_history, e.horse_id, race.date) or 0.0
                for e in entries
            ],
        }
        idata, frame = bayes_loaded
        try:
            preds = predict_from_posterior(idata, frame, race_in)
        except Exception:  # noqa: BLE001
            return None
        return preds[0] if preds else None

    def _bayes_skip_reason(top) -> str | None:
        if top is None:
            return None
        if top.mean_prob < bayes_min_prob:
            return (
                f"Bayes top-1 mean {top.mean_prob:.0%}<"
                f"{bayes_min_prob:.0%}"
            )
        if top.lo_5 < bayes_min_lo5:
            return (
                f"Bayes top-1 lo₅ {top.lo_5:.0%}<{bayes_min_lo5:.0%}"
            )
        return None

    try:
        edge_stats = strategy_edge_stats(
            session,
            strategies=BETTING_STRATEGIES,
            before_date=target_date,
        )

        races = (
            session.query(Race)
            .options(joinedload(Race.track), joinedload(Race.entries))
            .filter(Race.date == target_date)
            .order_by(Race.post_time.nulls_last(), Race.race_number)
            .all()
        )
        race_ids = [r.id for r in races]
        picks = (
            session.query(Pick)
            .filter(
                Pick.race_id.in_(race_ids),
                Pick.strategy.in_(BETTING_STRATEGIES),
            )
            .all() if race_ids else []
        )
        picks_by_race: dict[int, list[Pick]] = {}
        for p in picks:
            picks_by_race.setdefault(p.race_id, []).append(p)

        # Winner name per resulted race (while session is open).
        winner_name_by_race: dict[int, str] = {}
        for r in races:
            if r.status != RaceStatus.resulted:
                continue
            winner = next(
                (e for e in r.entries if e.finish_position == 1), None,
            )
            if winner and winner.horse:
                winner_name_by_race[r.id] = winner.horse.name

        # Harville expansion per race — the individual win probs and the
        # top-N trifecta / exacta rankings, so the advice UI can show
        # *why* a particular combination got picked.  Keyed by race_id.
        harville_by_race: dict[int, dict] = {}
        for r in races:
            if not r.entries:
                continue
            # Resolve names + probs in a single dict we can reuse.
            horse_names: dict[int, str] = {}
            win_probs: dict[int, float] = {}
            finish_pos: dict[int, int | None] = {}
            for e in r.entries:
                if e.horse:
                    horse_names[e.horse_id] = e.horse.name
                if e.predicted_probability is not None:
                    win_probs[e.horse_id] = max(
                        float(e.predicted_probability) / 100.0, 0.0,
                    )
                finish_pos[e.horse_id] = e.finish_position

            if not win_probs:
                continue

            # Actual finish for hit-highlighting on resulted races.
            actual_top3: tuple[int, int, int] | None = None
            actual_top2: tuple[int, int] | None = None
            if r.status == RaceStatus.resulted:
                by_pos = {pos: hid for hid, pos in finish_pos.items() if pos}
                if all(k in by_pos for k in (1, 2, 3)):
                    actual_top3 = (by_pos[1], by_pos[2], by_pos[3])
                if all(k in by_pos for k in (1, 2)):
                    actual_top2 = (by_pos[1], by_pos[2])

            ganyan_ranked = [
                {
                    "horse_id": c.horses[0],
                    "horse_name": horse_names.get(c.horses[0], "?"),
                    "prob_pct": c.probability * 100,
                    "finish_position": finish_pos.get(c.horses[0]),
                }
                for c in ganyan_probabilities(win_probs)
            ]

            uclu_top = uclu_probabilities(win_probs)[:8]
            uclu_ranked = [
                {
                    "combination_names": [
                        horse_names.get(h, "?") for h in c.horses
                    ],
                    "prob_pct": c.probability * 100,
                    "hit": (
                        actual_top3 is not None and tuple(c.horses) == actual_top3
                    ),
                }
                for c in uclu_top
            ]

            si_top = sirali_ikili_probabilities(win_probs)[:6]
            si_ranked = [
                {
                    "combination_names": [
                        horse_names.get(h, "?") for h in c.horses
                    ],
                    "prob_pct": c.probability * 100,
                    "hit": (
                        actual_top2 is not None and tuple(c.horses) == actual_top2
                    ),
                }
                for c in si_top
            ]

            harville_by_race[r.id] = {
                "win_probs": ganyan_ranked,
                "uclu_top": uclu_ranked,
                "sirali_top": si_ranked,
            }

        # Bayes top-1 (with 90% CI) per race. Pre-computed once so the
        # per-race loop below can trigger the gate decision and the kelly
        # multiplier without re-running ADVI per race.
        bayes_top_by_race: dict[int, object] = {}
        if bayes_loaded is not None:
            for r in races:
                top = _bayes_top_pred(r)
                if top is not None:
                    bayes_top_by_race[r.id] = top

        # Build view model — per-race card with pick rows enriched with
        # Kelly suggestion + outcome flags.
        races_with_picks = []
        skipped_cohort_races: list[dict] = []
        skipped_bayes_races: list[dict] = []
        n_advised = n_with_hit = n_no_pool = 0
        n_skipped_cohort = 0
        n_skipped_bayes = 0
        gross_stake = effective_stake = total_payout = 0.0
        any_graded = False

        for r in races:
            rpicks = picks_by_race.get(r.id)
            if not rpicks:
                continue
            if cohort_filter:
                reason = _cohort_skip_reason(r)
                if reason:
                    skipped_cohort_races.append({
                        "track": r.track.name if r.track else "?",
                        "race_number": r.race_number,
                        "post_time": r.post_time,
                        "reason": reason,
                    })
                    n_skipped_cohort += 1
                    continue
            bayes_top = bayes_top_by_race.get(r.id)
            if bayes_skip:
                br = _bayes_skip_reason(bayes_top)
                if br:
                    skipped_bayes_races.append({
                        "track": r.track.name if r.track else "?",
                        "race_number": r.race_number,
                        "post_time": r.post_time,
                        "reason": br,
                        "bayes_top1_horse": (
                            bayes_top.horse_name if bayes_top else None
                        ),
                        "bayes_top1_mean": (
                            bayes_top.mean_prob if bayes_top else None
                        ),
                        "bayes_top1_lo5": (
                            bayes_top.lo_5 if bayes_top else None
                        ),
                        "bayes_top1_hi95": (
                            bayes_top.hi_95 if bayes_top else None
                        ),
                    })
                    n_skipped_bayes += 1
                    continue
            # Uniformity guard — degenerate 1/N softmax catches AGF-NULL etc.
            if halt_state is None:
                race_probs = [
                    float(e.predicted_probability or 0) for e in r.entries
                ]
                _u_reason = uniformity_guard.check_race_field(
                    race_id=r.id, probabilities=race_probs,
                )
                if _u_reason:
                    halt_flag.set_halt(
                        reason=_u_reason, source="uniformity_guard",
                    )
                    halt_state = halt_flag.is_halted()

            kelly_mult_eff = kelly_mult
            bayes_conf = None
            if bayes_top is not None:
                bayes_conf = max(
                    0.0, min(1.0, (bayes_top.mean_prob - 0.35) / 0.15),
                )
                kelly_mult_eff = kelly_mult * (0.5 + 0.5 * bayes_conf)

            race_had_hit = False
            pick_rows = []
            rpicks.sort(key=lambda p: STRATEGY_ORDER.get(p.strategy, 99))
            for p in rpicks:
                prob = float(p.model_prob_pct or 0)
                stake = float(p.stake_tl)
                stats = edge_stats.get(p.strategy)
                kelly_tl = 0.0
                calibrated_prob_pct = 0.0
                if stats and stats.avg_b > 0 and stats.avg_model_prob > 0:
                    calibrated_p = stats.calibrate(prob / 100.0)
                    calibrated_prob_pct = calibrated_p * 100
                    kelly_tl = suggested_stake_tl(
                        win_prob=calibrated_p,
                        b=stats.avg_b,
                        bankroll_tl=bankroll,
                        base_stake_tl=stake,
                        kelly_multiplier=kelly_mult_eff,
                    )

                is_miss = p.graded and not p.hit
                is_no_pool = (
                    not p.graded and r.status == RaceStatus.resulted
                )
                if is_no_pool:
                    n_no_pool += 1
                if p.graded and p.hit:
                    race_had_hit = True
                    any_graded = True
                    total_payout += float(p.payout_tl or 0)
                elif p.graded:
                    any_graded = True

                gross_stake += stake
                if not is_no_pool:
                    effective_stake += stake

                pick_rows.append({
                    "strategy": p.strategy,
                    "combination_display": " → ".join(
                        p.combination_names or [],
                    ),
                    "model_prob_pct": prob,
                    "calibrated_prob_pct": calibrated_prob_pct,
                    # Halt suppresses stake/kelly display; template renders "—".
                    "stake_tl": None if halt_state else stake,
                    "kelly_tl": None if halt_state else kelly_tl,
                    "hit": bool(p.hit) if p.graded else False,
                    "is_miss": is_miss,
                    "is_no_pool": is_no_pool,
                    "payout_tl": float(p.payout_tl) if p.payout_tl else 0.0,
                })

            races_with_picks.append({
                "race": r,
                "picks": pick_rows,
                "winner_name": winner_name_by_race.get(r.id),
                "harville": harville_by_race.get(r.id),
                "bayes_top": bayes_top,
                "bayes_conf": bayes_conf,
                "kelly_mult_eff": kelly_mult_eff,
            })
            n_advised += 1
            if race_had_hit:
                n_with_hit += 1

        # Strategy edge dict for the template (pre-compute Kelly on 10K).
        edge_display = {}
        for k, v in edge_stats.items():
            kf = kelly_fraction(v.hit_rate, v.avg_b, kelly_multiplier=kelly_mult)
            d = v.to_dict()
            d["kelly_quarter_10k"] = kf * 10000
            edge_display[k] = d

        net = total_payout - effective_stake
        roi = (100 * net / effective_stake) if effective_stake > 0 else 0.0
        summary = {
            "n_advised": n_advised,
            "n_total": len(races),
            "gross_stake": gross_stake,
            "effective_stake": effective_stake,
            "n_no_pool": n_no_pool,
            "n_skipped_cohort": n_skipped_cohort,
            "n_skipped_bayes": n_skipped_bayes,
            "cohort_filter_on": cohort_filter,
            "bayes_skip_on": bayes_skip,
            "bayes_loaded": bayes_loaded is not None,
            "bayes_load_error": bayes_load_error,
            "bayes_min_prob": bayes_min_prob,
            "bayes_min_lo5": bayes_min_lo5,
            "bayes_posterior_path": bayes_posterior_path,
            "graded": any_graded,
            "payout": total_payout,
            "net": net,
            "roi_pct": roi,
            "n_with_hit": n_with_hit,
        }

        if _wants_json():
            return jsonify({
                "date": str(target_date),
                "bankroll_tl": bankroll,
                "kelly_multiplier": kelly_mult,
                "cohort_filter_on": cohort_filter,
                "bayes_skip_on": bayes_skip,
                "bayes_min_prob": bayes_min_prob,
                "bayes_min_lo5": bayes_min_lo5,
                "summary": summary,
                "edge_stats": edge_display,
                "skipped_cohort_races": skipped_cohort_races,
                "skipped_bayes_races": skipped_bayes_races,
                "races": [
                    {
                        "race_id": r["race"].id,
                        "track": r["race"].track.name if r["race"].track else None,
                        "race_number": r["race"].race_number,
                        "post_time": r["race"].post_time,
                        "distance_meters": r["race"].distance_meters,
                        "status": (
                            r["race"].status.value if r["race"].status else None
                        ),
                        "winner_name": r["winner_name"],
                        "picks": r["picks"],
                        "harville": r["harville"],
                        "bayes_top1_horse": (
                            r["bayes_top"].horse_name if r["bayes_top"] else None
                        ),
                        "bayes_top1_mean": (
                            r["bayes_top"].mean_prob if r["bayes_top"] else None
                        ),
                        "bayes_top1_lo5": (
                            r["bayes_top"].lo_5 if r["bayes_top"] else None
                        ),
                        "bayes_top1_hi95": (
                            r["bayes_top"].hi_95 if r["bayes_top"] else None
                        ),
                        "bayes_confidence": r["bayes_conf"],
                        "kelly_multiplier_effective": r["kelly_mult_eff"],
                    } for r in races_with_picks
                ],
            })

        trip_info = compute_trip_wire(session, target_date)
        trip_halt = is_halt(trip_info, trip_wire_sigma) and not trip_wire_bypass
        trip_warn = (
            is_anomalous(trip_info, trip_wire_sigma)
            and not is_halt(trip_info, trip_wire_sigma)
        )

        return render_template(
            "advice.html",
            target_date=str(target_date),
            races_with_picks=races_with_picks,
            skipped_cohort_races=skipped_cohort_races,
            skipped_bayes_races=skipped_bayes_races,
            cohort_filter_on=cohort_filter,
            bayes_skip_on=bayes_skip,
            summary=summary,
            edge_stats=edge_display,
            trip_info=trip_info,
            trip_sigma=trip_wire_sigma,
            trip_halt=trip_halt,
            trip_warn=trip_warn,
            trip_bypassed=trip_wire_bypass,
            halt_state=halt_state,
        )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# GET /multi-picks — 5'lı / 6'lı / 7'lı coupons across today's tracks
# ---------------------------------------------------------------------------


@bp.route("/multi-picks")
def multi_picks_page():
    """Render today's 6'lı GANYAN coupon for every track with ≥6 races.

    For each track, we attempt the largest pool the program can fit
    (7'lı if ≥7 races, otherwise 6'lı, otherwise 5'lı). Coupons are
    generated on demand via ``generate_coupon`` — we don't persist
    here; the CLI does that with ``--persist`` when an operator
    actually wants to bet. Recently graded picks (last 14 days) are
    listed below for tracking.
    """
    from datetime import timedelta

    from ganyan.db.models import MultiRacePool, MultiRacePick, Track
    from ganyan.predictor.multi_race_picks import generate_coupon

    target_qs = request.args.get("date")
    target = (
        datetime.strptime(target_qs, "%Y-%m-%d").date()
        if target_qs else date.today()
    )

    try:
        max_tickets = int(request.args.get("max_tickets", "512"))
    except ValueError:
        max_tickets = 512
    max_tickets = max(50, min(max_tickets, 2000))

    session = _get_session()
    try:
        tracks = (
            session.query(Track, Race)
            .join(Race, Race.track_id == Track.id)
            .filter(Race.date == target)
            .order_by(Track.name, Race.race_number)
            .all()
        )
        races_per_track: dict[str, list[Race]] = {}
        for track, race in tracks:
            races_per_track.setdefault(track.name, []).append(race)

        # Collect published race windows from multi_race_pools rows that
        # have already been scraped for this date (start_race_no non-NULL).
        # Key: (track_name, pool_type) → ordered list of (start, end) tuples.
        # Falls back to every valid mathematical window when missing or NULL.
        pub_rows = (
            session.query(MultiRacePool, Track)
            .join(Track, Track.id == MultiRacePool.track_id)
            .filter(
                MultiRacePool.date == target,
                MultiRacePool.start_race_no.isnot(None),
                MultiRacePool.end_race_no.isnot(None),
            )
            .order_by(Track.name, MultiRacePool.pool_type, MultiRacePool.pool_index)
            .all()
        )
        published_windows: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for pool_row, track_row in pub_rows:
            key = (track_row.name, pool_row.pool_type)
            published_windows.setdefault(key, []).append(
                (pool_row.start_race_no, pool_row.end_race_no),
            )

        coupons: list[dict] = []
        skipped: list[dict] = []
        for track_name, race_list in races_per_track.items():
            n = len(race_list)
            if n < 5:
                skipped.append({
                    "track": track_name,
                    "reason": f"only {n} race(s) on program (need ≥5)",
                })
                continue

            # Pool types this program can fit. 6'lı is the iconic Turkish
            # multi-race pool so always show it when feasible; add 7'lı
            # on top for ≥7-race programs. 5'lı reserved for tiny 5-race
            # programs where neither 6'lı nor 7'lı is possible.
            applicable = []
            if n == 5:
                applicable.append(("5li", 5))
            if n >= 6:
                applicable.append(("6li", 6))
            if n >= 7:
                applicable.append(("7li", 7))

            for pool_type, leg_count in applicable:
                pub_key = (track_name, pool_type)
                if pub_key in published_windows:
                    windows = published_windows[pub_key]
                    window_source = "tjk"
                else:
                    n_fallback = n - leg_count + 1
                    windows = [
                        (s, s + leg_count - 1)
                        for s in range(1, n_fallback + 1)
                    ]
                    window_source = "inferred"

                for start_race_no, end_race_no in windows:
                    try:
                        draft = generate_coupon(
                            session, target, track_name, start_race_no,
                            pool_type=pool_type, max_tickets=max_tickets,
                        )
                    except ValueError as exc:
                        skipped.append({
                            "track": (
                                f"{track_name} {pool_type} "
                                f"R{start_race_no}-R{end_race_no}"
                            ),
                            "reason": str(exc),
                        })
                        continue

                    tier_labels = []
                    for w in (len(leg) for leg in draft.kept_horses_per_leg):
                        if w == 1:
                            tier_labels.append(("LOCK", "success"))
                        elif w in (2, 3):
                            tier_labels.append(("medium", "primary"))
                        elif w == 4:
                            tier_labels.append(("wide", "warning"))
                        else:
                            tier_labels.append(("spread", "secondary"))

                    legs = []
                    for i, (kept, conv, (label, color)) in enumerate(zip(
                        draft.kept_horses_per_leg,
                        draft.conviction_per_leg,
                        tier_labels,
                    )):
                        legs.append({
                            "leg_no": i + 1,
                            "race_no": start_race_no + i,
                            "kept": kept,
                            "kept_str": ", ".join(str(g) for g in kept),
                            "width": len(kept),
                            "conviction_pct": conv * 100.0,
                            "tier_label": label,
                            "tier_color": color,
                        })

                    coupons.append({
                        "track": track_name,
                        "pool_type": pool_type,
                        "leg_count": leg_count,
                        "start_race_no": start_race_no,
                        "end_race_no": end_race_no,
                        "n_windows": len(windows),
                        "window_source": window_source,
                        "legs": legs,
                        "total_tickets": draft.total_tickets,
                    })

        recent = (
            session.query(MultiRacePick, Track)
            .join(Track, Track.id == MultiRacePick.track_id)
            .filter(
                MultiRacePick.graded == True,  # noqa: E712
                MultiRacePick.date >= target - timedelta(days=14),
            )
            .order_by(MultiRacePick.date.desc(), Track.name)
            .all()
        )
        recent_picks = [
            {
                "date": p.date,
                "track": t.name,
                "pool_type": p.pool_type,
                "pool_index": p.pool_index,
                "stake_tl": float(p.stake_tl),
                "hit": p.hit,
                "payout_tl": float(p.payout_tl) if p.payout_tl is not None else 0.0,
                "net_tl": float(p.net_tl) if p.net_tl is not None else 0.0,
                "total_tickets": p.total_tickets,
            }
            for p, t in recent
        ]

        if _wants_json():
            return jsonify({
                "date": target.isoformat(),
                "max_tickets": max_tickets,
                "coupons": [
                    {**c, "legs": [{**leg} for leg in c["legs"]]}
                    for c in coupons
                ],
                "skipped": skipped,
                "recent_picks": [
                    {**rp, "date": rp["date"].isoformat()}
                    for rp in recent_picks
                ],
            })

        return render_template(
            "multi_picks.html",
            target_date=target,
            max_tickets=max_tickets,
            coupons=coupons,
            skipped=skipped,
            recent_picks=recent_picks,
        )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# GET /bet-picks — Per-race recommendations for every TJK structure
# ---------------------------------------------------------------------------


@bp.route("/bet-picks")
def bet_picks_page():
    """One-page view of bet recommendations for every race on a date.

    For each race, shows the model's pick across every TJK structure:
    Tek/Ganyan, Plase, Sıralı İkili, İkili, Üçlü Sıralı, Üçlü Komple-3,
    Üçlü Sıralı Virgüllü, 4'lü Sıralı. Aligned with the master directive
    (2026-05-03): pick winners across all bet types.
    """
    from ganyan.predictor.bet_recommendations import compute_bet_recommendations
    from ganyan.predictor.ml.ensemble import EnsemblePredictor
    from ganyan.db.models import RaceEntry

    target_qs = request.args.get("date")
    target = (
        datetime.strptime(target_qs, "%Y-%m-%d").date()
        if target_qs else date.today()
    )

    session = _get_session()
    try:
        races = (
            session.query(Race)
            .filter(Race.date == target)
            .order_by(Race.post_time.asc().nullslast(), Race.race_number.asc())
            .all()
        )

        try:
            predictor = EnsemblePredictor(session)
        except FileNotFoundError:
            predictor = None

        items: list[dict] = []
        for race in races:
            if predictor is None:
                continue
            preds = predictor.predict(race.id)
            if not preds:
                continue
            entries = (
                session.query(RaceEntry).filter(RaceEntry.race_id == race.id).all()
            )
            gates = {
                e.horse_id: int(e.gate_number) for e in entries
                if e.gate_number is not None
            }
            win_probs = {p.horse_id: p.mean_probability / 100.0 for p in preds}
            names = {p.horse_id: p.horse_name for p in preds}
            recs = compute_bet_recommendations(win_probs, names, gates)
            items.append({
                "race": race,
                "track_name": race.track.name if race.track else "?",
                "recommendations": [r.to_dict() for r in recs],
            })

        if _wants_json():
            return jsonify({
                "date": target.isoformat(),
                "items": [
                    {
                        "race_id": it["race"].id,
                        "track": it["track_name"],
                        "race_number": it["race"].race_number,
                        "recommendations": it["recommendations"],
                    }
                    for it in items
                ],
            })

        return render_template(
            "bet_picks.html", target_date=target, items=items,
        )
    finally:
        session.close()
