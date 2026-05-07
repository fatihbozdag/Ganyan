"""Ganyan CLI — Turkish horse racing prediction system."""

import asyncio
import json
import logging
import subprocess
from datetime import date, datetime
from typing import TYPE_CHECKING

import typer

from ganyan.config import get_settings

if TYPE_CHECKING:
    from ganyan.predictor.bayes.predictor import BayesPrediction

app = typer.Typer(name="ganyan", help="Turkish horse racing prediction system")
scrape_app = typer.Typer(help="Scrape race data from TJK")
predict_app = typer.Typer(help="Generate race predictions")
evaluate_app = typer.Typer(help="Evaluate prediction accuracy")
races_app = typer.Typer(help="View race information")
db_app = typer.Typer(help="Database management")
train_app = typer.Typer(help="Train the ML ranker model")
crawl_app = typer.Typer(help="Crawl per-horse detail pages (pedigree, etc.)")
value_app = typer.Typer(help="Find horses the value-betting model thinks are mispriced")

app.add_typer(scrape_app, name="scrape")
app.add_typer(predict_app, name="predict")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(races_app, name="races")
app.add_typer(db_app, name="db")
app.add_typer(train_app, name="train")
app.add_typer(crawl_app, name="crawl")
app.add_typer(value_app, name="value-picks")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# scrape
# ---------------------------------------------------------------------------


@scrape_app.callback(invoke_without_command=True)
def scrape(
    ctx: typer.Context,
    today: bool = typer.Option(False, "--today", help="Fetch today's race cards"),
    results: bool = typer.Option(False, "--results", help="Fetch today's results"),
    backfill: bool = typer.Option(False, "--backfill", help="Run backfill from a date"),
    history: bool = typer.Option(
        False, "--history", help="Bulk-load historical winners via KosuSorgulama"
    ),
    results_range: bool = typer.Option(
        False, "--results-range",
        help="Full-field historical results via GunlukYarisSonuclari (preferred for training data).",
    ),
    from_date: str = typer.Option(
        None, "--from", help="Start date (YYYY-MM-DD)"
    ),
    to_date: str = typer.Option(
        None, "--to", help="End date (YYYY-MM-DD, default: today)"
    ),
    rescrape: bool = typer.Option(
        False, "--rescrape",
        help="Re-scrape even dates already marked complete in scrape_log.",
    ),
) -> None:
    """Scrape race data from TJK."""
    if ctx.invoked_subcommand is not None:
        return  # `scrape external <source>` etc. — let the subcommand run.
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    if today:
        asyncio.run(_scrape_today(settings))
    elif results:
        asyncio.run(_scrape_results(settings))
    elif backfill:
        if from_date is None:
            typer.echo("Error: --from is required with --backfill", err=True)
            raise typer.Exit(code=1)
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = (
            datetime.strptime(to_date, "%Y-%m-%d").date() if to_date else None
        )
        asyncio.run(_run_backfill(settings, start, end, rescrape=rescrape))
    elif history:
        if from_date is None:
            typer.echo("Error: --from is required with --history", err=True)
            raise typer.Exit(code=1)
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = (
            datetime.strptime(to_date, "%Y-%m-%d").date()
            if to_date
            else date.today()
        )
        asyncio.run(_run_historical_backfill(settings, start, end))
    elif results_range:
        if from_date is None:
            typer.echo("Error: --from is required with --results-range", err=True)
            raise typer.Exit(code=1)
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = (
            datetime.strptime(to_date, "%Y-%m-%d").date()
            if to_date else date.today()
        )
        asyncio.run(_run_full_results_backfill(settings, start, end, rescrape))
    else:
        typer.echo(
            "Use --today, --results, --backfill, --history, or --results-range. See --help.",
        )


async def _scrape_today(settings) -> None:
    """Fetch today's race cards, parse, and store them."""
    from ganyan.db import get_session
    from ganyan.scraper import TJKClient, parse_race_card
    from ganyan.scraper.backfill import store_race_card, log_scrape
    from ganyan.db.models import ScrapeStatus

    session = get_session()
    try:
        async with TJKClient(
            base_url=settings.tjk_base_url, delay=settings.scrape_delay
        ) as client:
            raw_cards = await client.get_race_card(date.today())
            if not raw_cards:
                typer.echo("No race cards found for today.")
                return
            for raw in raw_cards:
                parsed = parse_race_card(raw)
                store_race_card(session, parsed)
                log_scrape(session, date.today(), parsed.track_name, ScrapeStatus.success)
            session.commit()
            typer.echo(f"Stored {len(raw_cards)} race card(s) for today.")
    except Exception as exc:
        session.rollback()
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    finally:
        session.close()


async def _scrape_results(settings) -> None:
    """Fetch today's results and update existing entries."""
    from ganyan.db import get_session
    from ganyan.scraper import TJKClient, parse_race_card
    from ganyan.scraper.backfill import update_race_results

    session = get_session()
    try:
        async with TJKClient(
            base_url=settings.tjk_base_url, delay=settings.scrape_delay
        ) as client:
            raw_cards = await client.get_race_results(date.today())
            if not raw_cards:
                typer.echo("No results found for today.")
                return
            updated = 0
            for raw in raw_cards:
                parsed = parse_race_card(raw)
                race = update_race_results(session, parsed)
                if race is not None:
                    updated += 1
            session.commit()
            typer.echo(f"Updated {updated} race(s) with results.")
    except Exception as exc:
        session.rollback()
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    finally:
        session.close()


async def _run_backfill(
    settings,
    from_date: date,
    to_date: date | None = None,
    *,
    rescrape: bool = False,
) -> None:
    """Run the BackfillManager for historical data."""
    from ganyan.db import get_session
    from ganyan.scraper import TJKClient
    from ganyan.scraper.backfill import BackfillManager

    session = get_session()
    try:
        async with TJKClient(
            base_url=settings.tjk_base_url, delay=settings.scrape_delay
        ) as client:
            manager = BackfillManager(session, client)
            await manager.backfill(
                from_date=from_date, to_date=to_date, rescrape=rescrape,
            )
            typer.echo("Backfill complete.")
    except Exception as exc:
        session.rollback()
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    finally:
        session.close()


async def _run_historical_backfill(
    settings, from_date: date, to_date: date
) -> None:
    """Run historical backfill via the KosuSorgulama bulk query endpoint."""
    from ganyan.db import get_session
    from ganyan.scraper import TJKClient
    from ganyan.scraper.backfill import BackfillManager

    session = get_session()
    try:
        async with TJKClient(
            base_url=settings.tjk_base_url, delay=settings.scrape_delay
        ) as client:
            manager = BackfillManager(session, client)
            count = await manager.backfill_historical(
                from_date=from_date, to_date=to_date,
            )
            typer.echo(
                f"Historical backfill complete: {count} race(s) stored "
                f"({from_date} -> {to_date})."
            )
    except Exception as exc:
        session.rollback()
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


async def _run_full_results_backfill(
    settings, from_date: date, to_date: date, rescrape: bool,
) -> None:
    """Full-field historical results via GunlukYarisSonuclari per-date."""
    from ganyan.db import get_session
    from ganyan.scraper import TJKClient
    from ganyan.scraper.backfill import BackfillManager

    session = get_session()
    try:
        async with TJKClient(
            base_url=settings.tjk_base_url, delay=settings.scrape_delay,
        ) as client:
            manager = BackfillManager(session, client)
            count = await manager.backfill_full_results(
                from_date=from_date, to_date=to_date, rescrape=rescrape,
            )
            typer.echo(
                f"Full-field results backfill complete: {count} race(s) stored "
                f"({from_date} -> {to_date})."
            )
    except Exception as exc:
        session.rollback()
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    finally:
        session.close()


@scrape_app.command("external")
def scrape_external(
    source: str = typer.Argument(
        ..., help="Source name (e.g. 'yarisrehberi'). See `--list`.",
    ),
    target_date: str = typer.Option(
        None, "--date", help="YYYY-MM-DD; defaults to today.",
    ),
) -> None:
    """Run a third-party signal scraper.

    Plugins are registered in :mod:`ganyan.scraper.external`.  Each
    persists its findings to the ``external_signals`` table.
    """
    from ganyan.scraper.external import REGISTRY
    from ganyan.scraper.external.base import persist_signals
    from ganyan.db import get_session

    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    if source == "--list" or source == "list":
        typer.echo("Registered external sources:")
        for name, cls in sorted(REGISTRY.items()):
            sig_types = ", ".join(cls.signal_types)
            typer.echo(f"  {name:<20} signals: {sig_types}")
        return

    if source not in REGISTRY:
        typer.echo(
            f"Unknown source {source!r}. "
            f"Known: {sorted(REGISTRY)}", err=True,
        )
        raise typer.Exit(code=1)

    if target_date is None:
        target = date.today()
    else:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()

    from ganyan.scraper.external.resolver import (
        bind_discipline_to_entries,
        bind_track_conditions_to_races,
        bind_workouts_to_entries,
        resolve_unbound_signals,
    )

    session = get_session()
    try:
        plugin = REGISTRY[source]()
        rows = plugin.fetch_for_date(session, target)
        n = persist_signals(session, rows)
        # Resolution pass: each plugin's signals get their appropriate
        # binder.  Same session, single commit so partial failures
        # roll back cleanly.
        bound = resolve_unbound_signals(session, target, source_name=source)
        if source == "tjk_discipline":
            bound += bind_discipline_to_entries(session, target)
        if source == "tjk_workouts":
            bound += bind_workouts_to_entries(session, target)
        if source in ("tjk_track_conditions", "tjk_steward_reports"):
            bound += bind_track_conditions_to_races(session, target)
        session.commit()
    finally:
        session.close()

    typer.echo(
        f"{source} → {n} signal row(s) persisted, "
        f"{bound} bound to TJK entries for {target}"
    )


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@predict_app.callback(invoke_without_command=True)
def predict(
    race_id: int = typer.Argument(None, help="Race ID to predict"),
    today: bool = typer.Option(False, "--today", help="Predict all today's races"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format"),
    model: str = typer.Option(
        "ml", "--model",
        help="Predictor to use: 'ml' (LightGBM ranker, default), 'bayesian' "
             "(hand-tuned fallback), or 'ensemble' (every saved model "
             "voting; convergence-based pick).",
    ),
) -> None:
    """Generate race predictions."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    if race_id is not None:
        _predict_race(race_id, json_output, model)
    elif today:
        _predict_today(json_output, model)
    else:
        typer.echo("Provide a race_id or use --today. See --help.")


def _build_predictor(session, model: str):
    """Resolve ``--model`` flag to the right predictor instance."""
    if model == "bayesian":
        from ganyan.predictor import BayesianPredictor
        return BayesianPredictor(session)
    if model == "ml":
        from ganyan.predictor.ml import MLPredictor
        return MLPredictor(session)
    if model == "ensemble":
        from ganyan.predictor.ml.ensemble import EnsemblePredictor

        class _EnsembleAdapter:
            """Wrap EnsemblePredictor to match the predict / predict_and_save API."""
            def __init__(self, sess):
                self.inner = EnsemblePredictor(sess)
            def predict(self, race_id):
                return self.inner.predict_as_predictions(race_id)
            def predict_and_save(self, race_id):
                # Persist by reusing MLPredictor's audit-row writer with
                # the ensemble's "model_version".
                from ganyan.db.models import RaceEntry, Prediction as PRow
                preds = self.predict(race_id)
                entries = {
                    (e.race_id, e.horse_id): e
                    for e in self.inner.session.query(RaceEntry)
                    .filter(RaceEntry.race_id == race_id).all()
                }
                version = (
                    f"ensemble-{len(self.inner.models)}heads"
                )
                for p in preds:
                    entry = entries.get((race_id, p.horse_id))
                    if entry is None:
                        continue
                    entry.predicted_probability = p.probability
                    self.inner.session.add(
                        PRow(
                            race_entry_id=entry.id,
                            model_version=version,
                            probability=p.probability,
                            confidence=p.confidence,
                            factors=p.contributing_factors,
                        )
                    )
                return preds
        return _EnsembleAdapter(session)
    raise typer.BadParameter(
        f"Unknown model: {model!r}. Use 'bayesian', 'ml', or 'ensemble'.",
    )


def _predict_race(race_id: int, json_output: bool, model: str) -> None:
    """Predict a single race and save predictions to DB."""
    from ganyan.db import get_session

    session = get_session()
    try:
        predictor = _build_predictor(session, model)
        predictions = predictor.predict_and_save(race_id)
        if not predictions:
            typer.echo(f"No predictions for race {race_id}.")
            return
        session.commit()
        _display_predictions(race_id, json_output)
    finally:
        session.close()


def _predict_today(json_output: bool, model: str) -> None:
    """Predict all races scheduled for today."""
    from ganyan.db import get_session, Race, RaceStatus

    session = get_session()
    try:
        races = (
            session.query(Race)
            .filter(Race.date == date.today(), Race.status == RaceStatus.scheduled)
            .all()
        )
        if not races:
            typer.echo("No races found for today.")
            return

        predictor = _build_predictor(session, model)
        for race in races:
            predictions = predictor.predict_and_save(race.id)
            _display_predictions(race.id, json_output)
            typer.echo("")  # blank line separator
        # Refresh picks so /advice reflects the fresh predictions rather
        # than a stale morning snapshot.  Graded picks are preserved.
        from ganyan.predictor.picks import refresh_picks_for_date
        added = refresh_picks_for_date(session, date.today())
        typer.echo(f"Refreshed picks: {added} new pick(s) written.")
        session.commit()
    finally:
        session.close()


def _display_predictions(race_id: int, json_output: bool) -> None:
    """Display predictions for a single race."""
    from ganyan.db import get_session, ViewPrediction
    session = get_session()

    try:
        _predictions = (
            session.query(ViewPrediction)
            .filter(ViewPrediction.race_id == race_id)
        )
        
        info = _predictions.first()
        predictions = _predictions.all()
    finally:
        session.close()
        
    if json_output:
        import json

        data = [
            {
                "horse_id": p.horse_id,
                "horse_name": p.horse_name,
                "probability": round(p.probability, 2),
                "confidence": round(p.confidence, 2),
                "factors": p.contributing_factors,
            }
            for p in predictions
        ]
        typer.echo(json.dumps({"race_id": race_id, "predictions": data}, indent=2))
    else:
        typer.echo(f"{info.track_name} - #{info.race_number} ({info.race_date.strftime('%d.%m.%Y')} @ {info.race_time})")
        typer.echo(f"{info.horse_type}, {info.race_type}")
        typer.echo("")
        typer.echo(f"{'#':<4} {'Horse':<25} {'Prob %':<10} {'Conf':<8}")
        typer.echo("-" * 50)
        for i, p in enumerate(predictions, 1):
            typer.echo(
                f"{i:<4} {p.horse_name:<25} {p.probability:>6.1f}%    {p.confidence:.2f}"
            )


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@evaluate_app.callback(invoke_without_command=True)
def evaluate(
    detail: bool = typer.Option(False, "--detail", help="Show per-race breakdown"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format"),
    cutoff: str | None = typer.Option(
        None, "--cutoff",
        help="Only evaluate races on/after this date (YYYY-MM-DD); acts as temporal holdout.",
    ),
    calibration_bins: int = typer.Option(
        10, "--bins", help="Number of calibration buckets for the reliability diagram.",
    ),
) -> None:
    """Evaluate prediction accuracy on resulted races."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from ganyan.db import get_session
    from ganyan.predictor.evaluate import evaluate_all

    cutoff_date = date.fromisoformat(cutoff) if cutoff else None

    session = get_session()
    try:
        summary, evaluations = evaluate_all(
            session, cutoff_date=cutoff_date, num_calibration_bins=calibration_bins,
        )
        _display_evaluation(summary, evaluations, detail, json_output)
    finally:
        session.close()


def _display_evaluation(summary, evaluations, detail: bool, json_output: bool) -> None:
    """Display evaluation results."""
    if json_output:
        import json

        data = {
            "summary": {
                "total_races": summary.total_races,
                "top1_accuracy": round(summary.top1_accuracy, 2),
                "top3_accuracy": round(summary.top3_accuracy, 2),
                "avg_winner_rank": round(summary.avg_winner_rank, 2),
                "avg_winner_probability": round(summary.avg_winner_probability, 2),
                "log_loss": round(summary.log_loss, 4),
                "brier_score": round(summary.brier_score, 4),
                "random_baseline_top1": round(summary.random_baseline_top1, 2),
                "agf_baseline_top1": (
                    round(summary.agf_baseline_top1, 2)
                    if summary.agf_baseline_top1 is not None else None
                ),
                "roi_simulation": round(summary.roi_simulation, 4),
                "cutoff_date": (
                    summary.cutoff_date.isoformat() if summary.cutoff_date else None
                ),
                "calibration": [
                    {
                        "lower": round(b.lower, 2),
                        "upper": round(b.upper, 2),
                        "count": b.count,
                        "mean_predicted": round(b.mean_predicted, 2),
                        "actual_win_rate": round(b.actual_win_rate, 2),
                    }
                    for b in summary.calibration
                ],
            },
        }
        if detail:
            data["races"] = [
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
            ]
        typer.echo(json.dumps(data, indent=2))
        return

    if summary.total_races == 0:
        typer.echo("No resulted races with predictions found.")
        return

    typer.echo("=== Prediction Evaluation Summary ===")
    if summary.cutoff_date is not None:
        typer.echo(f"Cutoff (holdout):      {summary.cutoff_date.isoformat()}")
    typer.echo(f"Total races evaluated: {summary.total_races}")
    typer.echo(f"Top-1 accuracy:        {summary.top1_accuracy:.1f}%")
    typer.echo(f"  Random baseline:     {summary.random_baseline_top1:.1f}%")
    if summary.agf_baseline_top1 is not None:
        typer.echo(f"  AGF (market) base.:  {summary.agf_baseline_top1:.1f}%")
    typer.echo(f"Top-3 accuracy:        {summary.top3_accuracy:.1f}%")
    typer.echo(f"Avg winner rank:       {summary.avg_winner_rank:.2f}")
    typer.echo(f"Avg winner prob:       {summary.avg_winner_probability:.1f}%")
    typer.echo(f"Log loss:              {summary.log_loss:.4f}")
    typer.echo(f"Brier score:           {summary.brier_score:.4f}")
    typer.echo(f"ROI (AGF-implied):     {summary.roi_simulation:+.1%}")

    if summary.calibration:
        typer.echo("")
        typer.echo("Calibration (reliability diagram):")
        typer.echo(f"  {'Bucket':<14} {'N':>5} {'Pred%':>7} {'Actual%':>8}")
        for b in summary.calibration:
            typer.echo(
                f"  {b.lower:>5.1f}-{b.upper:<6.1f} {b.count:>5} "
                f"{b.mean_predicted:>7.2f} {b.actual_win_rate:>8.2f}"
            )

    if detail:
        typer.echo("")
        typer.echo(
            f"{'Race':<6} {'Track':<15} {'Date':<12} {'#':<4} "
            f"{'Horses':<7} {'Winner':<20} {'Prob%':<8} {'Rank':<6} "
            f"{'Top1':<6} {'Top3':<6}"
        )
        typer.echo("-" * 95)
        for ev in evaluations:
            prob_str = f"{ev.winner_predicted_prob:.1f}" if ev.winner_predicted_prob is not None else "N/A"
            rank_str = str(ev.winner_predicted_rank) if ev.winner_predicted_rank is not None else "N/A"
            t1 = "Y" if ev.top1_correct else ""
            t3 = "Y" if ev.top3_correct else ""
            typer.echo(
                f"{ev.race_id:<6} {ev.track:<15} {ev.date.isoformat():<12} "
                f"{ev.race_number:<4} {ev.num_horses:<7} {ev.winner_name:<20} "
                f"{prob_str:<8} {rank_str:<6} {t1:<6} {t3:<6}"
            )


# ---------------------------------------------------------------------------
# races
# ---------------------------------------------------------------------------


@races_app.callback(invoke_without_command=True)
def races(
    today: bool = typer.Option(False, "--today", help="Show today's races"),
    race_date: str = typer.Option(
        None, "--date", help="Show races for date (YYYY-MM-DD)"
    ),
) -> None:
    """View race information."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    if today:
        target_date = date.today()
    elif race_date:
        target_date = datetime.strptime(race_date, "%Y-%m-%d").date()
    else:
        typer.echo("Use --today or --date YYYY-MM-DD. See --help.")
        return

    _list_races(target_date)


def _list_races(target_date: date) -> None:
    """List races for a given date."""
    from ganyan.db import get_session, Race

    session = get_session()
    try:
        race_list = (
            session.query(Race)
            .filter(Race.date == target_date)
            .order_by(Race.track_id, Race.race_number)
            .all()
        )
        if not race_list:
            typer.echo(f"No races found for {target_date}.")
            return

        typer.echo(f"Races for {target_date}:")
        typer.echo(
            f"{'ID':<6} {'Track':<15} {'#':<4} {'Dist':<8} {'Surface':<10} "
            f"{'Entries':<8} {'Status':<10}"
        )
        typer.echo("-" * 65)
        for race in race_list:
            track_name = race.track.name if race.track else "?"
            entry_count = len(race.entries)
            typer.echo(
                f"{race.id:<6} {track_name:<15} {race.race_number:<4} "
                f"{race.distance_meters or '?':<8} {race.surface or '?':<10} "
                f"{entry_count:<8} {race.status.value:<10}"
            )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# db
# ---------------------------------------------------------------------------


@db_app.command("init")
def db_init() -> None:
    """Initialize the database (run alembic upgrade head)."""
    typer.echo("Running database migrations...")
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        typer.echo("Database initialized successfully.")
    else:
        typer.echo(f"Migration failed:\n{result.stderr}", err=True)
        raise typer.Exit(code=1)


@db_app.command("reset")
def db_reset() -> None:
    """Reset the database (downgrade to base, then upgrade to head)."""
    confirm = typer.confirm("This will destroy all data. Continue?")
    if not confirm:
        typer.echo("Aborted.")
        raise typer.Exit()

    typer.echo("Downgrading database...")
    down = subprocess.run(
        ["alembic", "downgrade", "base"],
        capture_output=True,
        text=True,
    )
    if down.returncode != 0:
        typer.echo(f"Downgrade failed:\n{down.stderr}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Upgrading database...")
    up = subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
    )
    if up.returncode != 0:
        typer.echo(f"Upgrade failed:\n{up.stderr}", err=True)
        raise typer.Exit(code=1)

    typer.echo("Database reset successfully.")


# ---------------------------------------------------------------------------
# train (ML ranker)
# ---------------------------------------------------------------------------


# Default training window — the profitable strategy depends on
# recent AGF calibration, not long history.  90 days captures current
# jockey/trainer form and a reasonable sire-offspring sample; more
# history just adds stale signal without deepening the tree.
_DEFAULT_TRAIN_WINDOW_DAYS = 90


@train_app.callback(invoke_without_command=True)
def train(
    ctx: typer.Context,
    from_date: str = typer.Option(
        None, "--from",
        help=(
            "Earliest race date to include (YYYY-MM-DD).  "
            f"Default: today minus {_DEFAULT_TRAIN_WINDOW_DAYS} days.  "
            "Pass an explicit date (or --all-history) to override."
        ),
    ),
    to_date: str = typer.Option(
        None, "--to", help="Latest race date to include (YYYY-MM-DD)."
    ),
    all_history: bool = typer.Option(
        False, "--all-history",
        help="Train on every resulted race in the DB (bypasses the "
             f"{_DEFAULT_TRAIN_WINDOW_DAYS}-day default).  Slower, rarely useful.",
    ),
    holdout: float = typer.Option(
        0.2, "--holdout", help="Fraction of latest dates held out for eval."
    ),
    rounds: int = typer.Option(
        500, "--rounds", help="Maximum LightGBM boosting rounds."
    ),
    exclude_agf: bool = typer.Option(
        False, "--exclude-agf",
        help="Train WITHOUT AGF features (for value-betting comparisons).",
    ),
    objective: str = typer.Option(
        "rank", "--objective",
        help="'rank' (LambdaRank) or 'finish_time' (regression on actual "
             "race seconds — sort ascending = winner ranker, joins "
             "ensemble as 4th vote).",
    ),
    model_name: str = typer.Option(
        None, "--model-name",
        help="Filename stem for the saved model.  Default depends on "
             "objective: lightgbm_ranker / lightgbm_finish_time, with "
             "`_value` suffix when --exclude-agf.",
    ),
) -> None:
    """Fit a LightGBM model on resulted races and save to disk.

    Two objectives:

    - ``--objective rank`` (default) — LambdaRank predicting the winner.
      Top-1/Top-3 accuracy is the headline metric.
    - ``--objective finish_time`` — regression on actual race seconds.
      Sort ascending = winner ranker; joins the ensemble as a vote.
    """
    # When a subcommand (linear / cv / specialists) is invoked, the
    # callback still runs under ``invoke_without_command=True``; skip
    # the callback body so we don't train a rank model as a side effect.
    if ctx.invoked_subcommand is not None:
        return

    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from datetime import timedelta
    from ganyan.db import get_session
    from ganyan.predictor.ml import train_ranker

    if from_date is not None:
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
    elif all_history:
        start = None
    else:
        start = date.today() - timedelta(days=_DEFAULT_TRAIN_WINDOW_DAYS)
    end = datetime.strptime(to_date, "%Y-%m-%d").date() if to_date else None

    if objective not in {"rank", "finish_time"}:
        raise typer.BadParameter(
            f"objective must be 'rank' or 'finish_time' (got {objective!r})",
        )
    excluded = ["agf_edge", "agf_raw"] if exclude_agf else None
    if model_name is None:
        if objective == "finish_time":
            model_name = (
                "lightgbm_finish_time_value" if exclude_agf
                else "lightgbm_finish_time"
            )
        else:
            model_name = "lightgbm_value" if exclude_agf else "lightgbm_ranker"

    window = f"from {start}" if start else "all history"
    typer.echo(f"Training window: {window} → {end or 'today'}")

    session = get_session()
    try:
        result = train_ranker(
            session,
            from_date=start,
            to_date=end,
            holdout_fraction=holdout,
            num_boost_round=rounds,
            exclude_features=excluded,
            model_name=model_name,
            objective=objective,
        )
    finally:
        session.close()

    typer.echo("=== Training complete ===")
    typer.echo(f"Model saved to:   {result.model_path}")
    typer.echo(f"Metadata:         {result.metadata_path}")
    typer.echo(f"Train races:      {result.train_races}")
    typer.echo(f"Holdout races:    {result.test_races}")
    typer.echo("")
    typer.echo("Holdout metrics:")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            typer.echo(f"  {k:<22} {v:.3f}")
        else:
            typer.echo(f"  {k:<22} {v}")
    typer.echo("")
    typer.echo("Top feature importances (gain):")
    for i, (feat, gain) in enumerate(result.feature_importance.items()):
        if i >= 10:
            break
        typer.echo(f"  {feat:<22} {gain:>10.1f}")


# Race-type buckets a specialist will be trained on.  Prefixes so
# "Handikap 14 /H1" / "Handikap 16 /H2" / etc all fold into one
# Handikap specialist — feature importance across sub-categories is
# similar enough that one bucket has acceptable sample size.
_SPECIALIST_BUCKETS = [
    ("Handikap", "lightgbm_spec_handikap"),
    ("Maiden", "lightgbm_spec_maiden"),
    ("ŞARTLI", "lightgbm_spec_sartli"),
    ("KV", "lightgbm_spec_kv"),
    ("SATIŞ", "lightgbm_spec_satis"),
    ("G", "lightgbm_spec_stakes"),  # G 3 / group races
]


@train_app.command("linear")
def train_linear(
    family: str = typer.Option(
        "conditional_logit", "--family",
        help="'conditional_logit' (winner MLE, McFadden) or "
             "'plackett_luce' (top-k ordered finish MLE).",
    ),
    all_history: bool = typer.Option(
        True, "--all-history/--no-all-history",
        help="Train on every resulted race (default).",
    ),
    epochs: int = typer.Option(200, "--epochs", help="Adam epochs."),
    lr: float = typer.Option(0.05, "--lr", help="Adam learning rate."),
    top_k: int = typer.Option(3, "--top-k", help="Plackett-Luce top-k."),
    model_name: str = typer.Option(
        None, "--model-name", help="Filename stem. Default reflects family.",
    ),
) -> None:
    """Train a pure-numpy linear ranker (Conditional Logit or Plackett-
    Luce) as an additional ensemble head.  Saves a ``.npz`` + meta.json
    pair the predictor can reload; runs in seconds, adds methodological
    diversity (no shared tree bias with LightGBM heads)."""
    if family not in {"conditional_logit", "plackett_luce"}:
        raise typer.BadParameter(
            f"family must be 'conditional_logit' or 'plackett_luce' "
            f"(got {family!r})",
        )
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    from datetime import timedelta
    from ganyan.db import get_session
    from ganyan.predictor.ml.linear_ranker import train_conditional_logit

    start = (
        None if all_history
        else date.today() - timedelta(days=_DEFAULT_TRAIN_WINDOW_DAYS)
    )
    name = model_name or (
        "linear_conditional_logit" if family == "conditional_logit"
        else "linear_plackett_luce"
    )

    session = get_session()
    try:
        res = train_conditional_logit(
            session,
            from_date=start,
            epochs=epochs,
            lr=lr,
            plackett_luce=(family == "plackett_luce"),
            top_k=top_k,
            model_name=name,
        )
    finally:
        session.close()

    typer.echo(f"=== {family} training complete ===")
    typer.echo(f"Weights:          {res.model_path}")
    typer.echo(f"Metadata:         {res.metadata_path}")
    typer.echo(f"Train races:      {res.train_races}")
    typer.echo(f"Holdout races:    {res.test_races}")
    typer.echo("Metrics:")
    for k, v in res.metrics.items():
        if isinstance(v, float):
            typer.echo(f"  {k:<22} {v:.3f}")
        else:
            typer.echo(f"  {k:<22} {v}")
    typer.echo("")
    typer.echo("Top coefficients (|β|):")
    for i, (feat, coef) in enumerate(res.coefficients.items()):
        if i >= 10:
            break
        typer.echo(f"  {feat:<22} {coef:>+8.3f}")


@train_app.command("cv")
def train_cv(
    folds: int = typer.Option(5, "--folds", help="Number of walk-forward folds."),
    all_history: bool = typer.Option(
        True, "--all-history/--no-all-history",
        help=f"Train on every resulted race (default); off = {_DEFAULT_TRAIN_WINDOW_DAYS}-day window.",
    ),
    rounds: int = typer.Option(
        500, "--rounds", help="Max boosting rounds per fold.",
    ),
    race_type_prefix: str = typer.Option(
        None, "--race-type",
        help="Restrict CV to one race-type bucket (e.g. 'Handikap').",
    ),
) -> None:
    """Walk-forward K-fold cross-validation to get honest confidence
    intervals on top-1 accuracy.

    Fits one LambdaRank per fold, evaluates on that fold's held-out
    chunk.  Reports per-fold top-1/top-3 plus mean ± 95% CI across
    folds.  Answers "is our 43% top-1 a stable number or a lucky
    holdout?".
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from datetime import timedelta
    from ganyan.db import get_session
    from ganyan.predictor.ml.trainer import cross_validate_ranker

    start = (
        None if all_history
        else date.today() - timedelta(days=_DEFAULT_TRAIN_WINDOW_DAYS)
    )
    session = get_session()
    try:
        res = cross_validate_ranker(
            session,
            n_folds=folds,
            from_date=start,
            num_boost_round=rounds,
            race_type_prefix=race_type_prefix,
        )
    finally:
        session.close()

    typer.echo(f"=== Walk-forward CV ({res['n_folds']} folds) ===")
    typer.echo(f"{'fold':>4} {'train_races':>12} {'test_races':>11} {'top1%':>7} {'top3%':>7}")
    for m in res["per_fold"]:
        typer.echo(
            f"{m['fold']:>4} {m['train_races']:>12} {m['test_races']:>11} "
            f"{m['top1_accuracy']:>7.2f} {m['top3_accuracy']:>7.2f}",
        )
    typer.echo("")
    typer.echo(
        f"top-1: mean={res['top1_mean']:.2f}%  std={res['top1_std']:.2f}  "
        f"95% CI ± {res['top1_95_ci_halfwidth']:.2f}pp",
    )
    typer.echo(
        f"top-3: mean={res['top3_mean']:.2f}%  std={res['top3_std']:.2f}",
    )


@train_app.command("specialists")
def train_specialists(
    all_history: bool = typer.Option(
        True, "--all-history/--no-all-history",
        help="Train on every resulted race (default).  Off falls back to "
             f"{_DEFAULT_TRAIN_WINDOW_DAYS}-day window.",
    ),
    objective: str = typer.Option(
        "rank", "--objective",
        help="Passed through to each specialist.  'rank' (default), 'ev', "
             "or 'finish_time'.",
    ),
    rounds: int = typer.Option(
        500, "--rounds", help="Max boosting rounds per specialist.",
    ),
    min_races: int = typer.Option(
        200, "--min-races",
        help="Skip buckets with fewer resulted races than this — too noisy.",
    ),
) -> None:
    """Train one LightGBM model per race-type bucket.

    Produces ``lightgbm_spec_<bucket>.txt`` files that the ensemble
    auto-loads.  Each specialist sees only races whose ``race_type``
    starts with the bucket's prefix — so Handikap and Maiden models
    learn different weights, exploiting the 4× AGF-reliability spread
    across race types.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from datetime import timedelta
    from ganyan.db import get_session
    from ganyan.predictor.ml import train_ranker

    start = (
        None if all_history
        else date.today() - timedelta(days=_DEFAULT_TRAIN_WINDOW_DAYS)
    )

    results: list[tuple[str, object, str | None]] = []
    for prefix, name in _SPECIALIST_BUCKETS:
        typer.echo(f"--- specialist: {prefix} → {name} ---")
        session = get_session()
        try:
            try:
                result = train_ranker(
                    session,
                    from_date=start,
                    holdout_fraction=0.2,
                    num_boost_round=rounds,
                    model_name=name,
                    objective=objective,
                    race_type_prefix=prefix,
                )
            except RuntimeError as exc:
                typer.echo(f"  skipped ({exc})")
                results.append((prefix, None, str(exc)))
                continue
            if result.train_races < min_races:
                typer.echo(
                    f"  skipped: only {result.train_races} training races "
                    f"(< {min_races} threshold)"
                )
                results.append(
                    (prefix, result, f"too few training races: {result.train_races}"),
                )
                continue
            typer.echo(
                f"  trained on {result.train_races} races, "
                f"test={result.test_races}, metrics={result.metrics}",
            )
            results.append((prefix, result, None))
        finally:
            session.close()

    typer.echo("")
    typer.echo("=== Specialist summary ===")
    for prefix, result, error in results:
        if error:
            typer.echo(f"  {prefix:<10} SKIPPED: {error}")
        else:
            top1 = result.metrics.get("top1_accuracy", 0.0)
            top3 = result.metrics.get("top3_accuracy", 0.0)
            typer.echo(
                f"  {prefix:<10} n={result.train_races:>5}+{result.test_races:>4}"
                f"  top1={top1:>5.1f}%  top3={top3:>5.1f}%",
            )


# ---------------------------------------------------------------------------
# crawl (horse detail pages)
# ---------------------------------------------------------------------------


@crawl_app.command("horses")
def crawl_horses(
    limit: int = typer.Option(
        None, "--limit", help="Maximum horses to crawl in this run."
    ),
    concurrency: int = typer.Option(
        5, "--concurrency", help="Parallel HTTP fetches."
    ),
    delay: float = typer.Option(
        0.5, "--delay", help="Seconds between requests per worker."
    ),
) -> None:
    """Fetch pedigree for horses that have a tjk_at_id but no profile yet."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    async def _run() -> int:
        from ganyan.db import get_session
        from ganyan.scraper.horse_crawler import HorseCrawler

        session = get_session()
        try:
            async with HorseCrawler(
                session,
                base_url=settings.tjk_base_url,
                delay=delay,
                concurrency=concurrency,
            ) as crawler:
                return await crawler.crawl_missing_profiles(limit=limit)
        finally:
            session.close()

    stored = asyncio.run(_run())
    typer.echo(f"Crawled {stored} horse profile(s).")


# ---------------------------------------------------------------------------
# value-picks (value-betting picks from the AGF-free model)
# ---------------------------------------------------------------------------


@value_app.callback(invoke_without_command=True)
def value_picks(
    race_date: str = typer.Option(
        None, "--date", help="Date to pick over (YYYY-MM-DD). Defaults to today.",
    ),
    race_id: int = typer.Option(None, "--race-id", help="Single race to score."),
    threshold: float = typer.Option(
        0.22, "--threshold",
        help=(
            "Minimum relative edge (model_prob - agf_prob) / agf_prob to flag. "
            "Default 0.22 covers the typical 18%% parimutuel takeout."
        ),
    ),
    min_agf: float = typer.Option(
        2.0, "--min-agf",
        help="Skip horses with AGF below this (avoids dividing by near-zero).",
    ),
    model_name: str = typer.Option(
        "lightgbm_value", "--model-name",
        help="Model file stem under models/.  Must be an AGF-free model.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """List horses the value model thinks are underpriced vs the AGF market."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    target = (
        datetime.strptime(race_date, "%Y-%m-%d").date() if race_date else date.today()
    )

    from ganyan.db import get_session
    from ganyan.db.models import Race, RaceEntry
    from ganyan.predictor.ml import MLPredictor, load_latest_model

    try:
        loaded = load_latest_model(model_name=model_name)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo(
            "Run `ganyan train --exclude-agf` first to build the value model.",
            err=True,
        )
        raise typer.Exit(code=1)

    if "agf_edge" in loaded.feature_columns or "agf_raw" in loaded.feature_columns:
        typer.echo(
            "Warning: the loaded model still sees AGF — edges will reflect "
            "circular self-agreement, not true value.",
            err=True,
        )

    session = get_session()
    try:
        predictor = MLPredictor(session, model=loaded)

        if race_id is not None:
            races = [session.get(Race, race_id)]
            races = [r for r in races if r is not None]
        else:
            races = (
                session.query(Race)
                .filter(Race.date == target)
                .order_by(Race.race_number.asc())
                .all()
            )
        if not races:
            typer.echo(f"No races found for {target}.")
            return

        all_picks: list[dict] = []
        for race in races:
            preds = predictor.predict(race.id)
            if not preds:
                continue
            # Build AGF lookup for this race.
            entries = {
                e.horse_id: e for e in
                session.query(RaceEntry).filter(RaceEntry.race_id == race.id).all()
            }
            for p in preds:
                entry = entries.get(p.horse_id)
                if entry is None or entry.agf is None:
                    continue
                agf_pct = float(entry.agf)
                if agf_pct < min_agf:
                    continue
                edge = (p.probability - agf_pct) / agf_pct
                if edge < threshold:
                    continue
                all_picks.append({
                    "race_id": race.id,
                    "track": race.track.name if race.track else "?",
                    "race_number": race.race_number,
                    "post_time": race.post_time,
                    "horse": p.horse_name,
                    "model_prob": round(p.probability, 2),
                    "agf": round(agf_pct, 2),
                    "edge_pct": round(edge * 100.0, 1),
                    "confidence": round(p.confidence, 2),
                })

        all_picks.sort(key=lambda x: x["edge_pct"], reverse=True)

        if json_output:
            import json
            typer.echo(json.dumps(all_picks, indent=2))
            return

        if not all_picks:
            typer.echo(
                f"No horses on {target} cleared the {threshold:.0%} edge threshold."
            )
            return

        typer.echo(
            f"=== Value picks for {target} "
            f"(threshold {threshold:.0%}) ==="
        )
        typer.echo(
            f"{'Race':<18} {'Horse':<25} {'Model%':>7} {'AGF%':>6} {'Edge%':>7}"
        )
        for pick in all_picks:
            label = (
                f"{pick['track']} R{pick['race_number']} "
                f"{pick['post_time'] or ''}"
            ).strip()
            typer.echo(
                f"{label:<18} {pick['horse']:<25} "
                f"{pick['model_prob']:>7.2f} {pick['agf']:>6.2f} "
                f"{pick['edge_pct']:>6.1f}%"
            )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# exotics — Harville-derived exotic-pool combinations
# ---------------------------------------------------------------------------

_EXOTIC_POOLS = {
    "ganyan", "plase", "ikili", "sirali-ikili", "uclu", "dortlu",
}


@app.command("exotics")
def exotics_cmd(
    race_id: int = typer.Argument(..., help="Race to score."),
    pool: str = typer.Option(
        "uclu", "--pool",
        help=(
            "Pool: ganyan | plase | ikili | sirali-ikili | uclu | dortlu"
        ),
    ),
    top_n: int = typer.Option(10, "--top-n", help="Show this many combinations."),
    plase_k: int = typer.Option(
        2, "--plase-k",
        help="For --pool plase: horse must finish in top K (2 or 3).",
    ),
    model: str = typer.Option(
        "bayesian", "--model",
        help="Win-probability source: 'bayesian' or 'ml'.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Rank exotic combinations for a race.

    Uses the chosen win-probability model (Ganyan) as input to the
    Harville conditional-probability model, which derives joint
    probabilities for multi-horse outcomes.
    """
    if pool not in _EXOTIC_POOLS:
        typer.echo(
            f"Unknown pool {pool!r}.  Choose one of: {sorted(_EXOTIC_POOLS)}",
            err=True,
        )
        raise typer.Exit(code=1)

    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from ganyan.db import get_session
    from ganyan.db.models import Race
    from ganyan.predictor.exotics import (
        cumulative_coverage, dortlu_probabilities,
        ganyan_probabilities, ikili_probabilities, plase_probabilities,
        sirali_ikili_probabilities, top_n as top_n_fn, uclu_probabilities,
    )

    session = get_session()
    try:
        race = session.get(Race, race_id)
        if race is None:
            typer.echo(f"Race {race_id} not found.", err=True)
            raise typer.Exit(code=1)

        predictor = _build_predictor(session, model)
        preds = predictor.predict(race_id)
        if not preds:
            typer.echo(f"No predictions available for race {race_id}.")
            return

        # Build horse_id → win probability mapping (normalized to sum 1).
        win_probs: dict[int, float] = {
            p.horse_id: p.probability / 100.0 for p in preds
        }
        name_for: dict[int, str] = {p.horse_id: p.horse_name for p in preds}

        if pool == "ganyan":
            combos = ganyan_probabilities(win_probs)
        elif pool == "plase":
            combos = plase_probabilities(win_probs, top_k=plase_k)
        elif pool == "ikili":
            combos = ikili_probabilities(win_probs)
        elif pool == "sirali-ikili":
            combos = sirali_ikili_probabilities(win_probs)
        elif pool == "uclu":
            combos = uclu_probabilities(win_probs)
        elif pool == "dortlu":
            combos = dortlu_probabilities(win_probs)
        else:
            raise AssertionError("unreachable")  # pragma: no cover

        shown = top_n_fn(combos, top_n)
        cum = cumulative_coverage(shown)

        if json_output:
            import json
            typer.echo(json.dumps({
                "race_id": race_id,
                "pool": pool,
                "combinations": [
                    {
                        "horses": list(c.horses),
                        "horse_names": [name_for.get(h, "?") for h in c.horses],
                        "probability": round(c.probability, 5),
                        "ordered": c.ordered,
                        "cumulative": round(cum[i], 5),
                    }
                    for i, c in enumerate(shown)
                ],
            }, indent=2))
            return

        track = race.track.name if race.track else "?"
        typer.echo(
            f"=== {pool} — {track} R{race.race_number} "
            f"({race.date}) ===",
        )
        if pool == "plase":
            typer.echo(f"top_k = {plase_k}")
        separator = " → " if pool in ("sirali-ikili", "uclu", "dortlu") else " + "
        typer.echo(f"{'#':<3} {'Prob':>7} {'Cum':>7}  Combination")
        for i, c in enumerate(shown, start=1):
            names = separator.join(name_for.get(h, "?") for h in c.horses)
            typer.echo(
                f"{i:<3} {c.probability*100:>6.2f}% {cum[i-1]*100:>6.2f}%  {names}"
            )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# exotics-backtest — back-test exotic-pool ROI against actual payouts
# ---------------------------------------------------------------------------


@app.command("exotics-backtest")
def exotics_backtest_cmd(
    pool: str = typer.Option(
        None, "--pool",
        help="Pool to backtest.  If omitted, all of ganyan/ikili/sirali-ikili/uclu.",
    ),
    top_n: int = typer.Option(
        None, "--top-n",
        help="Combinations per race to bet.  If omitted, sweep [1, 3, 6, 10].",
    ),
    from_date: str = typer.Option(
        None, "--from", help="Earliest race date (YYYY-MM-DD)."
    ),
    to_date: str = typer.Option(
        None, "--to", help="Latest race date (YYYY-MM-DD)."
    ),
    model: str = typer.Option(
        "bayesian", "--model",
        help="Win-probability source: 'bayesian' (recommended) or 'ml'.",
    ),
    stake: float = typer.Option(
        100.0, "--stake", help="Flat TL stake per ticket."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Back-test Harville-derived exotic-pool strategies vs real payouts."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from ganyan.db import get_session
    from ganyan.predictor.exotic_evaluate import (
        _COMBO_FUNCS, evaluate_all_pools,
    )

    valid_pools = list(_COMBO_FUNCS.keys())
    # normalise kebab-case
    if pool is not None:
        pool_norm = pool.replace("-", "_")
        if pool_norm not in valid_pools:
            typer.echo(
                f"Unknown pool {pool!r}. Choose from {valid_pools}.", err=True,
            )
            raise typer.Exit(code=1)
        pools = [pool_norm]
    else:
        pools = ["ganyan", "ikili", "sirali_ikili", "uclu"]

    top_ns = [top_n] if top_n is not None else [1, 3, 6, 10]
    start = datetime.strptime(from_date, "%Y-%m-%d").date() if from_date else None
    end = datetime.strptime(to_date, "%Y-%m-%d").date() if to_date else None

    def _factory(session):
        return _build_predictor(session, model)

    session = get_session()
    try:
        results = evaluate_all_pools(
            session,
            pools=pools,
            top_ns=top_ns,
            from_date=start,
            to_date=end,
            predictor_factory=_factory,
            ticket_stake_tl=stake,
        )
    finally:
        session.close()

    if json_output:
        import json
        typer.echo(json.dumps([r.summary_row() for r in results], indent=2))
        return

    typer.echo("=== Exotic-pool backtest ===")
    if start or end:
        typer.echo(
            f"Window: {start.isoformat() if start else '…'} → "
            f"{end.isoformat() if end else 'today'}"
        )
    typer.echo(f"Model: {model}   Ticket stake: {stake:.0f} TL")
    typer.echo("")
    typer.echo(
        f"{'Pool':<14} {'TopN':>5} {'Races':>6} {'Hits':>5} "
        f"{'Hit%':>6} {'Stake':>10} {'Payout':>12} {'ROI':>8}"
    )
    typer.echo("-" * 78)
    for r in results:
        typer.echo(
            f"{r.pool:<14} {r.top_n:>5} {r.races:>6} {r.hits:>5} "
            f"{r.hit_rate:>5.1f}% {r.total_stake_tl:>10,.0f} "
            f"{r.total_payout_tl:>12,.0f} {r.roi*100:>+7.1f}%"
        )


# ---------------------------------------------------------------------------
# uclu-picks — live/forward picker for the empirically validated edge
# ---------------------------------------------------------------------------


@app.command("uclu-picks")
def uclu_picks_cmd(
    race_date: str = typer.Option(
        None, "--date",
        help="Date of the card (YYYY-MM-DD).  Defaults to today.",
    ),
    top_n: int = typer.Option(
        1, "--top-n",
        help=(
            "Combinations per race to show.  Backtest edge is strongest "
            "at top_n=1 (+150% ROI); widening quickly pays down to -0%."
        ),
    ),
    stake: float = typer.Option(
        100.0, "--stake",
        help="Per-ticket TL stake (for bet-sizing display only).",
    ),
    model: str = typer.Option(
        "ml", "--model",
        help="Win-probability source: 'ml' (recommended) or 'bayesian'.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Top-N Üçlü (ordered trifecta) picks for every race on a date.

    Paper-trading tool for the backtested Harville-from-AGF strategy.
    Each line shows: post time, track, race, the predicted winning
    1-2-3 order, our model probability for that combination, and the
    AGF rank of each horse for context.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    target = (
        datetime.strptime(race_date, "%Y-%m-%d").date()
        if race_date else date.today()
    )

    from ganyan.db import get_session
    from ganyan.db.models import Race, RaceEntry
    from ganyan.predictor.exotics import uclu_probabilities

    session = get_session()
    try:
        predictor = _build_predictor(session, model)
        races = (
            session.query(Race)
            .filter(Race.date == target)
            .order_by(Race.post_time.asc(), Race.race_number.asc())
            .all()
        )
        if not races:
            typer.echo(f"No races scheduled for {target}.")
            return

        picks: list[dict] = []
        skipped = 0
        for race in races:
            entries = {
                e.horse_id: e for e in session.query(RaceEntry)
                .filter(RaceEntry.race_id == race.id).all()
            }
            if len(entries) < 3:
                skipped += 1
                continue

            preds = predictor.predict(race.id)
            if not preds:
                skipped += 1
                continue
            win_probs = {p.horse_id: p.probability / 100.0 for p in preds}
            name_for = {p.horse_id: p.horse_name for p in preds}

            # Also capture AGF rank per horse for "is this the obvious
            # combo or a contrarian one?" context.
            agf_ranked = sorted(
                [e for e in entries.values() if e.agf is not None],
                key=lambda e: float(e.agf), reverse=True,
            )
            agf_rank_by_id = {e.horse_id: i + 1 for i, e in enumerate(agf_ranked)}

            combos = uclu_probabilities(win_probs)[:top_n]
            for idx, c in enumerate(combos, start=1):
                agf_ranks = [agf_rank_by_id.get(h, "?") for h in c.horses]
                picks.append({
                    "race_id": race.id,
                    "date": race.date.isoformat(),
                    "post_time": race.post_time,
                    "track": race.track.name if race.track else "?",
                    "race_number": race.race_number,
                    "rank_within_race": idx,
                    "horses": [name_for.get(h, "?") for h in c.horses],
                    "horse_ids": list(c.horses),
                    "model_probability_pct": round(c.probability * 100.0, 3),
                    "agf_ranks": agf_ranks,
                    "stake_tl": stake,
                })

        if json_output:
            import json
            typer.echo(json.dumps({
                "date": target.isoformat(),
                "top_n": top_n,
                "stake_tl": stake,
                "races_covered": len(races) - skipped,
                "races_skipped": skipped,
                "picks": picks,
            }, indent=2))
            return

        if not picks:
            typer.echo(f"No Üçlü-eligible races on {target}.")
            return

        typer.echo(
            f"=== Üçlü picks for {target} "
            f"(top-{top_n}, stake {stake:.0f} TL/ticket, model={model}) ==="
        )
        typer.echo(
            "Backtest (2026 out-of-sample, 1,477 races):  hit rate ~5%,  "
            "ROI +150–800% per month.\n"
            "Single-day variance is brutal.  At 5%, probability of zero "
            "hits in 18 picks is ~40% — losing streaks of 3-5 days are "
            "normal.  Only the long run is positive.\n"
        )
        typer.echo(
            f"{'Post':<6} {'Race':<18} {'Ord.Pick (1→2→3)':<55} "
            f"{'Prob%':>6} {'AGF rk':>7}"
        )
        typer.echo("-" * 100)
        for p in picks:
            race_label = f"{p['track']} R{p['race_number']}"
            combo = " → ".join(p["horses"])
            agf_ranks = "/".join(str(r) for r in p["agf_ranks"])
            post = p["post_time"] or "—"
            typer.echo(
                f"{post:<6} {race_label:<18} {combo:<55} "
                f"{p['model_probability_pct']:>6.3f} {agf_ranks:>7}"
            )

        total_stake = len(picks) * stake
        typer.echo("")
        typer.echo(
            f"Total: {len(picks)} tickets × {stake:.0f} TL = "
            f"{total_stake:,.0f} TL stake"
        )
        # Expected results based on observed 5% hit rate at top-1:
        exp_hits = len(picks) * 0.05
        typer.echo(
            f"At 5% hit rate (backtest median), expect ~{exp_hits:.1f} "
            f"winning ticket(s) on this card."
        )
        if skipped:
            typer.echo(f"({skipped} race(s) skipped: missing predictions or field < 3)")
    finally:
        session.close()


@app.command("bet-picks")
def bet_picks_cmd(
    race_id: int = typer.Argument(
        None,
        help="Specific race ID. If omitted, lists every race on --date.",
    ),
    race_date: str = typer.Option(
        None, "--date",
        help="Date of the card (YYYY-MM-DD). Defaults to today.",
    ),
    track: str = typer.Option(
        None, "--track", help="Filter by track name (case-insensitive)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Per-race bet recommendations across every TJK structure.

    Shows the model's pick for: Tek/Ganyan, Plase, Sıralı İkili, İkili,
    Üçlü Sıralı, Üçlü Komple-3, Üçlü Sıralı Virgüllü, 4'lü Sıralı.
    For each, prints horses (gate NO), ticket count, birim TL, total TL,
    and the model's combination probability.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from ganyan.db import get_session
    from ganyan.db.models import Race, RaceEntry, Track
    from ganyan.predictor.bet_recommendations import compute_bet_recommendations
    from ganyan.predictor.ml.ensemble import EnsemblePredictor

    target = (
        datetime.strptime(race_date, "%Y-%m-%d").date()
        if race_date else date.today()
    )

    session = get_session()
    try:
        if race_id is not None:
            races = [session.get(Race, race_id)]
            if races[0] is None:
                typer.echo(f"Race {race_id} not found.")
                raise typer.Exit(1)
        else:
            q = session.query(Race).filter(Race.date == target)
            if track:
                q = q.join(Track).filter(Track.name.ilike(track))
            races = q.order_by(Race.post_time.asc(), Race.race_number.asc()).all()

        if not races:
            typer.echo(f"No races for {target}.")
            return

        try:
            predictor = EnsemblePredictor(session)
        except FileNotFoundError:
            typer.echo("No trained models found. Run `ganyan train` first.")
            raise typer.Exit(1)

        all_out: list[dict] = []
        for race in races:
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
            track_name = race.track.name if race.track else "?"
            all_out.append({
                "race_id": race.id,
                "track": track_name,
                "race_number": race.race_number,
                "post_time": (
                    race.post_time.isoformat()
                    if hasattr(race.post_time, "isoformat") else
                    str(race.post_time) if race.post_time else None
                ),
                "recommendations": [r.to_dict() for r in recs],
            })

        if json_output:
            typer.echo(json.dumps(all_out, ensure_ascii=False, indent=2))
            return

        for item in all_out:
            typer.echo(f"\n=== {item['track']} R{item['race_number']} (race_id={item['race_id']}) ===")
            for r in item["recommendations"]:
                horses = (
                    " → ".join(str(h) for h in r["horses"]) if r["separator"] == "→"
                    else " / ".join(str(h) for h in r["horses"]) if r["separator"] == "/"
                    else " ".join(str(h) for h in r["horses"])
                )
                if r.get("spread_groups"):
                    horses = " , ".join(
                        " | ".join(g) for g in r["spread_groups"]
                    )
                typer.echo(
                    f"  {r['label']:38s} {horses:30s}  "
                    f"{r['tickets']:>3} bilet × {r['birim_tl']:>4.1f} TL "
                    f"= {r['stake_tl']:>7.1f} TL  "
                    f"P=%{r['model_prob_pct']:>5.2f}"
                )
    finally:
        session.close()


@app.command("multi-picks")
def multi_picks_cmd(
    race_date: str = typer.Option(
        None, "--date",
        help="Date of the card (YYYY-MM-DD). Defaults to today.",
    ),
    track_name: str = typer.Option(
        ..., "--track",
        help="Track name (e.g. 'İstanbul', 'İzmir', 'Ankara').",
    ),
    start_race_no: int = typer.Option(
        1, "--start-race",
        help="First leg's race number on the program (default 1).",
    ),
    pool_type: str = typer.Option(
        "6li", "--pool-type",
        help="'5li' / '6li' / '7li' (default 6li).",
    ),
    max_tickets: int = typer.Option(
        512, "--max-tickets",
        help="Budget cap on total combinations (default 512).",
    ),
    persist: bool = typer.Option(
        False, "--persist",
        help="Write the coupon to multi_race_picks (idempotent on date+track+pool+strategy).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Generate a 5'lı / 6'lı / 7'lı GANYAN coupon for a track-day window.

    Asymmetric leg sizing: lock 1-2 horses where the model is confident,
    spread 4-6 where it's uncertain, then shrink the widest legs until
    the ticket-count product fits under --max-tickets.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    target = (
        datetime.strptime(race_date, "%Y-%m-%d").date()
        if race_date else date.today()
    )

    from ganyan.db import get_session
    from ganyan.predictor.multi_race_picks import (
        generate_coupon, persist_coupon,
    )

    session = get_session()
    try:
        draft = generate_coupon(
            session, target, track_name, start_race_no,
            pool_type=pool_type, max_tickets=max_tickets,
        )
        pick_id = None
        if persist:
            pick = persist_coupon(
                session, target, track_name, start_race_no, draft,
                pool_type=pool_type,
            )
            session.commit()
            pick_id = pick.id

        if json_output:
            import json
            typer.echo(json.dumps({
                "date": target.isoformat(),
                "track": track_name,
                "pool_type": pool_type,
                "start_race_no": start_race_no,
                "end_race_no": start_race_no + len(draft.kept_horses_per_leg) - 1,
                "kept_horses_per_leg": draft.kept_horses_per_leg,
                "conviction_per_leg": draft.conviction_per_leg,
                "total_tickets": draft.total_tickets,
                "persisted_pick_id": pick_id,
            }, indent=2))
            return

        leg_count = len(draft.kept_horses_per_leg)
        end_race_no = start_race_no + leg_count - 1
        typer.echo(
            f"=== {pool_type.upper()} GANYAN — {track_name} {target} "
            f"R{start_race_no}-R{end_race_no} ==="
        )
        typer.echo(
            f"{'Leg':<6} {'Race':<6} {'Top-1 prob':>11}  {'Width':>5}  Kept (gate NOs)"
        )
        typer.echo("-" * 72)
        for i, (kept, conv) in enumerate(
            zip(draft.kept_horses_per_leg, draft.conviction_per_leg)
        ):
            kept_str = ", ".join(str(g) for g in kept)
            typer.echo(
                f"{i+1:<6} R{start_race_no + i:<5} {conv*100:>9.1f}%   "
                f"{len(kept):>5}  {kept_str}"
            )
        typer.echo("")
        typer.echo(f"Total combinations: {draft.total_tickets}")
        if pick_id is not None:
            typer.echo(f"Persisted as multi_race_picks.id={pick_id}")
    finally:
        session.close()


@app.command("multi-grade")
def multi_grade_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Grade every pending MultiRacePick whose pool has resulted.

    Idempotent — runs on every results-poll already; this command is
    for manual catch-up after rescraping pool payouts.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from ganyan.db import get_session
    from ganyan.predictor.multi_race_picks import (
        grade_all_pending_multi, multi_strategy_summary,
    )

    session = get_session()
    try:
        n = grade_all_pending_multi(session)
        session.commit()
        summary = multi_strategy_summary(session)
    finally:
        session.close()

    if json_output:
        import json
        typer.echo(json.dumps({
            "graded_now": n, "summary": summary,
        }, indent=2, default=str))
        return

    typer.echo(f"Graded {n} multi-race pick(s).")
    if not summary:
        typer.echo("(no graded multi-race picks yet)")
        return
    typer.echo(
        f"\n{'Strategy':<20} {'N':>5} {'Hits':>5} "
        f"{'Stake TL':>12} {'Payout TL':>12} {'Net TL':>12} "
        f"{'Hit %':>6} {'ROI %':>7}"
    )
    typer.echo("-" * 96)
    for strat, row in summary.items():
        typer.echo(
            f"{strat:<20} {row['n']:>5} {row['hits']:>5} "
            f"{row['stake_tl']:>12.2f} {row['payout_tl']:>12.2f} "
            f"{row['net_tl']:>12.2f} {row['hit_rate_pct']:>6.2f} "
            f"{row['roi_pct']:>7.2f}"
        )


# ---------------------------------------------------------------------------
# daemon — standalone scheduler (no Flask)
# ---------------------------------------------------------------------------


@app.command("daemon")
def daemon_cmd() -> None:
    """Run the Ganyan scheduler in the foreground.

    Blocks forever running the four scheduled jobs (morning card scrape,
    results polling, weekly pedigree refresh, monthly retrain).
    Typical deployment: wrap this with launchd (macOS) or systemd
    (Linux) for auto-restart.  Use Ctrl-C to stop gracefully.
    """
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )

    from ganyan.scheduler import build_scheduler

    scheduler = build_scheduler(settings, blocking=True)
    typer.echo(
        f"Ganyan daemon starting.  Jobs: "
        f"{[j.id for j in scheduler.get_jobs()]}"
    )
    try:
        scheduler.start()  # blocks
    except (KeyboardInterrupt, SystemExit):
        typer.echo("Daemon stopping.")


# ---------------------------------------------------------------------------
# picks — bet-ledger summary + grade missing
# ---------------------------------------------------------------------------


@app.command("picks")
def picks_cmd(
    grade: bool = typer.Option(
        False, "--grade",
        help="Grade any ungraded picks whose races have resulted.",
    ),
    since: str = typer.Option(
        None, "--since", help="Only summarise picks on/after this date (YYYY-MM-DD).",
    ),
    strategy: str = typer.Option(
        None, "--strategy",
        help="Filter to a single strategy (uclu_box6 / sirali_ikili_top1, or any in STRATEGIES for reference).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output JSON."),
) -> None:
    """Summarise the picks ledger — your actual running ROI per strategy."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from datetime import datetime as _dt
    from ganyan.db import get_session
    from ganyan.predictor.picks import grade_all_pending, strategy_summary
    from ganyan.predictor.terminology import strategy_display

    since_date = _dt.strptime(since, "%Y-%m-%d") if since else None

    session = get_session()
    try:
        if grade:
            n = grade_all_pending(session)
            session.commit()
            typer.echo(f"Graded {n} pick(s).")
        summary = strategy_summary(session, strategy=strategy, since=since_date)
    finally:
        session.close()

    if json_output:
        import json
        typer.echo(json.dumps(summary, indent=2, default=str))
        return

    if not summary:
        typer.echo("No graded picks yet.")
        return

    # Betting strategies have positive historical ROI and are the ones
    # we actually stake.  Reference strategies are kept for "did we pick
    # the winner?" feedback but are known-losing structurally (takeout
    # eats the edge on public favourites) so they're shown separately.
    #
    # sirali_ikili_top1 was previously classified as "reference" based on
    # Bayesian-v3's track record (-11% ROI).  Post-rescrape head-to-head
    # on 303 held-out races showed LightGBM lifts it to +44% ROI, so it
    # is now a live betting strategy.
    # uclu_top1 dropped 2026-05-02: 0 hits on n=16 gated picks, ROI −100%.
    BETTING = {"uclu_box6", "sirali_ikili_top1"}
    REFERENCE = {"ganyan_top1", "uclu_top1"}

    header = (
        f"{'Strateji (TJK)':<28} {'N':>5} {'Hits':>5} {'Hit%':>6} "
        f"{'Stake':>11} {'Payout':>12} {'Net':>11} {'ROI':>8}"
    )

    def _fmt(strat_key: str, row: dict) -> str:
        label = strategy_display(strat_key, short=True) if strat_key in (
            "uclu_top1", "uclu_box6", "sirali_ikili_top1", "ganyan_top1"
        ) else strat_key
        return (
            f"{label:<28} {row['n']:>5} {row['hits']:>5} "
            f"{row['hit_rate_pct']:>5.1f}% "
            f"{row['stake_tl']:>11,.0f} {row['payout_tl']:>12,.0f} "
            f"{row['net_tl']:>11,.0f} {row['roi_pct']:>+7.1f}%"
        )

    betting_keys = sorted(k for k in summary if k in BETTING)
    reference_keys = sorted(k for k in summary if k in REFERENCE)
    other_keys = sorted(k for k in summary if k not in BETTING and k not in REFERENCE)

    if betting_keys:
        typer.echo("=== Betting strategies (real P&L) ===")
        typer.echo(header)
        typer.echo("-" * 85)
        agg_n = agg_hits = 0
        agg_stake = agg_payout = agg_net = 0.0
        for k in betting_keys:
            row = summary[k]
            typer.echo(_fmt(k, row))
            agg_n += row["n"]
            agg_hits += row["hits"]
            agg_stake += row["stake_tl"]
            agg_payout += row["payout_tl"]
            agg_net += row["net_tl"]
        if len(betting_keys) > 1:
            agg_hr = 100 * agg_hits / agg_n if agg_n else 0
            agg_roi = 100 * agg_net / agg_stake if agg_stake else 0
            typer.echo(_fmt("TOTAL (betting)", {
                "n": agg_n, "hits": agg_hits, "hit_rate_pct": agg_hr,
                "stake_tl": agg_stake, "payout_tl": agg_payout,
                "net_tl": agg_net, "roi_pct": agg_roi,
            }))

    if reference_keys:
        if betting_keys:
            typer.echo("")
        typer.echo("=== Reference strategies (display only — not staked) ===")
        typer.echo(header)
        typer.echo("-" * 85)
        for k in reference_keys:
            typer.echo(_fmt(k, summary[k]))

    if other_keys:
        typer.echo("")
        typer.echo("=== Other ===")
        typer.echo(header)
        typer.echo("-" * 85)
        for k in other_keys:
            typer.echo(_fmt(k, summary[k]))


@app.command("morning")
def morning_cmd(
    grade: bool = typer.Option(
        True, "--grade/--no-grade",
        help="Also grade any already-resulted races at the end (default: yes).",
    ),
) -> None:
    """One-shot morning flow: scrape today's cards → predict → generate picks.

    Mirrors what the scheduler's ``morning_card`` job does at 08:30.  Use
    when the scheduler hasn't fired yet (early morning, Postgres was
    down, fresh environment) so /advice has something to show.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from datetime import date as _date
    import asyncio
    from sqlalchemy import func

    from ganyan.db import get_session
    from ganyan.db.models import Race, RaceEntry, ScrapeStatus
    from ganyan.scraper import TJKClient, parse_race_card
    from ganyan.scraper.backfill import log_scrape, store_race_card
    from ganyan.predictor.ml import MLPredictor
    from ganyan.predictor.picks import generate_picks_for_race, grade_all_pending

    today = _date.today()

    async def _scrape() -> int:
        session = get_session()
        stored = 0
        try:
            async with TJKClient(
                base_url=settings.tjk_base_url, delay=settings.scrape_delay,
            ) as client:
                raw = await client.get_race_card(today)
                for card in raw:
                    parsed = parse_race_card(card)
                    store_race_card(session, parsed)
                    log_scrape(
                        session, today, parsed.track_name,
                        ScrapeStatus.success,
                    )
                    stored += 1
                session.commit()
        finally:
            session.close()
        return stored

    typer.echo(f"[1/3] scraping race cards for {today}...")
    try:
        n_cards = asyncio.run(_scrape())
        typer.echo(f"      stored {n_cards} race card(s)")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"      scrape failed: {exc}", err=True)
        raise typer.Exit(code=1)

    typer.echo("[2/3] generating ML predictions + picks for today's races...")
    session = get_session()
    picks_created = 0
    predicted = 0
    try:
        predictor = MLPredictor(session)
        races = (
            session.query(Race).join(RaceEntry)
            .filter(Race.date == today)
            .group_by(Race.id)
            .having(func.count(RaceEntry.id) >= 3)
            .all()
        )
        for r in races:
            try:
                predictor.predict_and_save(r.id)
                picks_created += len(generate_picks_for_race(session, r.id))
                session.commit()
                predicted += 1
            except Exception as exc:  # noqa: BLE001
                session.rollback()
                typer.echo(f"      race {r.id} failed: {exc}", err=True)
        typer.echo(f"      predicted {predicted} race(s), "
                   f"created {picks_created} pick(s)")

        if grade:
            typer.echo("[3/3] grading any already-resulted races...")
            n = grade_all_pending(session)
            session.commit()
            typer.echo(f"      graded {n} pick(s)")
        else:
            typer.echo("[3/3] grading skipped (--no-grade)")
    finally:
        session.close()

    typer.echo(f"\nmorning flow complete for {today}.  "
               f"Next: uv run ganyan advice")


@app.command("advice")
def advice_cmd(
    date_str: str = typer.Option(
        None, "--date", help="Target date (YYYY-MM-DD). Defaults to today.",
    ),
    min_prob: float = typer.Option(
        0.0, "--min-prob",
        help="Deprecated since 2026-05-02 (uclu_top1 retired from advice). "
             "Kept for back-compat; has no effect on current strategies.",
    ),
    bankroll: float = typer.Option(
        10000.0, "--bankroll",
        help="Bankroll in TL for Kelly-fraction stake sizing.",
    ),
    kelly_fraction: float = typer.Option(
        0.25, "--kelly",
        help="Kelly multiplier (0.25 = quarter-Kelly, the standard).",
    ),
    cohort_filter: bool = typer.Option(
        True, "--cohort-filter/--no-cohort-filter",
        help="Skip races in known-loss cohorts: Maiden /Dişi (-46%% ROI), "
             "field≥13 (-22%%), Şanlıurfa/Bursa (-27%%/-34%%). Default ON.",
    ),
    bayes_skip: bool = typer.Option(
        True, "--bayes-skip/--no-bayes-skip",
        help="Skip races where Bayes posterior says top-1 isn't strong enough. "
             "Holdout: gate at mean≥35%% lifts top-1 from 35%% to 50%%. "
             "Default ON when posterior file present.",
    ),
    bayes_min_prob: float = typer.Option(
        0.35, "--bayes-min-prob",
        help="Bayes top-1 mean threshold (0-1). Skip race if below.",
    ),
    bayes_min_lo5: float = typer.Option(
        0.20, "--bayes-min-lo5",
        help="Bayes top-1 5th-percentile threshold (0-1). Skip race if below.",
    ),
    bayes_posterior: str = typer.Option(
        "models/bayes_pl_v3", "--bayes-posterior",
        help="Path base (no suffix) to .nc/.indices.json posterior files. "
             "Default v3 (track-variant speed figures, +0.64pp top-1 vs v1).",
    ),
    trip_wire: bool = typer.Option(
        True, "--trip-wire/--no-trip-wire",
        help="Halt advice when today's avg top-1 prob deviates >sigma from "
             "90-day baseline. Catches silent feature-pipeline regressions "
             "(2026-04-19 cliff-style). Default ON.",
    ),
    trip_wire_sigma: float = typer.Option(
        2.0, "--trip-wire-sigma",
        help="Z-score threshold for trip-wire halt.",
    ),
    trip_wire_lookback: int = typer.Option(
        90, "--trip-wire-lookback",
        help="Days of history for the trip-wire baseline.",
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON output."),
) -> None:
    """Günün bahis tavsiyeleri — scheduler'ın ürettiği picks'i "bugün ne
    oynayım?" formatında gösterir.

    /picks panosu (geriye dönük ledger) vs advice (ileriye dönük
    "bugün bu bahisleri koy") farkını net tutmak için ayrı komut.
    Sadece BETTING stratejileri gösterilir (uclu_top1 / uclu_box6 /
    sirali_ikili_top1) — ganyan_top1 referans olduğu için dışarıda.
    """
    def _cohort_skip_reason(race) -> str | None:
        rt = (race.race_type or "")
        # Filly-maiden: -45.9% ROI on n=74 picks
        if "Maiden /Dişi" in rt or "Maiden Dişi" in rt:
            return "Maiden /Dişi (-46% ROI hist.)"
        # Large fields: -22% ROI on n=373, hit rate drops to 28%
        n_entries = len(race.entries)
        if n_entries >= 13:
            return f"field {n_entries}≥13 (-22% ROI hist.)"
        # Worst tracks: Şanlıurfa -27%, Bursa -34%
        track_name = race.track.name if race.track else ""
        if track_name in {"Şanlıurfa", "Bursa"}:
            return f"track {track_name} (heavy losing cohort)"
        return None
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from datetime import date as _date, datetime as _dt
    from pathlib import Path as _Path
    from ganyan.db import get_session
    from ganyan.db.models import Race, Pick, RaceStatus
    from ganyan.predictor.kelly import strategy_edge_stats, suggested_stake_tl
    from ganyan.predictor.terminology import strategy_display
    from sqlalchemy.orm import joinedload

    target_date = (
        _dt.strptime(date_str, "%Y-%m-%d").date() if date_str else _date.today()
    )
    # uclu_top1 dropped 2026-05-02: 0/16 hits on gated picks, ROI −100%.
    BETTING_STRATEGIES = ("uclu_box6", "sirali_ikili_top1")

    bayes_idata = bayes_frame = None
    bayes_speed_history = None
    bayes_workout_history = None
    bayes_pace_history = None
    bayes_posterior_path = _Path(bayes_posterior)
    if bayes_skip:
        nc_file = bayes_posterior_path.with_suffix(".nc")
        idx_file = bayes_posterior_path.with_suffix(".indices.json")
        if nc_file.exists() and idx_file.exists():
            try:
                from ganyan.predictor.bayes.trainer import load_posterior
                bayes_idata, bayes_frame = load_posterior(bayes_posterior_path)
            except Exception as e:  # noqa: BLE001
                typer.echo(
                    f"⚠️  Bayes posterior load failed ({e}); skipping bayes gate.",
                    err=True,
                )
                bayes_idata = bayes_frame = None
        else:
            typer.echo(
                f"⚠️  Bayes posterior not found at {bayes_posterior_path}; "
                f"skipping bayes gate. Train via "
                f"`uv run python logs/bayes_train.py`.",
                err=True,
            )

    def _bayes_top_pred(race) -> "BayesPrediction | None":
        if bayes_idata is None or bayes_frame is None:
            return None
        from ganyan.predictor.bayes.predictor import (
            predict_from_posterior,
        )
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
        }
        if bayes_speed_history is not None:
            from ganyan.predictor.speed_figures import horse_speed_score
            race_in["speeds"] = [
                horse_speed_score(bayes_speed_history, e.horse_id, race.date) or 0.0
                for e in entries
            ]
        if bayes_workout_history is not None:
            from ganyan.predictor.workouts import horse_workout_score
            race_in["workouts"] = [
                horse_workout_score(bayes_workout_history, e.horse_id, race.date) or 0.0
                for e in entries
            ]
        if bayes_pace_history is not None:
            from ganyan.predictor.pace import horse_pace_score
            race_in["paces"] = [
                horse_pace_score(bayes_pace_history, e.horse_id, race.date) or 0.0
                for e in entries
            ]
        try:
            preds = predict_from_posterior(bayes_idata, bayes_frame, race_in)
        except Exception:  # noqa: BLE001
            return None
        return preds[0] if preds else None

    def _bayes_skip_reason(top: "BayesPrediction | None") -> str | None:
        if top is None:
            return None
        if top.mean_prob < bayes_min_prob:
            return (
                f"Bayes top-1 mean {top.mean_prob:.0%}<"
                f"{bayes_min_prob:.0%}"
            )
        if top.lo_5 < bayes_min_lo5:
            return (
                f"Bayes top-1 lo_5 {top.lo_5:.0%}<"
                f"{bayes_min_lo5:.0%}"
            )
        return None

    from ganyan.predictor.trip_wire import compute_trip_wire, is_anomalous, is_halt
    from ganyan.predictor import halt_flag, rolling_pnl_halt

    session = get_session()
    try:
        # Halt-flag canary: mirrors web /advice. Once set, banner prints and
        # stake/kelly columns render as "—" (informational only).
        halt_state = halt_flag.is_halted()
        if halt_state is None:
            for _strat, _reason in rolling_pnl_halt.check_all_strategies(
                session, BETTING_STRATEGIES,
            ).items():
                if _reason:
                    halt_flag.set_halt(reason=_reason, source="rolling_pnl_halt")
                    halt_state = halt_flag.is_halted()
                    break
        if halt_state is not None:
            typer.secho(
                f"⚠️  ADVICE HALTED — informational only.\n"
                f"  reason: {halt_state['reason']}\n"
                f"  source: {halt_state['source']}",
                fg=typer.colors.YELLOW,
                err=True,
            )
        if bayes_idata is not None:
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
        if not races:
            typer.echo(f"{target_date} için yarış bulunamadı.")
            if target_date == _date.today():
                typer.echo("  → uv run ganyan morning "
                           "(scrape + predict + generate picks)")
            else:
                typer.echo(f"  → uv run ganyan scrape --backfill --rescrape "
                           f"--from {target_date} --to {target_date}")
            return

        trip_info = (
            compute_trip_wire(session, target_date, lookback_days=trip_wire_lookback)
            if trip_wire else None
        )
        if is_halt(trip_info, trip_wire_sigma):
            typer.echo(
                f"⚠️  TRIP-WIRE FIRED — model is UNDER-confident today.\n"
                f"  today avg top-1 prob: {trip_info['today_avg']:.1f}%\n"
                f"  {trip_info['n_baseline_days']}d baseline:    "
                f"{trip_info['baseline_mean']:.1f}% ± {trip_info['baseline_std']:.1f}%  "
                f"(z = {trip_info['z_score']:+.2f})\n"
                f"  Likely silent feature-pipeline regression. HALTING advice.\n"
                f"  → bypass with --no-trip-wire if you've verified the cause."
            )
            return
        if is_anomalous(trip_info, trip_wire_sigma):
            # Over-confident — note but don't halt. Could mean a new feature
            # shipped, the re-predict cron is concentrating late money, or
            # genuinely a sharper-than-usual card.
            typer.echo(
                f"ℹ️  Trip-wire note: model is OVER-confident today "
                f"(z = {trip_info['z_score']:+.2f}, "
                f"today {trip_info['today_avg']:.1f}% vs baseline "
                f"{trip_info['baseline_mean']:.1f}% ± {trip_info['baseline_std']:.1f}%). "
                f"Continuing — over-confidence isn't the failure mode the trip-wire halts on."
            )

        race_ids = [r.id for r in races]
        picks = (
            session.query(Pick)
            .filter(Pick.race_id.in_(race_ids),
                    Pick.strategy.in_(BETTING_STRATEGIES))
            .all()
        )
        picks_by_race: dict[int, dict[str, Pick]] = {}
        for p in picks:
            picks_by_race.setdefault(p.race_id, {})[p.strategy] = p

        # Actual top-3 for resulted races (so we can show hit/miss).
        # Resolve the winner's display name while the session is still
        # open so we don't hit a DetachedInstanceError later.
        results_by_race: dict[int, tuple[int, int, int]] = {}
        winner_name_by_race: dict[int, str] = {}
        for r in races:
            if r.status != RaceStatus.resulted:
                continue
            ordered = sorted(
                [e for e in r.entries if e.finish_position in (1, 2, 3)],
                key=lambda e: e.finish_position,
            )
            if len(ordered) >= 3:
                results_by_race[r.id] = (
                    ordered[0].horse_id, ordered[1].horse_id, ordered[2].horse_id,
                )
                winner_entry = ordered[0]
                winner_name_by_race[r.id] = (
                    winner_entry.horse.name if winner_entry.horse
                    else f"#{winner_entry.horse_id}"
                )

        # Bayes posterior predictions per race — compute while session
        # is open so e.horse / e.horse.sire lazy-loads still work.
        bayes_top_by_race: dict[int, "BayesPrediction"] = {}
        if bayes_idata is not None and bayes_frame is not None:
            for r in races:
                top = _bayes_top_pred(r)
                if top is not None:
                    bayes_top_by_race[r.id] = top
    finally:
        session.close()

    if json_output:
        import json
        out = []
        for race in races:
            rpicks = picks_by_race.get(race.id, {})
            if not rpicks:
                continue
            if cohort_filter:
                reason = _cohort_skip_reason(race)
                if reason:
                    out.append({
                        "race_id": race.id,
                        "track": race.track.name if race.track else None,
                        "race_number": race.race_number,
                        "post_time": race.post_time,
                        "skipped_cohort_reason": reason,
                    })
                    continue
            bayes_top = bayes_top_by_race.get(race.id)
            if bayes_skip:
                br = _bayes_skip_reason(bayes_top)
                if br:
                    out.append({
                        "race_id": race.id,
                        "track": race.track.name if race.track else None,
                        "race_number": race.race_number,
                        "post_time": race.post_time,
                        "skipped_bayes_reason": br,
                        "bayes_top1_mean": bayes_top.mean_prob if bayes_top else None,
                        "bayes_top1_lo5": bayes_top.lo_5 if bayes_top else None,
                        "bayes_top1_hi95": bayes_top.hi_95 if bayes_top else None,
                    })
                    continue
            kelly_mult_eff = kelly_fraction
            bayes_conf = None
            if bayes_top is not None:
                bayes_conf = max(0.0, min(1.0, (bayes_top.mean_prob - 0.35) / 0.15))
                kelly_mult_eff = kelly_fraction * (0.5 + 0.5 * bayes_conf)
            pick_data = {}
            for strat, p in rpicks.items():
                prob = float(p.model_prob_pct) if p.model_prob_pct else 0.0
                stake = float(p.stake_tl)
                kelly_tl = None
                calibrated_pct = None
                stats = edge_stats.get(strat)
                if stats and stats.avg_b > 0 and stats.avg_model_prob > 0:
                    cal_p = stats.calibrate(prob / 100.0)
                    calibrated_pct = round(cal_p * 100, 3)
                    kelly_tl = suggested_stake_tl(
                        win_prob=cal_p,
                        b=stats.avg_b,
                        bankroll_tl=bankroll,
                        base_stake_tl=stake,
                        kelly_multiplier=kelly_mult_eff,
                    )
                pick_data[strat] = {
                    "combination_names": p.combination_names,
                    # Halt suppresses stake/kelly; consumers see None.
                    "stake_tl": None if halt_state is not None else stake,
                    "kelly_suggested_tl": (
                        None if halt_state is not None
                        else (round(kelly_tl, 2) if kelly_tl is not None else None)
                    ),
                    "kelly_multiplier_effective": round(kelly_mult_eff, 4),
                    "bayes_confidence": round(bayes_conf, 3) if bayes_conf is not None else None,
                    "model_prob_pct": prob,
                    "calibrated_prob_pct": calibrated_pct,
                    "graded": p.graded,
                    "hit": p.hit,
                    "payout_tl": float(p.payout_tl) if p.payout_tl else None,
                }
            item = {
                "race_id": race.id,
                "track": race.track.name if race.track else None,
                "race_number": race.race_number,
                "post_time": race.post_time,
                "distance_meters": race.distance_meters,
                "status": race.status.value if race.status else None,
                "picks": pick_data,
                "bayes_top1_horse": bayes_top.horse_name if bayes_top else None,
                "bayes_top1_mean": bayes_top.mean_prob if bayes_top else None,
                "bayes_top1_lo5": bayes_top.lo_5 if bayes_top else None,
                "bayes_top1_hi95": bayes_top.hi_95 if bayes_top else None,
            }
            out.append(item)
        meta = {
            "date": str(target_date),
            "bankroll_tl": bankroll,
            "kelly_multiplier": kelly_fraction,
            "halt": halt_state,
            "strategy_edge_stats": {
                k: v.to_dict() for k, v in edge_stats.items()
            },
            "races": out,
        }
        typer.echo(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
        return

    # Text output
    typer.echo(f"=== {target_date} BET ADVICE ({len(races)} races) ===")
    typer.echo(
        "  Strateji kodları → TJK bahis türleri: "
        "uclu_top1=Üçlü Tek · uclu_box6=Üçlü Kutu 6 · "
        "sirali_ikili_top1=İkili Sıralı Tek"
    )
    typer.echo(
        "  ⚠️  Sıralı Üçlü Bahis tek kombinasyon için TJK min 20 TL; "
        "Kutu 6 için min ~12 TL.  (Kutu ≠ Komple/K toggle.)\n"
    )

    total_stake_advised = 0.0      # what you'd pay if every pool formed
    total_stake_effective = 0.0    # stakes actually resolved (graded) or still pending
    total_tickets = 0
    n_races_advised = 0
    n_races_with_any_hit = 0
    total_payout = 0.0
    skipped_cohort = 0              # races skipped by --cohort-filter
    skipped_bayes = 0               # races skipped by --bayes-skip
    n_no_pool = 0                  # picks for strategies without a published pool

    for race in races:
        rpicks = picks_by_race.get(race.id, {})
        if not rpicks:
            continue
        if cohort_filter:
            reason = _cohort_skip_reason(race)
            if reason:
                track = race.track.name if race.track else "?"
                post = race.post_time or "--:--"
                typer.echo(
                    f"{track:<10} R{race.race_number:<2} {post}  "
                    f"[SKIP cohort] {reason}\n"
                )
                skipped_cohort += 1
                continue
        bayes_top = bayes_top_by_race.get(race.id)
        if bayes_skip:
            br = _bayes_skip_reason(bayes_top)
            if br:
                track = race.track.name if race.track else "?"
                post = race.post_time or "--:--"
                typer.echo(
                    f"{track:<10} R{race.race_number:<2} {post}  "
                    f"[SKIP bayes] {br}\n"
                )
                skipped_bayes += 1
                continue
        result = results_by_race.get(race.id)
        status_badge = ""
        if race.status == RaceStatus.resulted:
            status_badge = " [resulted]"
        elif race.status == RaceStatus.scheduled:
            status_badge = " [scheduled]"

        track = race.track.name if race.track else "?"
        post = race.post_time or "--:--"
        header = (f"{track:<10} R{race.race_number:<2} {post}  "
                  f"{race.distance_meters or 0}m{status_badge}")
        typer.echo(header)
        kelly_mult_eff = kelly_fraction
        bayes_conf = None
        if bayes_top is not None:
            bayes_conf = max(0.0, min(1.0, (bayes_top.mean_prob - 0.35) / 0.15))
            kelly_mult_eff = kelly_fraction * (0.5 + 0.5 * bayes_conf)
            typer.echo(
                f"  bayes top-1: {bayes_top.horse_name[:22]:<22} "
                f"{bayes_top.mean_prob:.0%} "
                f"[{bayes_top.lo_5:.0%}–{bayes_top.hi_95:.0%}]  "
                f"conf={bayes_conf:.2f} → kelly×{kelly_mult_eff:.2f}"
            )

        race_had_hit = False
        for strat in BETTING_STRATEGIES:
            p = rpicks.get(strat)
            if p is None:
                continue

            prob = float(p.model_prob_pct) if p.model_prob_pct else 0.0
            names = " → ".join(p.combination_names) if p.combination_names else "?"
            if len(names) > 42:
                names = names[:40] + "…"
            stake = float(p.stake_tl)

            bet_mark = "       —" if halt_state is not None else f"{stake:>4.0f} TL"
            outcome = ""
            is_effective = True
            if p.graded:
                if p.hit:
                    race_had_hit = True
                    pay = float(p.payout_tl or 0)
                    outcome = f"  ✓ HIT  payout {pay:,.0f}"
                    total_payout += pay
                else:
                    outcome = "  ✗ miss"
            elif race.status == RaceStatus.resulted:
                # Resulted but ungraded → pool wasn't published (no bet
                # could've been placed / TJK would refund).  Don't count
                # in the effective stake.
                outcome = "  · no-pool (refunded)"
                is_effective = False
                n_no_pool += 1
            else:
                outcome = "  (pending)"

            # Kelly-suggested stake per race.  We calibrate the raw
            # Harville joint prob first so Kelly gets a usable p: the
            # model's joint probs are systematically compressed (mean
            # ~1% when the empirical hit rate is 4%), so raw-in Kelly
            # would skip every bet.  Calibration preserves per-race
            # ranking while aligning the level with history — strong
            # picks get more Kelly, weak picks stay at skip.
            stats = edge_stats.get(strat)
            kelly_label = ""
            calibrated_pct = None
            if stats and stats.avg_b > 0 and stats.avg_model_prob > 0:
                calibrated_p = stats.calibrate(prob / 100.0)
                calibrated_pct = calibrated_p * 100
                suggested = suggested_stake_tl(
                    win_prob=calibrated_p,
                    b=stats.avg_b,
                    bankroll_tl=bankroll,
                    base_stake_tl=stake,
                    kelly_multiplier=kelly_mult_eff,
                )
                if halt_state is not None:
                    kelly_label = "  kelly: —"
                elif suggested <= 0:
                    kelly_label = "  kelly: skip"
                else:
                    kelly_label = f"  kelly: {suggested:,.0f} TL"

            strat_tjk = strategy_display(strat, short=True)
            strat_col = f"{strat_tjk} ({strat})"
            typer.echo(
                f"  [BET ] {strat_col:<34} {names}  prob {prob:>4.1f}%  "
                f"{bet_mark}{kelly_label}{outcome}"
            )
            total_stake_advised += stake
            if is_effective:
                total_stake_effective += stake
            total_tickets += int(p.ticket_count or 1)

        if race.status == RaceStatus.resulted and result:
            winner_name = winner_name_by_race.get(race.id, f"#{result[0]}")
            typer.echo(f"         actual winner: {winner_name}")

        if race_had_hit:
            n_races_with_any_hit += 1
        n_races_advised += 1
        typer.echo("")

    # Summary
    typer.echo("--- SUMMARY ---")
    typer.echo(f"advised races          : {n_races_advised} / {len(races)}")
    typer.echo(f"total advised tickets  : {total_tickets}")
    typer.echo(f"advised stake (gross)  : {total_stake_advised:,.0f} TL")
    if n_no_pool:
        typer.echo(f"no-pool refunded       : {n_no_pool} picks "
                   f"(-{total_stake_advised - total_stake_effective:,.0f} TL stake removed)")
    typer.echo(f"effective stake        : {total_stake_effective:,.0f} TL")
    any_graded = any(
        p.graded for rpicks in picks_by_race.values() for p in rpicks.values()
    )
    if any_graded:
        net = total_payout - total_stake_effective
        roi = 100 * net / total_stake_effective if total_stake_effective else 0
        sign = "+" if roi >= 0 else ""
        typer.echo(f"payout (graded hits)   : {total_payout:,.0f} TL")
        typer.echo(f"net P&L                : {net:+,.0f} TL "
                   f"({sign}{roi:.1f}% ROI on effective stake)")
        typer.echo(f"races with any hit     : {n_races_with_any_hit}")
    if skipped_cohort:
        typer.echo(f"skipped (cohort filter): {skipped_cohort} races "
                   f"(Maiden /Dişi · field≥13 · Şanlıurfa/Bursa)")
    if skipped_bayes:
        typer.echo(
            f"skipped (bayes gate)   : {skipped_bayes} races "
            f"(top-1 mean<{bayes_min_prob:.0%} or lo_5<{bayes_min_lo5:.0%})"
        )


@app.command("tune-thresholds")
def tune_thresholds_cmd(
    lookback_days: int = typer.Option(
        90, "--lookback-days",
        help="How many days of graded picks to sweep over.",
    ),
    strategy: str = typer.Option(
        None, "--strategy",
        help="Limit tuning to one strategy (default: all betting strategies).",
    ),
    min_sample: int = typer.Option(
        30, "--min-sample",
        help="Require at least this many graded picks above a threshold "
             "before considering that threshold (avoid single-pick noise).",
    ),
) -> None:
    """Find the ``--min-prob`` threshold per strategy that maximises
    historical ROI on the ledger.

    Walks every graded Pick in the window, sorts by ``model_prob_pct``,
    and for each candidate threshold computes cumulative ROI on the
    subset of picks at or above it.  Reports the threshold with the
    highest ROI subject to ``--min-sample`` races remaining above it.

    Output is both human-readable and machine-parseable — callers can
    use the recommended thresholds to feed ``ganyan advice --min-prob``
    or wire them into a config.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from datetime import date as _date, timedelta
    from ganyan.db import get_session
    from ganyan.db.models import Pick, Race

    since = _date.today() - timedelta(days=lookback_days)

    BETTING_STRATEGIES = ("uclu_top1", "uclu_box6", "sirali_ikili_top1")
    strategies = [strategy] if strategy else list(BETTING_STRATEGIES)

    session = get_session()
    try:
        for strat in strategies:
            rows = (
                session.query(Pick)
                .join(Race, Race.id == Pick.race_id)
                .filter(
                    Pick.strategy == strat,
                    Pick.graded == True,  # noqa: E712
                    Race.date >= since,
                )
                .all()
            )
            if not rows:
                typer.echo(f"{strat}: no graded picks in last {lookback_days} days.")
                continue

            # Pre-compute (prob, stake, payout) tuples sorted by prob desc.
            data = sorted([
                (
                    float(p.model_prob_pct or 0),
                    float(p.stake_tl),
                    float(p.payout_tl or 0) if p.hit else 0.0,
                )
                for p in rows
            ], key=lambda t: -t[0])

            # Evaluate thresholds: one per observed prob value.  At
            # threshold ``t``, the subset is rows[:i] where all
            # probs >= t.  Walk the sorted list once, accumulating.
            best = None  # (threshold, n, stake, payout, roi)
            cum_stake = cum_payout = 0.0
            for i, (prob, stake, payout) in enumerate(data, start=1):
                cum_stake += stake
                cum_payout += payout
                if i < min_sample:
                    continue
                if cum_stake <= 0:
                    continue
                roi = (cum_payout - cum_stake) / cum_stake * 100
                if best is None or roi > best["roi"]:
                    best = {
                        "threshold_pct": prob,
                        "n_above": i,
                        "stake": cum_stake,
                        "payout": cum_payout,
                        "net": cum_payout - cum_stake,
                        "roi": roi,
                    }

            # Baseline: no threshold (bet every graded pick).
            total_stake = sum(s for _, s, _ in data)
            total_payout = sum(p for _, _, p in data)
            baseline_roi = (
                (total_payout - total_stake) / total_stake * 100
                if total_stake > 0 else 0
            )

            typer.echo(f"=== {strat} ({len(rows)} graded picks, "
                       f"last {lookback_days}d) ===")
            typer.echo(f"  baseline (no threshold): stake {total_stake:>10,.0f}  "
                       f"payout {total_payout:>10,.0f}  ROI {baseline_roi:+6.1f}%")
            if best is None:
                typer.echo(f"  not enough data for --min-sample {min_sample}")
            else:
                delta = best["roi"] - baseline_roi
                typer.echo(f"  best --min-prob = {best['threshold_pct']:.2f}%")
                typer.echo(f"    keeps {best['n_above']}/{len(rows)} picks")
                typer.echo(f"    stake {best['stake']:>10,.0f}  "
                           f"payout {best['payout']:>10,.0f}  "
                           f"net {best['net']:+>10,.0f}  ROI {best['roi']:+6.1f}%  "
                           f"(Δ {delta:+.1f}pp)")
            typer.echo("")
    finally:
        session.close()
