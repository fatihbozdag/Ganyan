"""Ganyan CLI — Turkish horse racing prediction system."""

import asyncio
import logging
import subprocess
from datetime import date, datetime

import typer

from ganyan.config import get_settings

app = typer.Typer(name="ganyan", help="Turkish horse racing prediction system")
scrape_app = typer.Typer(help="Scrape race data from TJK")
predict_app = typer.Typer(help="Generate race predictions")
evaluate_app = typer.Typer(help="Evaluate prediction accuracy")
races_app = typer.Typer(help="View race information")
db_app = typer.Typer(help="Database management")
train_app = typer.Typer(help="Train the ML ranker model")
crawl_app = typer.Typer(help="Crawl per-horse detail pages (pedigree, etc.)")

app.add_typer(scrape_app, name="scrape")
app.add_typer(predict_app, name="predict")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(races_app, name="races")
app.add_typer(db_app, name="db")
app.add_typer(train_app, name="train")
app.add_typer(crawl_app, name="crawl")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# scrape
# ---------------------------------------------------------------------------


@scrape_app.callback(invoke_without_command=True)
def scrape(
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
        asyncio.run(_run_backfill(settings, start))
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


async def _run_backfill(settings, from_date: date) -> None:
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
            await manager.backfill(from_date=from_date)
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


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@predict_app.callback(invoke_without_command=True)
def predict(
    race_id: int = typer.Argument(None, help="Race ID to predict"),
    today: bool = typer.Option(False, "--today", help="Predict all today's races"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format"),
    model: str = typer.Option(
        "bayesian", "--model",
        help="Predictor to use: 'bayesian' (hand-tuned) or 'ml' (LightGBM ranker).",
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
    raise typer.BadParameter(f"Unknown model: {model!r}. Use 'bayesian' or 'ml'.")


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
        _display_predictions(predictions, race_id, json_output)
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
            _display_predictions(predictions, race.id, json_output)
            typer.echo("")  # blank line separator
        session.commit()
    finally:
        session.close()


def _display_predictions(predictions, race_id: int, json_output: bool) -> None:
    """Display predictions for a single race."""
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
        typer.echo(f"Race {race_id} predictions:")
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


@train_app.callback(invoke_without_command=True)
def train(
    from_date: str = typer.Option(
        None, "--from", help="Earliest race date to include (YYYY-MM-DD)."
    ),
    to_date: str = typer.Option(
        None, "--to", help="Latest race date to include (YYYY-MM-DD)."
    ),
    holdout: float = typer.Option(
        0.2, "--holdout", help="Fraction of latest dates held out for eval."
    ),
    rounds: int = typer.Option(
        500, "--rounds", help="Maximum LightGBM boosting rounds."
    ),
) -> None:
    """Fit a LightGBM LambdaRank model on resulted races and save to disk."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    from ganyan.db import get_session
    from ganyan.predictor.ml import train_ranker

    start = datetime.strptime(from_date, "%Y-%m-%d").date() if from_date else None
    end = datetime.strptime(to_date, "%Y-%m-%d").date() if to_date else None

    session = get_session()
    try:
        result = train_ranker(
            session,
            from_date=start,
            to_date=end,
            holdout_fraction=holdout,
            num_boost_round=rounds,
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
