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
races_app = typer.Typer(help="View race information")
db_app = typer.Typer(help="Database management")

app.add_typer(scrape_app, name="scrape")
app.add_typer(predict_app, name="predict")
app.add_typer(races_app, name="races")
app.add_typer(db_app, name="db")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# scrape
# ---------------------------------------------------------------------------


@scrape_app.callback(invoke_without_command=True)
def scrape(
    today: bool = typer.Option(False, "--today", help="Fetch today's race cards"),
    results: bool = typer.Option(False, "--results", help="Fetch today's results"),
    backfill: bool = typer.Option(False, "--backfill", help="Run backfill from a date"),
    from_date: str = typer.Option(
        None, "--from", help="Start date for backfill (YYYY-MM-DD)"
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
    else:
        typer.echo("Use --today, --results, or --backfill. See --help.")


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


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@predict_app.callback(invoke_without_command=True)
def predict(
    race_id: int = typer.Argument(None, help="Race ID to predict"),
    today: bool = typer.Option(False, "--today", help="Predict all today's races"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format"),
) -> None:
    """Generate race predictions."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    if race_id is not None:
        _predict_race(race_id, json_output)
    elif today:
        _predict_today(json_output)
    else:
        typer.echo("Provide a race_id or use --today. See --help.")


def _predict_race(race_id: int, json_output: bool) -> None:
    """Predict a single race."""
    from ganyan.db import get_session
    from ganyan.predictor import BayesianPredictor

    session = get_session()
    try:
        predictor = BayesianPredictor(session)
        predictions = predictor.predict(race_id)
        if not predictions:
            typer.echo(f"No predictions for race {race_id}.")
            return
        _display_predictions(predictions, race_id, json_output)
    finally:
        session.close()


def _predict_today(json_output: bool) -> None:
    """Predict all races scheduled for today."""
    from ganyan.db import get_session, Race, RaceStatus

    session = get_session()
    try:
        from ganyan.predictor import BayesianPredictor

        races = (
            session.query(Race)
            .filter(Race.date == date.today(), Race.status == RaceStatus.scheduled)
            .all()
        )
        if not races:
            typer.echo("No races found for today.")
            return

        predictor = BayesianPredictor(session)
        for race in races:
            predictions = predictor.predict(race.id)
            _display_predictions(predictions, race.id, json_output)
            typer.echo("")  # blank line separator
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
