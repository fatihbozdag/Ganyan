"""APScheduler-based job runner for Ganyan.

Exposes :func:`build_scheduler` which returns a configured
:class:`BackgroundScheduler` with the four jobs the system needs to run
itself without human intervention:

1. **Morning card pull** — every day 08:30 Europe/Istanbul: scrape
   today's program, then pre-predict every race.
2. **Results poller** — every 20 minutes during race hours
   (13:45-23:30): pull today's results so the DB stays current
   for the web dashboard.
3. **Weekly pedigree refresh** — Sunday 03:00: crawl horses that
   picked up a ``tjk_at_id`` in the past week but still lack pedigree.
4. **Monthly model retrain** — first of the month 03:30: run
   ``train_ranker`` on the rolling 90-day window for both the main and
   value models.

The scheduler runs in the same process as the Flask app by default
(:class:`BackgroundScheduler`) — one process, one lifecycle.  Can be
run standalone via ``ganyan daemon``.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from ganyan.config import Settings


logger = logging.getLogger(__name__)

# Turkish racing is local to Europe/Istanbul; anchor crons there so
# "08:30 morning" means 08:30 TJK time regardless of host timezone.
_TZ = ZoneInfo("Europe/Istanbul")


# ---------------------------------------------------------------------------
# Job implementations
# ---------------------------------------------------------------------------


def _job_morning_card(settings: Settings) -> None:
    """Scrape today's program + predict every race.

    Runs once in the morning so the web UI / picker CLI has fresh
    data before the first post.
    """
    from ganyan.db import get_session
    from ganyan.db.models import Race, RaceEntry
    from ganyan.scraper import TJKClient, parse_race_card
    from ganyan.scraper.backfill import log_scrape, store_race_card
    from ganyan.db.models import ScrapeStatus
    from ganyan.predictor.ml import MLPredictor
    from sqlalchemy import func

    today = date.today()
    logger.info("scheduler: morning-card starting for %s", today)

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

    try:
        count = asyncio.run(_scrape())
    except Exception:  # noqa: BLE001
        logger.exception("scheduler: morning-card scrape failed")
        return

    # Predict all of today's races that have enough entries.
    session = get_session()
    try:
        predictor = MLPredictor(session)
        races = (
            session.query(Race).join(RaceEntry)
            .filter(Race.date == today)
            .group_by(Race.id)
            .having(func.count(RaceEntry.id) >= 3)
            .all()
        )
        for race in races:
            try:
                predictor.predict_and_save(race.id)
                session.commit()
            except Exception:  # noqa: BLE001
                session.rollback()
    finally:
        session.close()

    logger.info(
        "scheduler: morning-card done (%d races scraped, predictions refreshed)",
        count,
    )


def _job_results_poll(settings: Settings) -> None:
    """Pull today's results — keeps the DB current throughout the day."""
    from ganyan.db import get_session
    from ganyan.scraper import TJKClient, parse_race_card
    from ganyan.scraper.backfill import update_race_results

    today = date.today()
    logger.info("scheduler: results-poll starting for %s", today)

    async def _scrape() -> int:
        session = get_session()
        updated = 0
        try:
            async with TJKClient(
                base_url=settings.tjk_base_url, delay=settings.scrape_delay,
            ) as client:
                raw_cards = await client.get_race_results(today)
                for raw in raw_cards:
                    parsed = parse_race_card(raw)
                    race = update_race_results(session, parsed)
                    if race is not None:
                        updated += 1
                session.commit()
        finally:
            session.close()
        return updated

    try:
        n = asyncio.run(_scrape())
    except Exception:  # noqa: BLE001
        logger.exception("scheduler: results-poll failed")
        return
    logger.info("scheduler: results-poll done (%d races updated)", n)


def _job_pedigree_refresh(settings: Settings) -> None:
    """Fetch pedigree for horses that gained a tjk_at_id this week."""
    from ganyan.db import get_session
    from ganyan.scraper.horse_crawler import HorseCrawler

    logger.info("scheduler: pedigree-refresh starting")

    async def _run() -> int:
        session = get_session()
        try:
            async with HorseCrawler(
                session,
                base_url=settings.tjk_base_url,
                delay=0.3, concurrency=5,
            ) as crawler:
                return await crawler.crawl_missing_profiles()
        finally:
            session.close()

    try:
        n = asyncio.run(_run())
    except Exception:  # noqa: BLE001
        logger.exception("scheduler: pedigree-refresh failed")
        return
    logger.info("scheduler: pedigree-refresh done (%d horses updated)", n)


def _job_monthly_retrain(settings: Settings) -> None:
    """Retrain main + value models on rolling 90-day window."""
    from ganyan.db import get_session
    from ganyan.predictor.ml import train_ranker

    start = date.today() - timedelta(days=90)
    logger.info("scheduler: monthly-retrain starting (window from %s)", start)

    session = get_session()
    try:
        # Main (AGF-aware)
        try:
            train_ranker(
                session, from_date=start, model_name="lightgbm_ranker",
            )
        except Exception:  # noqa: BLE001
            logger.exception("scheduler: main retrain failed")
        # Value (no AGF)
        try:
            train_ranker(
                session, from_date=start,
                exclude_features=["agf_edge", "agf_raw"],
                model_name="lightgbm_value",
            )
        except Exception:  # noqa: BLE001
            logger.exception("scheduler: value retrain failed")
    finally:
        session.close()

    logger.info("scheduler: monthly-retrain done")


# ---------------------------------------------------------------------------
# Scheduler assembly
# ---------------------------------------------------------------------------


def _add_jobs(scheduler, settings: Settings) -> None:
    """Register the four jobs with the given scheduler."""
    scheduler.add_job(
        _job_morning_card,
        CronTrigger(hour=8, minute=30, timezone=_TZ),
        args=[settings],
        id="morning_card",
        name="Morning card scrape + predict",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=3600,
    )
    scheduler.add_job(
        _job_results_poll,
        # Every 20 minutes between 13:45 and 23:30 Turkish time.
        CronTrigger(
            minute="*/20",
            hour="13-23",
            timezone=_TZ,
        ),
        args=[settings],
        id="results_poll",
        name="Results polling",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=600,
    )
    scheduler.add_job(
        _job_pedigree_refresh,
        CronTrigger(day_of_week="sun", hour=3, minute=0, timezone=_TZ),
        args=[settings],
        id="pedigree_refresh",
        name="Weekly pedigree refresh",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        _job_monthly_retrain,
        CronTrigger(day=1, hour=3, minute=30, timezone=_TZ),
        args=[settings],
        id="monthly_retrain",
        name="Monthly model retrain",
        replace_existing=True,
        max_instances=1,
    )


def build_scheduler(
    settings: Settings, *, blocking: bool = False,
):
    """Build either a background or blocking scheduler pre-loaded with jobs."""
    scheduler = BlockingScheduler() if blocking else BackgroundScheduler()
    _add_jobs(scheduler, settings)
    return scheduler
