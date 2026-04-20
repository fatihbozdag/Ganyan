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
import shutil
import subprocess
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
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

    # Predict all of today's races that have enough entries, then write
    # strategy-level Pick rows so we track real-world ROI over time.
    session = get_session()
    picks_created = 0
    try:
        from ganyan.predictor.picks import generate_picks_for_race

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
                picks = generate_picks_for_race(session, race.id)
                picks_created += len(picks)
                session.commit()
            except Exception:  # noqa: BLE001
                session.rollback()
    finally:
        session.close()

    logger.info(
        "scheduler: morning-card done (%d races scraped, %d picks written)",
        count, picks_created,
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

    # Grade any picks whose races just resulted.  Cheap no-op when
    # nothing new finished since the last poll.
    session = get_session()
    try:
        from ganyan.predictor.picks import grade_all_pending

        graded = grade_all_pending(session)
        session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("scheduler: pick grading failed")
        session.rollback()
        graded = 0
    finally:
        session.close()

    logger.info(
        "scheduler: results-poll done (%d races updated, %d picks graded)",
        n, graded,
    )


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
    """Build either a background or blocking scheduler pre-loaded with jobs
    plus a listener that persists every run to the ``job_runs`` table and
    pops a macOS notification on failure.
    """
    scheduler = BlockingScheduler() if blocking else BackgroundScheduler()
    _add_jobs(scheduler, settings)
    _attach_run_listener(scheduler)
    return scheduler


# ---------------------------------------------------------------------------
# Run persistence + notifications
# ---------------------------------------------------------------------------


def _attach_run_listener(scheduler) -> None:
    """Record every job execution and alert on failures."""
    scheduler.add_listener(
        _on_job_event,
        EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED,
    )


def _on_job_event(event) -> None:
    """APScheduler event handler.

    ``event.code`` is one of the EVENT_JOB_* constants.  On ERROR or
    MISSED events we emit a macOS notification so the user knows to
    look at the logs without needing to poll the dashboard.
    """
    from ganyan.db import get_session
    from ganyan.db.models import JobRun, JobStatus

    session = get_session()
    try:
        status: str
        error_message: str | None = None
        duration_ms: int | None = None

        if event.code == EVENT_JOB_EXECUTED:
            status = JobStatus.success.value
        elif event.code == EVENT_JOB_ERROR:
            status = JobStatus.failed.value
            error_message = (
                f"{event.exception.__class__.__name__}: {event.exception}"
            )[:2000]
        elif event.code == EVENT_JOB_MISSED:
            status = JobStatus.missed.value
            error_message = "scheduler missed run window"
        else:
            return

        # APScheduler's event.scheduled_run_time is tz-aware; strip tzinfo
        # to match our DB column (DateTime without timezone).
        started_at = event.scheduled_run_time.replace(tzinfo=None)
        finished_at = datetime.now()
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        run = JobRun(
            job_id=event.job_id,
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            duration_ms=duration_ms,
            error_message=error_message,
        )
        session.add(run)
        session.commit()
    except Exception:  # noqa: BLE001 — listener must never crash scheduler
        logger.exception("job-run persistence failed for job %s", event.job_id)
        session.rollback()
    finally:
        session.close()

    if event.code in (EVENT_JOB_ERROR, EVENT_JOB_MISSED):
        _notify_failure(event)


def _notify_failure(event) -> None:
    """Pop a native macOS notification for a failed/missed job.

    Silently no-ops on non-Darwin hosts or if ``osascript`` isn't on
    PATH, so the same code runs fine under Linux deployments too.
    """
    if shutil.which("osascript") is None:
        return
    title = "Ganyan"
    code_label = (
        "failed" if event.code == EVENT_JOB_ERROR else "missed"
    )
    message = f"Job {event.job_id} {code_label}."
    if event.code == EVENT_JOB_ERROR and event.exception is not None:
        message += f" {event.exception.__class__.__name__}: {event.exception}"
    # AppleScript is finicky with quotes; simple replace is enough.
    message = message.replace('"', "'")[:200]
    try:
        subprocess.run(
            [
                "osascript", "-e",
                f'display notification "{message}" with title "{title}"',
            ],
            check=False, timeout=5,
        )
    except Exception:  # noqa: BLE001
        logger.exception("macOS notification failed for job %s", event.job_id)
