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
    from ganyan.predictor.ml.ensemble import EnsemblePredictor
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

        predictor = EnsemblePredictor(session)
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
                # refresh=True so intraday re-runs rewrite ungraded picks
                # instead of silently no-op'ing on the morning snapshot.
                picks = generate_picks_for_race(
                    session, race.id, refresh=True,
                )
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

    # Multi-race 6'lı paper-trade coupon — per project_pivot_steps_2026_05_11.
    # All single-race structures are negative-EV; multi-race exotics are
    # the only remaining path with mathematical EV. Daily paper trade
    # records what we WOULD have bet to validate Step 2's EV math.
    # 6'lı pool = last 6 races by TJK convention (16/16 days verified
    # 2026-05-11 across all tracks). Max 144 tickets per coupon = 144 TL
    # nominal stake.
    session = get_session()
    multi_coupons = 0
    try:
        from sqlalchemy import func
        from ganyan.predictor.multi_race_picks import (
            generate_coupon, persist_coupon,
        )
        from ganyan.db.models import Track

        track_max_race = (
            session.query(Track.name, func.max(Race.race_number))
            .join(Race, Race.track_id == Track.id)
            .filter(Race.date == today)
            .group_by(Track.name)
            .all()
        )
        for track_name, max_race in track_max_race:
            if max_race is None or max_race < 6:
                continue
            start_race = max_race - 5  # 6 legs ending at max_race
            try:
                draft = generate_coupon(
                    session, today, track_name, start_race,
                    pool_type="6li", max_tickets=144,
                )
                persist_coupon(
                    session, today, track_name, start_race, draft,
                    pool_type="6li",
                )
                session.commit()
                multi_coupons += 1
            except Exception:  # noqa: BLE001
                logger.exception(
                    "scheduler: multi-race coupon generation failed for "
                    "%s R%d-R%d", track_name, start_race, max_race,
                )
                session.rollback()
    finally:
        session.close()
    logger.info("scheduler: morning-card generated %d multi-race coupon(s)", multi_coupons)


def _job_repredict_upcoming(settings: Settings) -> None:
    """Re-predict every still-upcoming race today using current AGF.

    Background: morning_card runs once at 08:30, before AGF is published.
    LGBM with NULL agf/last_six features falls through to uniform 1/N
    softmax, so every horse gets the same probability and the picks are
    just gate-ordered fallbacks. AGF is first scraped at 11:00 (the
    agf_snapshot job), so any race that posts after AGF settles needs
    its prediction redone before its post time.

    Trigger: every 30 minutes between 11:05 and 20:35 — five minutes
    after each agf_snapshot run so AGF has committed. Skips races that
    have already resulted *and* races whose post_time has passed (TJK's
    results-publish often lags the race itself by 30+ minutes; refreshing
    a pick after the gates open silently rewrites the operator's
    morning-recorded recommendation, which is the audit-of-record).
    """
    from datetime import datetime, timedelta

    from sqlalchemy import func

    from ganyan.db import get_session
    from ganyan.db.models import Race, RaceEntry, RaceStatus
    from ganyan.predictor.ml.ensemble import EnsemblePredictor
    from ganyan.predictor.picks import generate_picks_for_race

    today = date.today()
    now_local = datetime.now()
    # Cut-off: don't re-predict races whose post_time is within the next
    # 5 minutes or already past.  Operator may have placed bets based on
    # the previous prediction; gates close at post_time.
    cutoff = (now_local + timedelta(minutes=5)).strftime("%H:%M")
    logger.info(
        "scheduler: repredict-upcoming starting for %s (cutoff post_time>%s)",
        today, cutoff,
    )

    session = get_session()
    n_repredicted = n_picks = n_skipped_late = 0
    try:
        predictor = EnsemblePredictor(session)
        races = (
            session.query(Race).join(RaceEntry)
            .filter(
                Race.date == today,
                Race.status != RaceStatus.resulted,
            )
            .group_by(Race.id)
            .having(func.count(RaceEntry.id) >= 3)
            .all()
        )
        races_kept = []
        for race in races:
            if race.post_time and race.post_time <= cutoff:
                n_skipped_late += 1
                continue
            races_kept.append(race)
        races = races_kept
        for race in races:
            try:
                predictor.predict_and_save(race.id)
                picks = generate_picks_for_race(
                    session, race.id, refresh=True,
                )
                n_repredicted += 1
                n_picks += len(picks)
                session.commit()
            except Exception:  # noqa: BLE001
                session.rollback()
                logger.exception(
                    "scheduler: repredict failed for race %s", race.id,
                )
    finally:
        session.close()

    logger.info(
        "scheduler: repredict-upcoming done "
        "(%d races re-predicted, %d picks rewritten, %d skipped post-cutoff)",
        n_repredicted, n_picks, n_skipped_late,
    )


def _job_agf_snapshot(settings: Settings) -> None:
    """Re-scrape today's program for the sole purpose of capturing the
    current AGF reading per entry.  Each call writes a row into
    ``agf_snapshots`` (via the existing scraper code path), so over a
    day the table accumulates a time-series of how the crowd's belief
    moves.  Late-money drift = late_snapshot − early_snapshot becomes
    a feature once enough samples accumulate.
    """
    from ganyan.db import get_session
    from ganyan.db.models import ScrapeStatus
    from ganyan.scraper import TJKClient, parse_race_card
    from ganyan.scraper.backfill import log_scrape, store_race_card

    today = date.today()
    logger.info("scheduler: agf-snapshot starting for %s", today)

    async def _scrape() -> int:
        session = get_session()
        n = 0
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
                    n += 1
                session.commit()
        finally:
            session.close()
        return n

    try:
        cards = asyncio.run(_scrape())
    except Exception:  # noqa: BLE001
        logger.exception("scheduler: agf-snapshot failed")
        return
    logger.info(
        "scheduler: agf-snapshot done (%d cards re-fetched)", cards,
    )


def _job_external_signals(settings: Settings) -> None:
    """Run every registered third-party signal scraper for today.

    Persists raw rows then runs the resolver to bind them to TJK
    race / race_entry IDs.  Idempotent — re-running just appends new
    rows; the resolver only updates rows whose race_entry_id is null.

    Intentional ordering: fires AFTER the morning_card job so the
    resolver has TJK's program available to bind against.
    """
    del settings  # plugins manage their own HTTP clients
    from ganyan.db import get_session
    from ganyan.scraper.external.resolver import fetch_and_resolve

    today = date.today()
    logger.info("scheduler: external-signals starting for %s", today)

    session = get_session()
    try:
        results = fetch_and_resolve(session, today)
        session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("scheduler: external-signals failed")
        session.rollback()
        return
    finally:
        session.close()

    counts = ", ".join(f"{k}={v}" for k, v in sorted(results.items())) or "no sources"
    logger.info("scheduler: external-signals done (%s)", counts)


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
        from ganyan.predictor.picks import grade_all_pending, resettle_plase_picks

        graded = grade_all_pending(session)
        # plase_top1 picks are graded the moment the finish positions
        # land, but their payout column may still be NULL until TJK
        # publishes "PLASE <program_no> <amount>" on the bahisSonucCard.
        # resettle_plase_picks fills in payout/net once the scraper
        # picks up that row on a later poll. Cheap when nothing's
        # missing.
        resettled = resettle_plase_picks(session)
        if resettled:
            logger.info("scheduler: resettled %d plase_top1 picks", resettled)
        session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("scheduler: pick grading failed")
        session.rollback()
        graded = 0
    finally:
        session.close()

    session = get_session()
    try:
        from ganyan.predictor.multi_race_picks import grade_all_pending_multi

        multi_graded = grade_all_pending_multi(session)
        session.commit()
    except Exception:  # noqa: BLE001
        logger.exception("scheduler: multi-race pick grading failed")
        session.rollback()
        multi_graded = 0
    finally:
        session.close()

    logger.info(
        "scheduler: results-poll done (%d races updated, %d picks graded, "
        "%d multi-race picks graded)",
        n, graded, multi_graded,
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
    """Register the scheduled jobs with the given scheduler."""
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
        _job_external_signals,
        # Twice per day: once in the morning after the program drops
        # so tipster picks for today's card are captured early, then
        # again pre-evening so any late-published picks are caught
        # before the last race.  Cheap (single HTTP fetch + parse) so
        # extra runs are harmless.
        CronTrigger(
            minute="15",
            hour="9,18",
            timezone=_TZ,
        ),
        args=[settings],
        id="external_signals",
        name="External-data plugins (tipsters, etc.)",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=600,
    )
    scheduler.add_job(
        _job_agf_snapshot,
        # Every 30 minutes between 11:00 and 22:30 Turkish time, so we
        # capture multiple readings per race day from morning AGF
        # publication through late-money flow at the last post.  Each
        # call writes a row to agf_snapshots per entry; over time this
        # builds the time-series the late-drift feature needs.
        CronTrigger(
            minute="0,30",
            hour="11-22",
            timezone=_TZ,
        ),
        args=[settings],
        id="agf_snapshot",
        name="AGF snapshot for late-drift signal",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        _job_repredict_upcoming,
        # 5 min after each agf_snapshot — gives AGF time to commit, then
        # re-predicts every still-upcoming race so post-08:30 picks no
        # longer ride a uniform 1/N fallback.
        CronTrigger(
            minute="5,35",
            hour="11-20",
            timezone=_TZ,
        ),
        args=[settings],
        id="repredict_upcoming",
        name="Re-predict still-upcoming races with current AGF",
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
    # DISARMED 2026-08-12 — CLAUDE.md invariant #7: this job bare-trained onto
    # the LIVE model files with no OOS gate and degraded top-1 42.94% → 40.92%
    # on 2026-06-01.  Re-enable only after it trains to a candidate name (NOT
    # in models/ root — the ensemble globs *.meta.json there) and passes
    # logs/discordance_oos_backtest.py.
    # scheduler.add_job(
    #     _job_monthly_retrain,
    #     CronTrigger(day=1, hour=3, minute=30, timezone=_TZ),
    #     args=[settings],
    #     id="monthly_retrain",
    #     name="Monthly model retrain",
    #     replace_existing=True,
    #     max_instances=1,
    # )


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
