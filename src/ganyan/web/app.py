"""Flask application factory for the Ganyan web interface."""

from __future__ import annotations

import logging
import os
import threading
from datetime import date, timedelta

from flask import Flask
from sqlalchemy.orm import sessionmaker

from ganyan.config import Settings, get_settings
from ganyan.db.session import get_session_factory


logger = logging.getLogger(__name__)


# Module-level flag so repeated ``create_app`` calls (e.g. tests,
# WSGI servers that pre-import the factory) don't each spawn a refresh
# thread.
_LAUNCH_REFRESH_STARTED = False


def create_app(
    session_factory: sessionmaker | None = None,
    *,
    refresh_on_launch: bool | None = None,
    refresh_lookback_days: int = 14,
    enable_scheduler: bool | None = None,
) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    session_factory:
        Optional SQLAlchemy session factory.  When ``None`` (the default),
        the factory is built from :func:`ganyan.db.session.get_session_factory`.
    refresh_on_launch:
        If ``True`` (or ``None`` and env ``GANYAN_SKIP_LAUNCH_REFRESH`` is
        unset), spawn a background thread that refreshes the last
        ``refresh_lookback_days`` of historical race results via the
        full-field results endpoint.  Non-blocking: the app starts
        serving immediately.
    refresh_lookback_days:
        How many days back to check for missing historical data.
    """
    app = Flask(__name__)

    settings = get_settings()
    app.config["SECRET_KEY"] = "dev"

    if session_factory is None:
        session_factory = get_session_factory()
    app.config["SESSION_FACTORY"] = session_factory

    @app.context_processor
    def inject_today():
        return {"today": date.today().isoformat()}

    from ganyan.web.routes import bp
    app.register_blueprint(bp)

    if refresh_on_launch is None:
        refresh_on_launch = (
            os.environ.get("GANYAN_SKIP_LAUNCH_REFRESH", "").lower()
            not in {"1", "true", "yes"}
        )
    if refresh_on_launch:
        _start_launch_refresh(settings, refresh_lookback_days)

    if enable_scheduler is None:
        enable_scheduler = (
            os.environ.get("GANYAN_SKIP_SCHEDULER", "").lower()
            not in {"1", "true", "yes"}
        )
    if enable_scheduler:
        _start_scheduler(settings)

    return app


# Separate guard so scheduler and refresh don't share the same flag.
_SCHEDULER_STARTED = False


def _start_scheduler(settings: Settings) -> None:
    """Spin up the APScheduler background jobs once per process."""
    global _SCHEDULER_STARTED
    if _SCHEDULER_STARTED:
        return
    _SCHEDULER_STARTED = True
    try:
        from ganyan.scheduler import build_scheduler

        scheduler = build_scheduler(settings, blocking=False)
        scheduler.start()
        logger.info(
            "APScheduler started with jobs: %s",
            [j.id for j in scheduler.get_jobs()],
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to start APScheduler")


def _start_launch_refresh(settings: Settings, lookback_days: int) -> None:
    """Kick off a one-shot background thread to update historical data."""
    global _LAUNCH_REFRESH_STARTED
    if _LAUNCH_REFRESH_STARTED:
        return
    _LAUNCH_REFRESH_STARTED = True

    def _worker() -> None:
        import asyncio

        from ganyan.db import get_session
        from ganyan.scraper import TJKClient
        from ganyan.scraper.backfill import BackfillManager

        today = date.today()
        start = today - timedelta(days=lookback_days)

        async def _refresh() -> None:
            session = get_session()
            try:
                async with TJKClient(
                    base_url=settings.tjk_base_url,
                    delay=settings.scrape_delay,
                ) as client:
                    manager = BackfillManager(session, client)
                    stored = await manager.backfill_full_results(
                        from_date=start, to_date=today,
                    )
                    logger.info(
                        "Launch refresh: %d race(s) stored (%s -> %s)",
                        stored, start, today,
                    )
            finally:
                session.close()

        try:
            logger.info(
                "Launch refresh: updating historical results for last %d days",
                lookback_days,
            )
            asyncio.run(_refresh())
        except Exception:  # noqa: BLE001 — log and continue; app must keep serving
            logger.exception("Launch-time historical refresh failed")

    thread = threading.Thread(
        target=_worker, name="ganyan-launch-refresh", daemon=True,
    )
    thread.start()


def run() -> None:
    """Run the Flask development server."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    app = create_app()
    app.run(host="0.0.0.0", port=settings.flask_port, debug=settings.flask_debug)
