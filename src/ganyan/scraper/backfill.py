"""Scraper-to-database integration and historical backfill manager."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from sqlalchemy.orm import Session

from ganyan.db.models import (
    Horse,
    Race,
    RaceEntry,
    RaceStatus,
    ScrapeLog,
    ScrapeStatus,
    Track,
)
from ganyan.scraper.parser import ParsedRaceCard

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: get-or-create
# ---------------------------------------------------------------------------


def get_or_create_track(session: Session, name: str) -> Track:
    """Return an existing Track or create a new one."""
    track = session.query(Track).filter(Track.name == name).first()
    if track is not None:
        return track
    track = Track(name=name)
    session.add(track)
    session.flush()
    return track


def get_or_create_horse(session: Session, name: str, **kwargs) -> Horse:
    """Return an existing Horse or create a new one.

    Mutable fields (age, owner, trainer, origin) are updated when provided
    on an existing record so the database always reflects the latest data.
    """
    horse = session.query(Horse).filter(Horse.name == name).first()
    if horse is not None:
        # Update mutable fields if a new value was provided
        for field in ("age", "origin", "owner", "trainer"):
            value = kwargs.get(field)
            if value is not None:
                setattr(horse, field, value)
        return horse
    horse = Horse(name=name, **kwargs)
    session.add(horse)
    session.flush()
    return horse


# ---------------------------------------------------------------------------
# Store / update
# ---------------------------------------------------------------------------


def store_race_card(session: Session, parsed: ParsedRaceCard) -> Race:
    """Persist a ParsedRaceCard to the database.

    Creates or reuses Track and Horse records.  Creates Race and RaceEntry
    records.  The operation is idempotent -- calling it twice with the same
    data is safe and will not duplicate records.
    """
    track = get_or_create_track(session, parsed.track_name)

    # Check for an existing race (idempotency)
    race = (
        session.query(Race)
        .filter(
            Race.track_id == track.id,
            Race.date == parsed.date,
            Race.race_number == parsed.race_number,
        )
        .first()
    )
    if race is None:
        race = Race(
            track_id=track.id,
            date=parsed.date,
            race_number=parsed.race_number,
            distance_meters=parsed.distance_meters,
            surface=parsed.surface,
            race_type=parsed.race_type,
            horse_type=parsed.horse_type,
            weight_rule=parsed.weight_rule,
            status=RaceStatus.scheduled,
        )
        session.add(race)
        session.flush()

    for h in parsed.horses:
        horse = get_or_create_horse(
            session,
            h.name,
            age=h.age,
            origin=h.origin,
            owner=h.owner,
            trainer=h.trainer,
        )

        # Check for existing entry (idempotency)
        existing_entry = (
            session.query(RaceEntry)
            .filter(
                RaceEntry.race_id == race.id,
                RaceEntry.horse_id == horse.id,
            )
            .first()
        )
        if existing_entry is not None:
            continue

        entry = RaceEntry(
            race_id=race.id,
            horse_id=horse.id,
            gate_number=h.gate_number,
            jockey=h.jockey,
            weight_kg=h.weight_kg,
            hp=h.hp,
            kgs=h.kgs,
            s20=h.s20,
            eid=h.eid,
            gny=h.gny,
            agf=h.agf,
            last_six=h.last_six,
        )
        session.add(entry)

    session.flush()
    return race


def update_race_results(session: Session, parsed: ParsedRaceCard) -> Race | None:
    """Update existing race entries with finish positions and times.

    Returns the Race if found, or None if the race does not exist yet.
    """
    track = session.query(Track).filter(Track.name == parsed.track_name).first()
    if track is None:
        return None

    race = (
        session.query(Race)
        .filter(
            Race.track_id == track.id,
            Race.date == parsed.date,
            Race.race_number == parsed.race_number,
        )
        .first()
    )
    if race is None:
        return None

    for h in parsed.horses:
        horse = session.query(Horse).filter(Horse.name == h.name).first()
        if horse is None:
            continue

        entry = (
            session.query(RaceEntry)
            .filter(
                RaceEntry.race_id == race.id,
                RaceEntry.horse_id == horse.id,
            )
            .first()
        )
        if entry is None:
            continue

        entry.finish_position = h.finish_position
        entry.finish_time = h.finish_time

    race.status = RaceStatus.resulted
    session.flush()
    return race


# ---------------------------------------------------------------------------
# Scrape log helpers
# ---------------------------------------------------------------------------


def get_scraped_dates(session: Session) -> set[date]:
    """Return the set of dates that have been successfully scraped."""
    rows = (
        session.query(ScrapeLog.date)
        .filter(ScrapeLog.status == ScrapeStatus.success)
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def log_scrape(
    session: Session,
    scrape_date: date,
    track: str,
    status: ScrapeStatus,
) -> None:
    """Record a scrape attempt in the scrape_log table."""
    entry = ScrapeLog(date=scrape_date, track=track, status=status)
    session.add(entry)
    session.flush()


# ---------------------------------------------------------------------------
# BackfillManager
# ---------------------------------------------------------------------------


class BackfillManager:
    """Manages incremental historical data loading from TJK.

    Processes dates in reverse chronological order and skips dates that
    have already been successfully scraped.
    """

    def __init__(self, session: Session, tjk_client) -> None:
        self.session = session
        self.tjk_client = tjk_client

    async def backfill(
        self,
        from_date: date,
        to_date: date | None = None,
    ) -> None:
        """Scrape race cards for a date range, newest first.

        Parameters
        ----------
        from_date:
            Earliest date to scrape (inclusive).
        to_date:
            Latest date to scrape (inclusive). Defaults to today.
        """
        if to_date is None:
            to_date = date.today()

        already_scraped = get_scraped_dates(self.session)

        # Build list of dates in reverse chronological order
        current = to_date
        while current >= from_date:
            if current not in already_scraped:
                await self._scrape_date(current)
            else:
                logger.debug("Skipping already-scraped date %s", current)
            current -= timedelta(days=1)

    async def _scrape_date(self, scrape_date: date) -> None:
        """Fetch and store all race cards for a single date."""
        logger.info("Scraping %s", scrape_date)
        try:
            raw_cards = await self.tjk_client.get_race_card(scrape_date)
        except Exception:
            logger.exception("Failed to fetch race card for %s", scrape_date)
            log_scrape(self.session, scrape_date, "ALL", ScrapeStatus.failed)
            return

        if not raw_cards:
            log_scrape(self.session, scrape_date, "ALL", ScrapeStatus.skipped)
            return

        from ganyan.scraper.parser import parse_race_card

        for raw in raw_cards:
            parsed = parse_race_card(raw)
            store_race_card(self.session, parsed)
            log_scrape(
                self.session,
                scrape_date,
                parsed.track_name,
                ScrapeStatus.success,
            )

        self.session.commit()
