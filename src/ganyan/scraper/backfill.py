"""Scraper-to-database integration and historical backfill manager."""

from __future__ import annotations

import asyncio
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


_ENTRY_REFRESH_FIELDS = (
    "gate_number", "jockey", "weight_kg", "hp", "kgs",
    "s20", "eid", "gny", "agf", "last_six",
)


def _refresh_entry_fields(existing: RaceEntry, h) -> None:
    """Copy non-None pre-race fields from parsed horse onto an existing entry.

    Also updates finish_position/finish_time when the parsed horse carries
    result data so callers that pass result-enriched cards through this path
    (rather than ``update_race_results``) don't silently lose the results.
    """
    for field in _ENTRY_REFRESH_FIELDS:
        value = getattr(h, field, None)
        if value is not None:
            setattr(existing, field, value)
    if getattr(h, "finish_position", None) is not None:
        existing.finish_position = h.finish_position
    if getattr(h, "finish_time", None) is not None:
        existing.finish_time = h.finish_time


def _fetch_horses_by_names(session: Session, names: list[str]) -> dict[str, Horse]:
    """Batch-load horses by name in a single query (avoids N+1)."""
    if not names:
        return {}
    rows = session.query(Horse).filter(Horse.name.in_(names)).all()
    return {h.name: h for h in rows}


def store_race_card(session: Session, parsed: ParsedRaceCard) -> Race:
    """Persist a ParsedRaceCard to the database.

    Creates or reuses Track and Horse records.  Creates Race and RaceEntry
    records.  The operation is idempotent -- calling it twice with the same
    data is safe.  On conflict, pre-race fields (jockey, weight, HP, etc.)
    and any finish data present on the parsed card are refreshed so the
    database always reflects the latest scrape.
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
            post_time=parsed.post_time,
            distance_meters=parsed.distance_meters,
            surface=parsed.surface,
            race_type=parsed.race_type,
            horse_type=parsed.horse_type,
            weight_rule=parsed.weight_rule,
            pace_l800_leader_s=parsed.pace_l800_leader_s,
            pace_l800_runner_up_s=parsed.pace_l800_runner_up_s,
            status=RaceStatus.scheduled,
        )
        session.add(race)
        session.flush()
    else:
        # Backfill race-level fields that weren't available on a prior scrape
        # (e.g. older scrape that didn't capture post_time).
        if parsed.post_time is not None and not race.post_time:
            race.post_time = parsed.post_time

    # Batch-load all existing entries for this race to avoid per-horse queries.
    existing_entries = {
        (e.race_id, e.horse_id): e
        for e in session.query(RaceEntry).filter(RaceEntry.race_id == race.id).all()
    }
    horse_cache = _fetch_horses_by_names(
        session, [h.name for h in parsed.horses if h.name],
    )

    for h in parsed.horses:
        horse = horse_cache.get(h.name)
        if horse is None:
            horse = get_or_create_horse(
                session,
                h.name,
                age=h.age,
                origin=h.origin,
                owner=h.owner,
                trainer=h.trainer,
            )
            horse_cache[h.name] = horse
        else:
            # Update mutable horse-level fields inline (same rule as get_or_create_horse)
            for field in ("age", "origin", "owner", "trainer"):
                value = getattr(h, field, None)
                if value is not None:
                    setattr(horse, field, value)

        existing = existing_entries.get((race.id, horse.id))
        if existing is not None:
            _refresh_entry_fields(existing, h)
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
            finish_position=h.finish_position,
            finish_time=h.finish_time,
        )
        session.add(entry)

    session.flush()
    return race


def store_historical_race(session: Session, parsed: ParsedRaceCard) -> Race:
    """Persist a historical race result to the database.

    Similar to ``store_race_card`` but immediately marks the race as
    ``resulted`` since historical query data represents completed races.
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
            post_time=parsed.post_time,
            distance_meters=parsed.distance_meters,
            surface=parsed.surface,
            race_type=parsed.race_type,
            horse_type=parsed.horse_type,
            weight_rule=parsed.weight_rule,
            pace_l800_leader_s=parsed.pace_l800_leader_s,
            pace_l800_runner_up_s=parsed.pace_l800_runner_up_s,
            status=RaceStatus.resulted,
        )
        session.add(race)
        session.flush()
    else:
        # Upgrade status if the race already existed as scheduled
        if race.status != RaceStatus.resulted:
            race.status = RaceStatus.resulted
        # Backfill race-level fields a prior scrape missed.
        if parsed.post_time is not None and not race.post_time:
            race.post_time = parsed.post_time
        if parsed.pace_l800_leader_s is not None and race.pace_l800_leader_s is None:
            race.pace_l800_leader_s = parsed.pace_l800_leader_s
        if parsed.pace_l800_runner_up_s is not None and race.pace_l800_runner_up_s is None:
            race.pace_l800_runner_up_s = parsed.pace_l800_runner_up_s

    existing_entries = {
        (e.race_id, e.horse_id): e
        for e in session.query(RaceEntry).filter(RaceEntry.race_id == race.id).all()
    }
    horse_cache = _fetch_horses_by_names(
        session, [h.name for h in parsed.horses if h.name],
    )

    for h in parsed.horses:
        horse = horse_cache.get(h.name)
        if horse is None:
            horse = get_or_create_horse(
                session,
                h.name,
                age=h.age,
                origin=h.origin,
                owner=h.owner,
                trainer=h.trainer,
            )
            horse_cache[h.name] = horse
        else:
            for field in ("age", "origin", "owner", "trainer"):
                value = getattr(h, field, None)
                if value is not None:
                    setattr(horse, field, value)

        existing = existing_entries.get((race.id, horse.id))
        if existing is not None:
            _refresh_entry_fields(existing, h)
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
            finish_position=h.finish_position,
            finish_time=h.finish_time,
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

    # Capture race-level results-page fields the program scrape couldn't
    # possibly have had (Son 800 only appears on the results endpoint).
    if parsed.pace_l800_leader_s is not None:
        race.pace_l800_leader_s = parsed.pace_l800_leader_s
    if parsed.pace_l800_runner_up_s is not None:
        race.pace_l800_runner_up_s = parsed.pace_l800_runner_up_s

    horse_cache = _fetch_horses_by_names(
        session, [h.name for h in parsed.horses if h.name],
    )
    entries_by_horse = {
        e.horse_id: e
        for e in session.query(RaceEntry).filter(RaceEntry.race_id == race.id).all()
    }

    for h in parsed.horses:
        horse = horse_cache.get(h.name)
        if horse is None:
            continue
        entry = entries_by_horse.get(horse.id)
        if entry is None:
            continue
        if h.finish_position is not None:
            entry.finish_position = h.finish_position
        if h.finish_time is not None:
            entry.finish_time = h.finish_time

    race.status = RaceStatus.resulted
    session.flush()
    return race


# ---------------------------------------------------------------------------
# Scrape log helpers
# ---------------------------------------------------------------------------


_ALL_TRACKS_SENTINEL = "ALL"


def get_scraped_dates(session: Session) -> set[date]:
    """Return the set of dates that have been *fully* scraped.

    A date counts as fully scraped only when a completion marker was
    written (track=ALL with status=success).  Partial-success days where
    some tracks failed are deliberately NOT in this set so the backfill
    manager retries them.
    """
    rows = (
        session.query(ScrapeLog.date)
        .filter(
            ScrapeLog.status == ScrapeStatus.success,
            ScrapeLog.track == _ALL_TRACKS_SENTINEL,
        )
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def get_successful_tracks_on(session: Session, scrape_date: date) -> set[str]:
    """Return set of track names that succeeded on a given date."""
    rows = (
        session.query(ScrapeLog.track)
        .filter(
            ScrapeLog.date == scrape_date,
            ScrapeLog.status == ScrapeStatus.success,
        )
        .distinct()
        .all()
    )
    return {row[0] for row in rows if row[0] != _ALL_TRACKS_SENTINEL}


def log_scrape(
    session: Session,
    scrape_date: date,
    track: str,
    status: ScrapeStatus,
    error_message: str | None = None,
) -> None:
    """Record a scrape attempt in the scrape_log table.

    ``error_message`` is stored when status is ``failed``; truncated to
    keep audit rows small.
    """
    msg = None
    if error_message is not None:
        msg = error_message[:2000]  # avoid unbounded blobs
    entry = ScrapeLog(
        date=scrape_date, track=track, status=status, error_message=msg,
    )
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
        """Fetch and store all race cards for a single date.

        Writes per-track success rows and, only if every discovered track
        succeeded, a single ``track=ALL`` completion marker so that
        :func:`get_scraped_dates` can distinguish fully- from partially-
        scraped days and retry the partials.
        """
        logger.info("Scraping %s", scrape_date)

        # Use the failure-aware variant when available so we can log the
        # specific tracks that did not return cards.
        getter = getattr(
            self.tjk_client, "get_race_card_with_failures", None,
        )
        try:
            if getter is not None:
                raw_cards, failed_tracks = await getter(scrape_date)
            else:
                raw_cards = await self.tjk_client.get_race_card(scrape_date)
                failed_tracks = []
        except Exception as exc:
            logger.exception("Failed to fetch race card for %s", scrape_date)
            log_scrape(
                self.session, scrape_date, _ALL_TRACKS_SENTINEL,
                ScrapeStatus.failed, error_message=str(exc),
            )
            self.session.commit()
            return

        if not raw_cards and not failed_tracks:
            log_scrape(
                self.session, scrape_date, _ALL_TRACKS_SENTINEL,
                ScrapeStatus.skipped,
            )
            self.session.commit()
            return

        from ganyan.scraper.parser import parse_race_card

        for raw in raw_cards:
            parsed = parse_race_card(raw)
            store_race_card(self.session, parsed)
            log_scrape(
                self.session, scrape_date, parsed.track_name,
                ScrapeStatus.success,
            )

        for track_name in failed_tracks:
            log_scrape(
                self.session, scrape_date, track_name,
                ScrapeStatus.failed,
                error_message="empty response or HTTP error",
            )

        # Only mark the whole date done when no tracks failed.
        if not failed_tracks:
            log_scrape(
                self.session, scrape_date, _ALL_TRACKS_SENTINEL,
                ScrapeStatus.success,
            )

        self.session.commit()

    async def backfill_full_results(
        self,
        from_date: date,
        to_date: date,
        *,
        rescrape: bool = False,
    ) -> int:
        """Backfill full historical race fields via the daily results page.

        Unlike :meth:`backfill_historical` (which uses the KosuSorgulama
        bulk query and captures only the winner of each race), this walks
        each date in range and calls the per-city GunlukYarisSonuclari
        endpoint, yielding every runner with AGF, gate, jockey, HP, and
        finish position.  That's the shape the predictor's features
        actually need.

        Parameters
        ----------
        from_date, to_date:
            Inclusive range, walked newest first.
        rescrape:
            If ``True``, re-scrape even dates already marked complete in
            :class:`ScrapeLog`.  If ``False`` (default), skips any date
            with an ``ALL`` success marker.

        Returns
        -------
        int
            Number of race records stored or refreshed.
        """
        if from_date > to_date:
            return 0

        from ganyan.scraper.parser import parse_race_card

        already_done = set() if rescrape else get_scraped_dates(self.session)
        total_stored = 0

        current = to_date
        while current >= from_date:
            if current in already_done:
                logger.debug("Skipping already-complete date %s", current)
                current -= timedelta(days=1)
                continue

            logger.info("Full-field results backfill: %s", current)
            getter = getattr(
                self.tjk_client, "get_race_results_with_failures", None,
            )
            try:
                if getter is not None:
                    raw_cards, failed_tracks = await getter(current)
                else:
                    raw_cards = await self.tjk_client.get_race_results(current)
                    failed_tracks = []
            except Exception as exc:
                logger.exception("Full-field results failed for %s", current)
                log_scrape(
                    self.session, current, _ALL_TRACKS_SENTINEL,
                    ScrapeStatus.failed, error_message=str(exc),
                )
                self.session.commit()
                current -= timedelta(days=1)
                continue

            if not raw_cards and not failed_tracks:
                log_scrape(
                    self.session, current, _ALL_TRACKS_SENTINEL,
                    ScrapeStatus.skipped,
                )
                self.session.commit()
                current -= timedelta(days=1)
                continue

            for raw in raw_cards:
                parsed = parse_race_card(raw)
                store_historical_race(self.session, parsed)
                log_scrape(
                    self.session, current, parsed.track_name,
                    ScrapeStatus.success,
                )
                total_stored += 1

            for track_name in failed_tracks:
                log_scrape(
                    self.session, current, track_name,
                    ScrapeStatus.failed,
                    error_message="empty response or HTTP error",
                )

            if not failed_tracks:
                log_scrape(
                    self.session, current, _ALL_TRACKS_SENTINEL,
                    ScrapeStatus.success,
                )
            self.session.commit()

            # Rate-limit between date fetches.
            if self.tjk_client.delay > 0:
                await asyncio.sleep(self.tjk_client.delay)

            current -= timedelta(days=1)

        logger.info(
            "Full-field results backfill complete: %d races stored", total_stored,
        )
        return total_stored

    async def backfill_historical(
        self,
        from_date: date,
        to_date: date,
        chunk_days: int = 30,
    ) -> int:
        """Backfill using the KosuSorgulama bulk query endpoint.

        Fetches results in date-range chunks to avoid timeouts and large
        responses.  Returns the total number of races stored.

        Parameters
        ----------
        from_date:
            Earliest date (inclusive).
        to_date:
            Latest date (inclusive).
        chunk_days:
            Maximum number of days per query chunk.
        """
        from ganyan.scraper.parser import parse_race_card

        total_stored = 0
        chunk_start = from_date

        while chunk_start <= to_date:
            chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), to_date)

            logger.info(
                "Historical backfill chunk: %s -> %s", chunk_start, chunk_end,
            )
            try:
                raw_cards = await self.tjk_client.fetch_historical_results(
                    chunk_start, chunk_end,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to fetch historical chunk %s -> %s",
                    chunk_start,
                    chunk_end,
                )
                log_scrape(
                    self.session, chunk_start, _ALL_TRACKS_SENTINEL,
                    ScrapeStatus.failed, error_message=str(exc),
                )
                chunk_start = chunk_end + timedelta(days=1)
                continue

            if not raw_cards:
                log_scrape(
                    self.session, chunk_start, _ALL_TRACKS_SENTINEL,
                    ScrapeStatus.skipped,
                )
                chunk_start = chunk_end + timedelta(days=1)
                continue

            for raw in raw_cards:
                parsed = parse_race_card(raw)
                store_historical_race(self.session, parsed)
                total_stored += 1

            log_scrape(
                self.session, chunk_start, _ALL_TRACKS_SENTINEL,
                ScrapeStatus.success,
            )
            self.session.commit()

            chunk_start = chunk_end + timedelta(days=1)

            # Rate-limit between chunks
            if self.tjk_client.delay > 0:
                await asyncio.sleep(self.tjk_client.delay)

        logger.info("Historical backfill complete: %d races stored", total_stored)
        return total_stored
