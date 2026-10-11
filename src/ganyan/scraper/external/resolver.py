"""Bind raw external signals to concrete TJK race / race_entry rows.

Plugins (e.g. ``yarisrehberi``) capture signals in a source-native
format — a tipster's altılı parlay carries ``(sub_race_index 1-6,
horse_number)`` but no TJK race_id.  This module performs the
resolution pass after ingestion: it figures out which 6 TJK races the
altılı bundle covered on that date, joins ``horse_number`` to
``race_entries.gate_number`` to get a concrete row, and back-fills the
``race_id`` / ``race_entry_id`` columns on the original signal.

Heuristic for the altılı-bundle discovery (TJK doesn't publish it
through a clean endpoint):

1. **Title-tagged tickets**: ``"Ankara 2 Altılı"`` → restrict to the
   Ankara card on that date.  ``"İstanbul Altılı"`` → İstanbul.  We
   do prefix matching against known track names.
2. **Untagged tickets**: applied to every track that has ≥6 races on
   the date (so a single ticket may produce multiple resolutions, one
   per track).  Cheap rows; the model can deduplicate via
   ``COUNT(DISTINCT ticket_timestamp)`` at feature time.
3. **Sub-race → TJK race_number**: altılı is by convention the *last
   6* races of the day on the chosen track.  Sub-race 1 maps to
   ``race[-6]`` (chronologically), 6 to ``race[-1]``.

Same-tipster duplicates are not deduplicated at the signal level —
each pick is one row — but the feature aggregator groups by
``ticket_timestamp`` so each ticket counts once per horse.
"""

from __future__ import annotations

import logging
import re
from datetime import date as date_type, timedelta
from typing import Iterable

from sqlalchemy.orm import Session

from ganyan.db.models import (
    ExternalSignal, Race, RaceEntry, Track,
)


logger = logging.getLogger(__name__)


# A ticket whose authored timestamp is older than this is treated as
# stale SEO/cached content and skipped during resolution — yarisrehberi
# notoriously serves Feb 2026 tickets on a current-day fetch.  Tighten
# to 1 day if/when fresh picks become available.
_TICKET_FRESHNESS_DAYS = 3
_DDMMYYYY_RE = re.compile(r"^\s*(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})")


# Token map: ticket-title prefix → canonical TJK track name.  Kept
# explicit so a typo in the source doesn't silently bind to a wrong
# track (e.g. "İzmir" must match "İzmir", not get truncated to "İz").
_TRACK_TOKENS: list[tuple[str, str]] = [
    ("İstanbul", "İstanbul"),
    ("Istanbul", "İstanbul"),
    ("Ankara", "Ankara"),
    ("İzmir", "İzmir"),
    ("Izmir", "İzmir"),
    ("Bursa", "Bursa"),
    ("Adana", "Adana"),
    ("Diyarbakır", "Diyarbakır"),
    ("Diyarbakir", "Diyarbakır"),
    ("Şanlıurfa", "Şanlıurfa"),
    ("Sanliurfa", "Şanlıurfa"),
    ("Elazığ", "Elazığ"),
    ("Elazig", "Elazığ"),
    ("Kocaeli", "Kocaeli"),
    ("Antalya", "Antalya"),
]


def _track_from_title(title: str | None) -> str | None:
    if not title:
        return None
    for token, canonical in _TRACK_TOKENS:
        if token.lower() in title.lower():
            return canonical
    return None


def _ticket_age_days(ticket_ts: str | None, target_date: date_type) -> int | None:
    """Days between the ticket's authored timestamp and ``target_date``.

    Returns ``None`` when the timestamp can't be parsed.  Used to
    suppress stale SEO content from binding to current races.
    """
    if not ticket_ts:
        return None
    m = _DDMMYYYY_RE.match(ticket_ts)
    if not m:
        return None
    dd, mm, yyyy, _hh, _mi = m.groups()
    try:
        ticket_date = date_type(int(yyyy), int(mm), int(dd))
    except ValueError:
        return None
    return abs((target_date - ticket_date).days)


def _race_lookup(
    session: Session, target_date: date_type, track_name: str | None,
) -> dict[str, list[Race]]:
    """Return ``{track_name: [races sorted by race_number]}`` for ``target_date``.

    When ``track_name`` is provided, only that track's races are
    returned (still as a single-key dict for caller uniformity).
    """
    q = (
        session.query(Race)
        .join(Track, Track.id == Race.track_id)
        .filter(Race.date == target_date)
    )
    if track_name is not None:
        q = q.filter(Track.name == track_name)
    out: dict[str, list[Race]] = {}
    for race in q.order_by(Race.race_number.asc()).all():
        out.setdefault(race.track.name, []).append(race)
    return out


def _entry_by_gate(session: Session, race_id: int, gate: int) -> RaceEntry | None:
    """Look up the entry whose gate_number matches ``gate`` in the given race.

    Multiple entries can in theory share a gate (rare data anomaly).
    Returns the first match or ``None`` when nothing binds.
    """
    return (
        session.query(RaceEntry)
        .filter(
            RaceEntry.race_id == race_id,
            RaceEntry.gate_number == gate,
        )
        .order_by(RaceEntry.id.asc())
        .first()
    )


def resolve_unbound_signals(
    session: Session,
    target_date: date_type,
    source_name: str | None = None,
) -> int:
    """Resolve unbound external_signals rows for ``target_date``.

    Iterates rows where ``race_entry_id IS NULL`` whose ``payload``
    carries the un-resolved bundle info.  Sets ``race_id`` and
    ``race_entry_id`` when the heuristic produces a unique match.

    Returns the count of rows updated.  Caller must commit.

    ``source_name`` filters to a specific plugin (useful when only one
    plugin's rows need re-binding); ``None`` resolves across all.
    """
    q = (
        session.query(ExternalSignal)
        .filter(ExternalSignal.race_entry_id.is_(None))
    )
    if source_name is not None:
        q = q.filter(ExternalSignal.source_name == source_name)

    candidates = list(q.all())
    if not candidates:
        return 0

    # Filter to rows whose payload says the pick is for ``target_date``.
    matched = [
        r for r in candidates
        if r.payload and r.payload.get("for_date") == target_date.isoformat()
    ]
    if not matched:
        return 0

    # Pre-load all races for that date by track once, reused across
    # rows.  The per-row gate→entry lookup hits the indexed
    # (race_id, gate_number) columns and is cheap.
    by_track_full = _race_lookup(session, target_date, None)
    if not by_track_full:
        return 0

    n_updated = 0
    n_stale_skipped = 0
    for row in matched:
        payload = row.payload or {}
        sub = int(payload.get("sub_race_index") or 0)
        horse_no = payload.get("horse_number")
        if sub < 1 or sub > 6 or horse_no is None:
            continue

        # Stale-ticket guard: when the source publishes Feb tickets on
        # an April fetch, don't bind them to today's program.  Skip
        # without resolving so they sit unbound (and harmless) in the
        # table.  Only enforce when we *can* parse the timestamp;
        # missing-timestamp tickets get the benefit of the doubt.
        age = _ticket_age_days(payload.get("ticket_timestamp"), target_date)
        if age is not None and age > _TICKET_FRESHNESS_DAYS:
            n_stale_skipped += 1
            continue

        title_track = _track_from_title(payload.get("ticket_title"))
        if title_track and title_track in by_track_full:
            tracks = {title_track: by_track_full[title_track]}
        else:
            tracks = by_track_full

        # Pick first track whose card has ≥6 races and a gate match.
        # When the ticket is untagged we resolve to the first match,
        # not all matches — multiple-track explosion would inflate
        # tipster counts spuriously.
        bound = False
        for track_name, races in tracks.items():
            if len(races) < 6:
                continue
            tjk_race = races[-6 + (sub - 1)]
            entry = _entry_by_gate(session, tjk_race.id, int(horse_no))
            if entry is not None:
                row.race_id = tjk_race.id
                row.race_entry_id = entry.id
                bound = True
                break
        if bound:
            n_updated += 1

    session.flush()
    logger.info(
        "external resolver: %d/%d signal(s) bound for %s "
        "(%d stale skipped)",
        n_updated, len(matched), target_date, n_stale_skipped,
    )
    return n_updated


def _normalize_jockey_name(name: str | None) -> str:
    """Loose match key for jockey names — strip case, whitespace,
    punctuation, dotted initials.  Disciplinary lists use uppercase
    full names (``"SAMET ÇAKAR"``); race-card jockeys often appear as
    ``"S. Çakar"`` or ``"Samet Çakar"``.  We squash both to a
    diacritic-preserved lowercase last-name + first-initial pattern
    so abbreviated forms still match.
    """
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"[\.\,\-_]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _name_match(jockey_full: str | None, entry_jockey: str | None) -> bool:
    """Return True when the discipline-list ``jockey_full`` plausibly
    matches the entry's ``entry_jockey`` field.

    Exact-equal after normalisation, or last-name match when one side
    has only an initial (``"S. Çakar"`` ↔ ``"Samet Çakar"``).
    """
    a = _normalize_jockey_name(jockey_full)
    b = _normalize_jockey_name(entry_jockey)
    if not a or not b:
        return False
    if a == b:
        return True
    a_parts = a.split()
    b_parts = b.split()
    if len(a_parts) >= 2 and len(b_parts) >= 2:
        # Last-name match required, plus first-initial overlap.
        if a_parts[-1] == b_parts[-1]:
            if a_parts[0][:1] == b_parts[0][:1]:
                return True
    return False


def bind_discipline_to_entries(
    session: Session, target_date: date_type,
) -> int:
    """Bind reported_jockey / penalized_jockey signals to entries.

    For each unbound discipline row carrying an active window
    [start_date, end_date], match every race entry whose race date
    falls within that window AND whose jockey field matches the
    disciplinary name.  ``target_date`` is passed for logging/scope
    only — the bind is window-driven so a row scraped today retro-
    fits historical races within its window.

    Returns total updated/duplicated rows.
    """
    candidates = (
        session.query(ExternalSignal)
        .filter(
            ExternalSignal.source_name == "tjk_discipline",
            ExternalSignal.race_entry_id.is_(None),
        )
        .all()
    )
    if not candidates:
        return 0

    n_bound = 0
    n_dup = 0
    for sig in candidates:
        payload = sig.payload or {}
        start = payload.get("start_date")
        end = payload.get("end_date")
        try:
            sd = date_type.fromisoformat(start) if start else None
            ed = date_type.fromisoformat(end) if end else None
        except ValueError:
            sd = ed = None
        if sd is None and ed is None:
            continue

        # Pull entries within the window, then filter by jockey.
        q = session.query(RaceEntry, Race).join(
            Race, Race.id == RaceEntry.race_id,
        )
        if sd is not None:
            q = q.filter(Race.date >= sd)
        if ed is not None:
            q = q.filter(Race.date <= ed)

        jockey_full = payload.get("jockey_name")
        matches = [
            (entry, race) for entry, race in q.all()
            if _name_match(jockey_full, entry.jockey)
        ]
        if not matches:
            continue

        # First match binds the original row; further matches duplicate
        # the signal so each affected entry gets its own row (allows
        # downstream feature COUNT(DISTINCT race_entry_id) to work).
        first_entry, first_race = matches[0]
        sig.race_entry_id = first_entry.id
        sig.race_id = first_race.id
        n_bound += 1
        for entry, race in matches[1:]:
            session.add(ExternalSignal(
                source_name=sig.source_name,
                signal_type=sig.signal_type,
                race_id=race.id,
                race_entry_id=entry.id,
                value=sig.value,
                payload=sig.payload,
                captured_at=sig.captured_at,
            ))
            n_dup += 1

    session.flush()
    logger.info(
        "discipline binder: %d primary, %d duplicated rows (target %s)",
        n_bound, n_dup, target_date,
    )
    return n_bound + n_dup


def bind_workouts_to_entries(
    session: Session, target_date: date_type,
    workout_lookback_days: int = 21,
) -> int:
    """Bind tjk_workouts signals to today's entries by horse name.

    Each workout signal gets resolved against ``Horse.name`` for
    horses entered in races on ``target_date``.  Workouts older than
    ``workout_lookback_days`` are skipped — only recent training
    bears on current fitness; ancient workouts would inflate the
    aggregator without adding signal.

    A horse with multiple recent workouts gets ALL of them bound to
    its entry today; the feature aggregator picks whichever distance
    matches or computes a recency-weighted average.
    """
    from ganyan.db.models import Horse

    cutoff = target_date - timedelta(days=workout_lookback_days)

    candidates = (
        session.query(ExternalSignal)
        .filter(
            ExternalSignal.source_name == "tjk_workouts",
            ExternalSignal.race_entry_id.is_(None),
        )
        .all()
    )
    if not candidates:
        return 0

    # Pre-load today's entries indexed by horse name.  Multiple entries
    # per horse name are uncommon (cross-track collisions); we bind
    # each workout to all matching entries.
    rows = (
        session.query(RaceEntry, Race, Horse)
        .join(Race, Race.id == RaceEntry.race_id)
        .join(Horse, Horse.id == RaceEntry.horse_id)
        .filter(Race.date == target_date)
        .all()
    )
    by_name: dict[str, list[tuple[RaceEntry, Race]]] = {}
    for entry, race, horse in rows:
        if horse.name:
            by_name.setdefault(horse.name.strip().upper(), []).append((entry, race))
    if not by_name:
        return 0

    n_bound = 0
    n_dup = 0
    for sig in candidates:
        payload = sig.payload or {}
        wd = payload.get("workout_date")
        try:
            wdate = date_type.fromisoformat(wd) if wd else None
        except ValueError:
            wdate = None
        if wdate is None or wdate < cutoff or wdate > target_date:
            continue

        horse_name = (payload.get("horse_name") or "").strip().upper()
        matches = by_name.get(horse_name, [])
        if not matches:
            continue

        first_entry, first_race = matches[0]
        sig.race_entry_id = first_entry.id
        sig.race_id = first_race.id
        n_bound += 1
        for entry, race in matches[1:]:
            session.add(ExternalSignal(
                source_name=sig.source_name,
                signal_type=sig.signal_type,
                race_id=race.id,
                race_entry_id=entry.id,
                value=sig.value,
                payload=sig.payload,
                captured_at=sig.captured_at,
            ))
            n_dup += 1

    session.flush()
    logger.info(
        "workout binder: %d primary, %d duplicated rows for %s",
        n_bound, n_dup, target_date,
    )
    return n_bound + n_dup


def bind_track_conditions_to_races(
    session: Session, target_date: date_type,
) -> int:
    """Bind track-condition + steward-report signals to races by
    (track, reading_date) — *not* restricted to ``target_date``.

    Empirical observation (2026-04): TJK's PistBilgileri endpoint
    returns *historical* readings, not today's live conditions.  A
    fetch on 2026-04-26 yielded readings from 2026-03-08 through
    2026-04-02.  Restricting the bind to ``target_date`` drops the
    entire batch.

    The honest semantics is: bind each reading to whichever race
    matches (track, reading_date), regardless of when we scraped.
    This retroactively enriches *past* races with their weather;
    today's races stay NaN until TJK eventually publishes today's
    reading (or never, if this endpoint truly is archive-only).

    Same logic for steward_report signals — they bind by
    (track, report_date).
    """
    candidates = (
        session.query(ExternalSignal)
        .filter(
            ExternalSignal.source_name.in_(
                ["tjk_track_conditions", "tjk_steward_reports"],
            ),
            ExternalSignal.race_id.is_(None),
        )
        .all()
    )
    if not candidates:
        return 0

    # Group candidates by (track_city, reading_date) so we can do one
    # race lookup per group instead of one per signal.
    groups: dict[tuple[str, date_type], list[ExternalSignal]] = {}
    for sig in candidates:
        payload = sig.payload or {}
        rd = payload.get("reading_date") or payload.get("report_date")
        try:
            read_date = date_type.fromisoformat(rd) if rd else None
        except ValueError:
            read_date = None
        track_city = (payload.get("track_city") or "").strip()
        if not track_city or read_date is None:
            continue
        groups.setdefault((track_city, read_date), []).append(sig)

    n_bound = 0
    n_dup = 0
    for (track_city, read_date), sigs in groups.items():
        races = (
            session.query(Race)
            .join(Track, Track.id == Race.track_id)
            .filter(Race.date == read_date, Track.name == track_city)
            .all()
        )
        if not races:
            continue
        for sig in sigs:
            sig.race_id = races[0].id
            n_bound += 1
            for race in races[1:]:
                session.add(ExternalSignal(
                    source_name=sig.source_name,
                    signal_type=sig.signal_type,
                    race_id=race.id,
                    race_entry_id=None,
                    value=sig.value,
                    payload=sig.payload,
                    captured_at=sig.captured_at,
                ))
                n_dup += 1

    session.flush()
    logger.info(
        "track-conditions binder: %d primary, %d duplicated (across past races)",
        n_bound, n_dup,
    )
    return n_bound + n_dup


def fetch_and_resolve(
    session: Session, target_date: date_type, sources: Iterable[str] | None = None,
) -> dict[str, int]:
    """Run every registered source, persist, then resolve.

    Convenience for the scheduler.  ``sources`` filters which plugins
    to fire (default: all registered).  Returns ``{source: n_persisted}``;
    resolution count is logged but not returned per-source since
    resolution is cross-source by date.
    """
    from . import REGISTRY
    from .base import persist_signals

    results: dict[str, int] = {}
    plugins = sources or list(REGISTRY.keys())
    for name in plugins:
        cls = REGISTRY.get(name)
        if cls is None:
            logger.warning("external: unknown source %s, skipping", name)
            continue
        plugin = cls()
        try:
            rows = plugin.fetch_for_date(session, target_date)
            n = persist_signals(session, rows)
            results[name] = n
        except Exception:  # noqa: BLE001
            logger.exception("external: %s fetch failed", name)
            raise
    # Tipster-bundle resolver (yarisrehberi + future altılı sources).
    resolve_unbound_signals(session, target_date)
    # Discipline name-match resolver (TJK reported/penalized jockeys).
    bind_discipline_to_entries(session, target_date)
    # Workout name-match resolver (TJK İdman İstatistikleri).
    bind_workouts_to_entries(session, target_date)
    # Track-conditions + steward-reports race-level binding.
    bind_track_conditions_to_races(session, target_date)
    return results
