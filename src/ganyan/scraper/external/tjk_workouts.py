"""TJK İdman İstatistikleri (workout statistics) scraper.

Per-horse workout data — split times, date, track, surface, workout
jockey.  This is the signal that previously sat behind paywalls
(yarisrehberi.com) or required paddock access; TJK exposes it
directly under ``/Query/DataRows/IdmanIstatistikleri``.

Schema (as of 2026-04):
    ATADI       — horse name (primary join key)
    INFO1400 / INFO1200 / INFO1000 / INFO800 / INFO600 / INFO400 / INFO200
                — split times at each distance, format ``M.SS.HH``
    GALOPKISA   — workout status code (HÇ, R, ÇR, ...)
    IDMANTARIH  — workout date DD.MM.YYYY
    KOSTUGUHIP  — workout track
    PISTTUR     — track type (Kum / Çim)
    IDMANTUR    — workout type (Galop = gallop)
    JOKEY       — workout jockey

Each horse typically posts 1-2 timed splits per workout; most rows
have an 800m + 400m pair.  We persist *every* timed split as a
separate signal so feature aggregation can pick whichever distance
matches the upcoming race.

Resolution: match ATADI to ``horses.name`` for entries on
``target_date`` whose horse hasn't yet had this workout bound.

Pagination: TJK serves 50 rows per ``PageNumber``.  Default fetch is
page 1 only (the last day's workouts) since older workouts get
captured on subsequent runs.  Override ``max_pages`` to backfill.
"""

from __future__ import annotations

import logging
from ganyan.time import utcnow
import re
from datetime import date as date_type
from typing import ClassVar

import httpx
from sqlalchemy.orm import Session

from .base import ExternalSignalRow, ExternalSource


logger = logging.getLogger(__name__)


_BASE_URL = "https://www.tjk.org"
_DATA_URL = f"{_BASE_URL}/TR/YarisSever/Query/DataRows/IdmanIstatistikleri"
_REFERER = f"{_BASE_URL}/TR/YarisSever/Query/Page/IdmanIstatistikleri"
_TIMEOUT = httpx.Timeout(20.0, connect=5.0)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": _REFERER,
}

_ROW_RE = re.compile(r"<tr[^>]*>([\s\S]*?)</tr>")
_CELL_RE = re.compile(
    r'<td class="sorgu-IdmanIstatistikleri-([^"]+)"[^>]*>([\s\S]*?)</td>'
)
# Split-time columns in ascending distance order — handy for picking
# whichever distance was actually timed.
_SPLIT_COLUMNS = (
    ("INFO200", 200),
    ("INFO400", 400),
    ("INFO600", 600),
    ("INFO800", 800),
    ("INFO1000", 1000),
    ("INFO1200", 1200),
    ("INFO1400", 1400),
)


def _parse_time_to_seconds(s: str | None) -> float | None:
    """``"0.58.70"`` → 58.70 seconds.  ``"1.30.45"`` → 90.45.
    Same M.SS.HH layout as TJK EID.  Tolerates malformed strings.
    """
    if not s or not s.strip():
        return None
    parts = s.strip().split(".")
    try:
        if len(parts) == 3:
            m = int(parts[0]) if parts[0] else 0
            sec = int(parts[1]) if parts[1] else 0
            hund = int(parts[2]) if parts[2] else 0
            return m * 60 + sec + hund / 100
        if len(parts) == 2:
            sec = int(parts[0]) if parts[0] else 0
            hund = int(parts[1]) if parts[1] else 0
            return sec + hund / 100
    except ValueError:
        return None
    return None


def _parse_ddmmyyyy(s: str | None) -> date_type | None:
    if not s:
        return None
    m = re.match(r"\s*(\d{2})\.(\d{2})\.(\d{4})", s)
    if not m:
        return None
    try:
        return date_type(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _parse_page(html: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for tr in _ROW_RE.finditer(html):
        cells: dict[str, str] = {}
        for c in _CELL_RE.finditer(tr.group(1)):
            field, raw = c.group(1), c.group(2)
            text = re.sub(r"<[^>]+>", " ", raw)
            cells[field] = " ".join(text.split())
        if cells.get("ATADI"):
            rows.append(cells)
    return rows


class TjkWorkoutSource(ExternalSource):
    """Scrape TJK workout statistics — per-horse split times.

    Each captured row produces ONE signal per timed split (a single
    workout typically yields 1-2 signals: an 800m and a 400m row).
    Bound by ``ATADI`` → ``horses.name`` against today's race entries.
    """

    source_name: ClassVar[str] = "tjk_workouts"
    signal_types: ClassVar[tuple[str, ...]] = ("workout_split",)

    def __init__(self, max_pages: int = 10) -> None:
        # Default 10 pages = 500 most recent workouts ≈ 7-10 days of
        # training history.  Horses racing *today* typically had
        # their last workout 2-7 days ago, not yesterday — pulling
        # only page 1 misses the population we need to bind.  10 is
        # a low-traffic compromise (10 HTTP fetches per scrape, run
        # twice daily by the scheduler).
        self.max_pages = max(1, max_pages)

    def fetch_for_date(
        self, session: Session, target_date: date_type,
    ) -> list[ExternalSignalRow]:
        del session
        captured_at = utcnow()
        all_rows: list[dict[str, str]] = []
        try:
            with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
                for page in range(1, self.max_pages + 1):
                    resp = client.post(
                        _DATA_URL, data={"PageNumber": str(page)},
                    )
                    resp.raise_for_status()
                    parsed = _parse_page(resp.text)
                    if not parsed:
                        break
                    all_rows.extend(parsed)
                    if len(parsed) < 50:
                        break  # last page
        except httpx.HTTPError as exc:
            logger.warning("tjk_workouts fetch failed: %s", exc)
            raise

        signals: list[ExternalSignalRow] = []
        for row in all_rows:
            horse_name = row.get("ATADI", "").strip()
            if not horse_name:
                continue
            workout_date = _parse_ddmmyyyy(row.get("IDMANTARIH"))
            base_payload = {
                "horse_name": horse_name,
                "workout_date": workout_date.isoformat() if workout_date else None,
                "workout_track": row.get("KOSTUGUHIP") or None,
                "workout_surface": row.get("PISTTUR") or None,
                "workout_type": row.get("IDMANTUR") or None,
                "workout_jockey": row.get("JOKEY") or None,
                "status_code": row.get("GALOPKISA") or None,
                "for_date": target_date.isoformat(),
            }
            # One signal per timed split.  Skip empty cells.
            emitted = False
            for col, distance_m in _SPLIT_COLUMNS:
                t_str = row.get(col)
                seconds = _parse_time_to_seconds(t_str)
                if seconds is None:
                    continue
                signals.append(ExternalSignalRow(
                    source_name=self.source_name,
                    signal_type="workout_split",
                    value=seconds,
                    payload={
                        **base_payload,
                        "distance_m": distance_m,
                        "raw_time": t_str,
                    },
                    captured_at=captured_at,
                ))
                emitted = True
            # Defensive: a workout with zero timed splits shouldn't
            # generate orphan signals.  Already handled by ``emitted``;
            # log if a row had ATADI but no usable times.
            if not emitted:
                logger.debug(
                    "tjk_workouts: row for %s on %s had no timed splits",
                    horse_name, workout_date,
                )

        logger.info(
            "tjk_workouts: %d workout splits parsed across %d horses",
            len(signals), len({s.payload.get("horse_name") for s in signals}),
        )
        return signals
