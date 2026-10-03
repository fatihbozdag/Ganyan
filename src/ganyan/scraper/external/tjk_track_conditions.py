"""TJK Pist Bilgileri (track conditions) + Komiser Raporları (steward
reports) — two thin plugins published together because they share a
common DataRows structure with sparse columns.

PistBilgileri yields per-track-per-date weather / surface readings:
    SehirAdi      (city — track key)
    HipodromAdi   (full track name)
    Tarih         (date)
    SAAT          (time of reading)
    Sicaklik      (temperature, °C, integer string)
    NEM           (humidity, %)
    BASINC        (pressure, mb)
    Gokyuzu       (sky condition: "Parçalı Bulutlu", "Açık", etc.)
    RUZGAR        (wind: e.g. "23 km/sa B")

This data binds at the race level (every entry in a given race shares
the same conditions), so we attach to ``Race.id`` rather than per
``RaceEntry.id``.

KomiserRaporlari is sparser — only ``(Rapor, SehirAdi, Tarih)`` — the
full report text sits behind a detail page link.  We persist the
existence flag for now; full text extraction is a larger NLP task
deferred for later.
"""

from __future__ import annotations

import logging
from ganyan.time import utcnow
import re
from datetime import date as date_type, datetime
from typing import ClassVar

import httpx
from sqlalchemy.orm import Session

from .base import ExternalSignalRow, ExternalSource


logger = logging.getLogger(__name__)


_BASE_URL = "https://www.tjk.org"
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}

_ROW_RE = re.compile(r"<tr[^>]*>([\s\S]*?)</tr>")


def _make_cell_re(page_name: str) -> re.Pattern:
    return re.compile(
        rf'<td class="sorgu-{page_name}-([^"]+)"[^>]*>([\s\S]*?)</td>'
    )


def _fetch_table(page_name: str) -> list[dict[str, str]]:
    """POST to /Query/DataRows/<page_name>, return list of row dicts."""
    headers = {
        **_HEADERS,
        "Referer": f"{_BASE_URL}/TR/YarisSever/Query/Page/{page_name}",
    }
    url = f"{_BASE_URL}/TR/YarisSever/Query/DataRows/{page_name}"
    try:
        with httpx.Client(timeout=_TIMEOUT, headers=headers) as client:
            resp = client.post(url, data={"PageNumber": "1"})
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("TJK %s fetch failed: %s", page_name, exc)
        raise

    cell_re = _make_cell_re(page_name)
    out: list[dict[str, str]] = []
    for tr in _ROW_RE.finditer(resp.text):
        cells: dict[str, str] = {}
        for c in cell_re.finditer(tr.group(1)):
            cells[c.group(1)] = " ".join(
                re.sub(r"<[^>]+>", " ", c.group(2)).split()
            )
        if cells:
            out.append(cells)
    return out


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


def _safe_int(s: str | None) -> int | None:
    if not s:
        return None
    m = re.match(r"-?\d+", s.strip())
    return int(m.group(0)) if m else None


def _parse_wind_kph(s: str | None) -> int | None:
    """``"23 km/sa B"`` → ``23``.  Direction (B/D/G/K) discarded."""
    if not s:
        return None
    m = re.match(r"\s*(\d+)\s*km", s)
    return int(m.group(1)) if m else None


# Sky condition canonical bucket keys — Turkish phrases coalesced into
# a small ordinal so the model can split on weather quality.
_SKY_BUCKETS = {
    "açık": 0,           # clear
    "parçalı bulutlu": 1, # partly cloudy
    "bulutlu": 2,         # cloudy
    "yağmurlu": 3,        # rainy
    "kapalı": 4,          # overcast
}


def _sky_bucket(s: str | None) -> int | None:
    if not s:
        return None
    norm = s.strip().lower()
    for key, bucket in _SKY_BUCKETS.items():
        if key in norm:
            return bucket
    return None


# ---------------------------------------------------------------------------
# Plugin: PistBilgileri
# ---------------------------------------------------------------------------


class TjkTrackConditionsSource(ExternalSource):
    """Per-track weather + surface readings."""

    source_name: ClassVar[str] = "tjk_track_conditions"
    signal_types: ClassVar[tuple[str, ...]] = ("track_conditions",)

    def fetch_for_date(
        self, session: Session, target_date: date_type,
    ) -> list[ExternalSignalRow]:
        del session
        captured_at = utcnow()
        rows = _fetch_table("PistBilgileri")
        out: list[ExternalSignalRow] = []
        for cells in rows:
            track_date = _parse_ddmmyyyy(cells.get("Tarih"))
            if track_date is None:
                continue
            payload = {
                "track_city": cells.get("SehirAdi") or None,
                "track_name": cells.get("HipodromAdi") or None,
                "reading_date": track_date.isoformat(),
                "reading_time": cells.get("SAAT") or None,
                "temperature_c": _safe_int(cells.get("Sicaklik")),
                "humidity_pct": _safe_int(cells.get("NEM")),
                "pressure_mb": _safe_int(cells.get("BASINC")),
                "sky_text": cells.get("Gokyuzu") or None,
                "sky_bucket": _sky_bucket(cells.get("Gokyuzu")),
                "wind_kph": _parse_wind_kph(cells.get("RUZGAR")),
                "wind_text": cells.get("RUZGAR") or None,
                "for_date": target_date.isoformat(),
            }
            out.append(ExternalSignalRow(
                source_name=self.source_name,
                signal_type="track_conditions",
                payload=payload,
                captured_at=captured_at,
            ))
        logger.info(
            "tjk_track_conditions: %d readings parsed for %s",
            len(out), target_date,
        )
        return out


# ---------------------------------------------------------------------------
# Plugin: KomiserRaporlari (steward report existence flag)
# ---------------------------------------------------------------------------


class TjkStewardReportsSource(ExternalSource):
    """Index of steward-report existence per (track, date).

    Only the existence flag is captured — the full report text sits
    behind a detail page that needs follow-up requests + NLP, deferred
    until the existence flag itself shows predictive value.
    """

    source_name: ClassVar[str] = "tjk_steward_reports"
    signal_types: ClassVar[tuple[str, ...]] = ("steward_report",)

    def fetch_for_date(
        self, session: Session, target_date: date_type,
    ) -> list[ExternalSignalRow]:
        del session
        captured_at = utcnow()
        rows = _fetch_table("KomiserRaporlari")
        out: list[ExternalSignalRow] = []
        for cells in rows:
            d = _parse_ddmmyyyy(cells.get("Tarih"))
            if d is None:
                continue
            out.append(ExternalSignalRow(
                source_name=self.source_name,
                signal_type="steward_report",
                value=1.0,  # presence flag
                payload={
                    "track_city": cells.get("SehirAdi") or None,
                    "report_date": d.isoformat(),
                    "for_date": target_date.isoformat(),
                },
                captured_at=captured_at,
            ))
        logger.info(
            "tjk_steward_reports: %d report existences for %s",
            len(out), target_date,
        )
        return out
