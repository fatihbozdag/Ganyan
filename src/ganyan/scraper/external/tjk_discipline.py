"""TJK reported & penalized jockeys scraper.

TJK publishes two query pages with disciplinary information:

- ``/TR/YarisSever/Query/Page/RaporluJokey`` — currently medically-
  reported jockeys (``rapor`` = doctor's note).  Fields:
  ``(JokeyAdi, BaslangicTarihi, BitTarihi)``.  Active *today* iff the
  current date is between Başlangıç and Bit dates.
- ``/TR/YarisSever/Query/Page/CezaliJokey`` — penalized jockeys
  (``ceza`` = punishment).  Fields: ``(JokeyAdi, AtAdi, BaslangicTarihi,
  BitisTarihi, CezaNedeni, Ceza)``.  Active during the same date window.

The actual data is loaded via AJAX from
``/TR/YarisSever/Query/DataRows/...`` (POST with ``PageNumber=1``),
returning HTML rows with class names like ``sorgu-RaporluJokey-JokeyAdi``.

This plugin emits one ``external_signals`` row per disciplined jockey
with ``signal_type`` either ``reported_jockey`` or ``penalized_jockey``.
The discipline-window dates land in the ``payload`` so the resolver
(specifically :func:`bind_discipline_to_entries`) can match jockey
names to ``race_entries.jockey`` only on races whose date falls
inside the active window.

Strong signal: when a horse's listed jockey is on the reported list
and the race date is within the active window, that horse is *certain*
to get a substitute rider — and the new rider's identity is unknown to
the model.  Captures one of the few last-minute factors a public
scraper can extract.
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
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}

_REPORTED_URL = f"{_BASE_URL}/TR/YarisSever/Query/DataRows/RaporluJokey"
_REPORTED_REFERER = f"{_BASE_URL}/TR/YarisSever/Query/Page/RaporluJokey"
_PENALIZED_URL = f"{_BASE_URL}/TR/YarisSever/Query/DataRows/CezaliJokey"
_PENALIZED_REFERER = f"{_BASE_URL}/TR/YarisSever/Query/Page/CezaliJokey"


# Each row is a `<tr>...<td class="sorgu-{Page}-{Field}">value</td>...`
# block.  Same regex skeleton works for both pages; only the page name
# differs in the class prefix.
_ROW_RE = re.compile(r"<tr[^>]*>([\s\S]*?)</tr>")


def _make_cell_re(page_name: str) -> re.Pattern:
    return re.compile(
        rf'<td class="sorgu-{page_name}-([^"]+)"[^>]*>([\s\S]*?)</td>'
    )


_REPORTED_CELL = _make_cell_re("RaporluJokey")
_PENALIZED_CELL = _make_cell_re("CezaliJokey")


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


def _fetch_rows(
    url: str, referer: str, page_name: str,
    cell_re: re.Pattern,
) -> list[dict[str, str]]:
    """Fetch one of the discipline pages and return parsed row dicts."""
    headers = {**_HEADERS, "Referer": referer}
    try:
        with httpx.Client(timeout=_TIMEOUT, headers=headers) as client:
            resp = client.post(url, data={"PageNumber": "1"})
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("TJK %s fetch failed: %s", page_name, exc)
        raise

    rows: list[dict[str, str]] = []
    for tr_match in _ROW_RE.finditer(resp.text):
        cells: dict[str, str] = {}
        for c in cell_re.finditer(tr_match.group(1)):
            field, raw = c.group(1), c.group(2)
            text = re.sub(r"<[^>]+>", " ", raw)
            text = " ".join(text.split())
            cells[field] = text
        if cells:
            rows.append(cells)
    return rows


class TjkDisciplineSource(ExternalSource):
    """Scrape TJK's RaporluJokey + CezaliJokey discipline lists."""

    source_name: ClassVar[str] = "tjk_discipline"
    signal_types: ClassVar[tuple[str, ...]] = (
        "reported_jockey", "penalized_jockey",
    )

    def fetch_for_date(
        self, session: Session, target_date: date_type,
    ) -> list[ExternalSignalRow]:
        del session  # binding happens in the resolver
        captured_at = utcnow()
        out: list[ExternalSignalRow] = []

        # Reported (medical) jockeys.
        for cells in _fetch_rows(
            _REPORTED_URL, _REPORTED_REFERER, "RaporluJokey", _REPORTED_CELL,
        ):
            jockey = cells.get("JokeyAdi", "").strip()
            if not jockey:
                continue
            start = _parse_ddmmyyyy(cells.get("BaslangicTarihi"))
            # Field name varies: RaporluJokey uses "BitTarihi",
            # CezaliJokey uses "BitisTarihi".  Tolerate both.
            end = _parse_ddmmyyyy(
                cells.get("BitTarihi") or cells.get("BitisTarihi"),
            )
            out.append(ExternalSignalRow(
                source_name=self.source_name,
                signal_type="reported_jockey",
                payload={
                    "jockey_name": jockey,
                    "start_date": start.isoformat() if start else None,
                    "end_date": end.isoformat() if end else None,
                    "for_date": target_date.isoformat(),
                },
                captured_at=captured_at,
            ))

        # Penalized jockeys.
        for cells in _fetch_rows(
            _PENALIZED_URL, _PENALIZED_REFERER, "CezaliJokey", _PENALIZED_CELL,
        ):
            jockey = cells.get("JokeyAdi", "").strip()
            if not jockey:
                continue
            start = _parse_ddmmyyyy(cells.get("BaslangicTarihi"))
            end = _parse_ddmmyyyy(
                cells.get("BitisTarihi") or cells.get("BitTarihi"),
            )
            payload = {
                "jockey_name": jockey,
                "horse_name": cells.get("AtAdi", "").strip() or None,
                "reason": (cells.get("CezaNedeni") or "")[:500] or None,
                "penalty": (cells.get("Ceza") or "")[:500] or None,
                "start_date": start.isoformat() if start else None,
                "end_date": end.isoformat() if end else None,
                "for_date": target_date.isoformat(),
            }
            out.append(ExternalSignalRow(
                source_name=self.source_name,
                signal_type="penalized_jockey",
                payload=payload,
                captured_at=captured_at,
            ))

        logger.info(
            "tjk_discipline: %d total signals (reported + penalized) for %s",
            len(out), target_date,
        )
        return out
