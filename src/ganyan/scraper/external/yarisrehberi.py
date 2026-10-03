"""yarisrehberi.com tipster-picks scraper.

The site's homepage contains a public "Yapılan At Yarışı Tahminleri"
section listing user-submitted altılı (6-race parlay) tickets.  Each
ticket carries:

- A free-text title (e.g. ``"Ankara 2 Altılı"``) — sometimes specifies
  the track, often does not.
- A timestamp.
- 6 sub-races labeled ``"1. Koşu"`` through ``"6. Koşu"``, each with
  one or more horse-number picks (the gate / program-NO of the
  picked horse, *not* the horse name).
- ``Misli`` (multiplier) and ``Kupon`` (coupon size) metadata.

We persist each (sub_race, horse_number) pair as one signal row.
Resolving sub-race index → TJK race_id is handled lazily: at feature-
extraction time we look up which 6 races composed the altılı bundle
on that date; the binding lives outside this scraper because TJK
publishes it elsewhere (or it has to be derived from the program).

**Limitation observed during framework design (2026-04-26):** the
homepage's "Güncel" tab serves stale picks (Feb 2026) on most days.
Fresh picks sit behind the paid subscription paywall.  This scraper
captures whatever the homepage exposes; the freshness gap is a known
data-quality caveat tracked in the feature pipeline.
"""

from __future__ import annotations

import logging
from ganyan.time import utcnow
import re
from datetime import date as date_type, datetime
from typing import ClassVar

import httpx
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from .base import ExternalSignalRow, ExternalSource


logger = logging.getLogger(__name__)


_HOMEPAGE_URL = "https://yarisrehberi.com/"
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
}

# Each ticket block on the homepage starts with a heading like
# "Ankara 2 Altılı" or a tipster handle (e.g. "son dene").  After the
# title comes a timestamp and 6 ``Koşu`` sub-blocks.  We parse loosely
# because the markup is hand-edited and inconsistent across rows.
_KOSU_RE = re.compile(r"^\s*(\d+)\s*\.\s*Koşu\s*$")


class YarisRehberiTipsterSource(ExternalSource):
    """Scrape the public tipster-pick aggregator on yarisrehberi.com."""

    source_name: ClassVar[str] = "yarisrehberi"
    signal_types: ClassVar[tuple[str, ...]] = ("tipster_pick",)

    def fetch_for_date(
        self, session: Session, target_date: date_type,
    ) -> list[ExternalSignalRow]:
        del session  # resolution happens later; raw rows here
        try:
            with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS) as client:
                resp = client.get(_HOMEPAGE_URL, follow_redirects=True)
                resp.raise_for_status()
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            logger.warning("yarisrehberi fetch failed: %s", exc)
            raise

        return self._parse(resp.text, target_date)

    def _parse(
        self, html: str, target_date: date_type,
    ) -> list[ExternalSignalRow]:
        """Extract ticket blocks → flat list of pick rows.

        Each pick row carries ``payload`` with the un-resolved binding
        info: ticket title, sub-race index (1-6), horse number, scrape
        timestamp.  Race / race_entry resolution is deferred to a
        downstream binder that joins (date, track, sub_race) to TJK's
        published altılı program.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Locate the predictions section.  The homepage uses a heading
        # ``"Yapılan At Yarışı Tahminleri"`` followed by a sequence of
        # ticket cards.  Cards aren't perfectly classed, so we walk
        # forward from the heading until we run out of recognisable
        # sub-blocks.
        section_anchor = soup.find(
            string=re.compile(r"Yapılan At Yarışı Tahminleri", re.IGNORECASE),
        )
        if section_anchor is None:
            return []

        rows: list[ExternalSignalRow] = []
        captured_at = utcnow()

        # Tickets group into parent containers; the simplest heuristic
        # that survives markup churn is: find every ``Koşu`` heading,
        # then walk siblings to gather horse numbers until we hit the
        # next ``Koşu`` heading or a ``Misli`` / ``Kupon`` marker.
        for kosu_h in soup.find_all(
            string=re.compile(r"^\s*\d+\.\s*Koşu\s*$"),
        ):
            m = _KOSU_RE.match(kosu_h)
            if not m:
                continue
            sub_race = int(m.group(1))
            ticket_title = self._title_for(kosu_h)
            ticket_ts = self._timestamp_for(kosu_h)

            # Walk siblings of the parent until we hit numeric markers
            # then stop on the next heading.
            container = kosu_h.parent
            if container is None:
                continue
            picks: list[int] = []
            for sib in container.find_next_siblings():
                txt = sib.get_text(" ", strip=True)
                if not txt:
                    continue
                if re.match(r"^\d+\.\s*Koşu", txt):
                    break
                if "Misli" in txt or "Kupon" in txt:
                    break
                # Each numeric token is a picked horse number.
                for num in re.findall(r"\b(\d{1,2})\b", txt):
                    picks.append(int(num))
                    if len(picks) > 12:
                        break
                if len(picks) > 12:
                    break

            for horse_no in picks:
                rows.append(ExternalSignalRow(
                    source_name=self.source_name,
                    signal_type="tipster_pick",
                    race_id=None,
                    race_entry_id=None,
                    value=None,
                    payload={
                        "ticket_title": ticket_title,
                        "ticket_timestamp": ticket_ts,
                        "sub_race_index": sub_race,
                        "horse_number": horse_no,
                        "for_date": target_date.isoformat(),
                    },
                    captured_at=captured_at,
                ))

        logger.info(
            "yarisrehberi: parsed %d tipster picks for %s",
            len(rows), target_date,
        )
        return rows

    @staticmethod
    def _title_for(node) -> str | None:
        """Walk back to find the ticket's title heading."""
        cur = node.parent
        for _ in range(8):
            if cur is None:
                return None
            prev = cur.find_previous(string=True)
            if prev and len(prev.strip()) > 3 and "Koşu" not in prev:
                return prev.strip()[:80]
            cur = cur.parent
        return None

    @staticmethod
    def _timestamp_for(node) -> str | None:
        """Find the timestamp string (e.g. "12.02.2026 10:28") near the ticket."""
        cur = node.parent
        for _ in range(8):
            if cur is None:
                return None
            txt = cur.get_text(" ", strip=True) if cur else ""
            m = re.search(r"\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}", txt)
            if m:
                return m.group(0)
            cur = cur.parent
        return None
