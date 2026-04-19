"""Per-horse detail crawler.

Fetches TJK's ``AtKosuBilgileri`` page for each horse in the DB and
extracts pedigree (sire, dam, birth_date).  TJK tags each horse with a
stable ``AtId`` that appears in the ``href`` of horse-name links on the
results page — once we capture it, we can hit the detail endpoint
directly.

The crawler is idempotent: ``profile_crawled_at`` marks horses we've
already processed, so re-runs only hit horses added since the last
crawl (or explicitly requested via ``horse_ids=``).
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import date as date_type, datetime

import httpx
from bs4 import BeautifulSoup, Tag
from sqlalchemy.orm import Session

from ganyan.db.models import Horse


logger = logging.getLogger(__name__)


_DETAIL_PATH = "/TR/YarisSever/Query/ConnectedPage/AtKosuBilgileri"
_KUNYE_SELECTOR = "div.kunye"


@dataclass
class HorseProfile:
    """Clean pedigree data returned by the parser."""

    at_id: int
    name: str
    sire: str | None = None
    dam: str | None = None
    birth_date: date_type | None = None
    origin: str | None = None


def _parse_birth_date(text: str) -> date_type | None:
    """Parse TJK dates like ``"5.04.2023"`` or ``"15.12.2022"``."""
    if not text:
        return None
    m = re.match(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})", text.strip())
    if not m:
        return None
    try:
        return date_type(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _parse_kunye(html: str, at_id: int) -> HorseProfile | None:
    """Extract pedigree fields from a detail-page HTML string.

    Returns ``None`` when the kunye block isn't present (e.g. page
    redirected to a "horse not found" stub).
    """
    soup = BeautifulSoup(html, "html.parser")
    kunye = soup.select_one(_KUNYE_SELECTOR)
    if kunye is None:
        return None

    spans = kunye.find_all("span")
    # Pairs come as [key, value, key, value, ...].
    pairs: dict[str, str] = {}
    for i in range(0, len(spans) - 1, 2):
        key = spans[i].get_text(strip=True)
        val = spans[i + 1].get_text(strip=True)
        if key:
            pairs[key] = val

    name = pairs.get("İsim") or ""
    return HorseProfile(
        at_id=at_id,
        name=name,
        sire=pairs.get("Baba") or None,
        dam=pairs.get("Anne") or None,
        birth_date=_parse_birth_date(pairs.get("Doğ. Trh", "")),
        origin=pairs.get("Orijin") or None,
    )


class HorseCrawler:
    """Async crawler that populates Horse pedigree columns.

    Usage::

        async with HorseCrawler(session, base_url) as crawler:
            await crawler.crawl_missing_profiles(limit=100)
    """

    def __init__(
        self,
        session: Session,
        base_url: str = "https://www.tjk.org",
        *,
        delay: float = 0.5,
        concurrency: int = 5,
        timeout: float = 30.0,
    ) -> None:
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.concurrency = max(1, concurrency)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            follow_redirects=True,
            timeout=timeout,
        )

    async def __aenter__(self) -> HorseCrawler:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def crawl_horses(
        self, horses: list[Horse],
    ) -> int:
        """Crawl the given horses.  Returns the count persisted."""
        semaphore = asyncio.Semaphore(self.concurrency)
        results = await asyncio.gather(
            *(self._crawl_one(h, semaphore) for h in horses if h.tjk_at_id is not None),
            return_exceptions=True,
        )

        stored = 0
        for h, outcome in zip(
            [h for h in horses if h.tjk_at_id is not None], results,
        ):
            if isinstance(outcome, Exception):
                logger.warning(
                    "Crawl failed for %s (at_id=%s): %s",
                    h.name, h.tjk_at_id, outcome,
                )
                continue
            if outcome is None:
                continue
            self._apply_profile(h, outcome)
            stored += 1

        self.session.commit()
        return stored

    async def crawl_missing_profiles(self, *, limit: int | None = None) -> int:
        """Crawl horses that have an ``tjk_at_id`` but no pedigree yet.

        Returns the number of horses updated in this pass.
        """
        q = (
            self.session.query(Horse)
            .filter(
                Horse.tjk_at_id.isnot(None),
                Horse.profile_crawled_at.is_(None),
            )
            .order_by(Horse.id.asc())
        )
        if limit is not None:
            q = q.limit(limit)
        horses = q.all()
        if not horses:
            return 0
        logger.info("Crawling %d horse profiles", len(horses))
        return await self.crawl_horses(horses)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _crawl_one(
        self, horse: Horse, semaphore: asyncio.Semaphore,
    ) -> HorseProfile | None:
        async with semaphore:
            try:
                resp = await self._client.get(
                    _DETAIL_PATH,
                    params={
                        "1": "1",
                        "QueryParameter_AtId": str(horse.tjk_at_id),
                    },
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(
                    "HTTP error for %s (at_id=%s): %s",
                    horse.name, horse.tjk_at_id, exc,
                )
                return None
            finally:
                if self.delay > 0:
                    await asyncio.sleep(self.delay)
            return _parse_kunye(resp.text, int(horse.tjk_at_id))

    def _apply_profile(self, horse: Horse, profile: HorseProfile) -> None:
        """Merge a parsed :class:`HorseProfile` onto the ORM object."""
        if profile.sire and not horse.sire:
            horse.sire = profile.sire
        if profile.dam and not horse.dam:
            horse.dam = profile.dam
        if profile.birth_date and not horse.birth_date:
            horse.birth_date = profile.birth_date
        if profile.origin and not horse.origin:
            horse.origin = profile.origin
        horse.profile_crawled_at = datetime.now()
