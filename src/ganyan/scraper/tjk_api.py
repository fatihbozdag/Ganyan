"""TJK (Turkish Jockey Club) async HTTP client.

Fetches daily race programs and results from tjk.org, parses the HTML,
and returns lists of RawRaceCard dataclasses ready for downstream processing.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Awaitable, Callable, TypeVar

import httpx
from bs4 import BeautifulSoup, Tag

from ganyan.scraper.parser import RawHorseEntry, RawRaceCard

logger = logging.getLogger(__name__)

T = TypeVar("T")

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 2.0  # seconds; grows as base ** attempt
_MAX_HISTORICAL_PAGES = 500  # hard cap: 500 pages × 50 rows = 25k winners per chunk


async def _with_retry(
    operation: Callable[[], Awaitable[T]],
    label: str,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    backoff_base: float = _DEFAULT_BACKOFF_BASE,
) -> T | None:
    """Run ``operation`` with exponential-backoff retries on httpx errors.

    Returns the operation's result, or ``None`` after the final attempt fails.
    Retries on :class:`httpx.TransportError` (network) and 5xx responses.
    4xx responses are not retried — they indicate a client-side problem.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await operation()
        except httpx.HTTPStatusError as exc:
            # Only retry server errors; client errors (4xx) are permanent.
            status = exc.response.status_code
            last_exc = exc
            if status < 500 or attempt == max_retries:
                logger.error(
                    "%s failed (status %d, attempt %d/%d): %s",
                    label, status, attempt, max_retries, exc,
                )
                return None
            wait = backoff_base ** attempt
            logger.warning(
                "%s status %d, retrying in %.1fs (attempt %d/%d)",
                label, status, wait, attempt, max_retries,
            )
            await asyncio.sleep(wait)
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt == max_retries:
                logger.error(
                    "%s failed after %d attempts: %s", label, max_retries, exc,
                )
                return None
            wait = backoff_base ** attempt
            logger.warning(
                "%s transient error, retrying in %.1fs (attempt %d/%d): %s",
                label, wait, attempt, max_retries, exc,
            )
            await asyncio.sleep(wait)
    # Unreachable, but keeps type-checker happy
    if last_exc is not None:
        logger.error("%s exhausted retries: %s", label, last_exc)
    return None

# ---------------------------------------------------------------------------
# CSS selectors (derived from live TJK HTML as of 2026-04)
# ---------------------------------------------------------------------------

# Full-page track tabs
_SEL_TRACK_TABS = "ul.gunluk-tabs li a[data-sehir-id]"

# Race pane containers (one per race within a track page)
_SEL_RACE_PANES = "div.races-panes > div"

# Race header elements inside each pane
_SEL_RACE_NO = "h3.race-no"
_SEL_RACE_CONFIG = "h3.race-config"

# Horse table (both program and results use tablesorter)
_SEL_HORSE_TABLE = "table.tablesorter"

# --- Program column classes ---
_P = "gunluk-GunlukYarisProgrami"
_P_NAME = f"td.{_P}-AtAdi"
_P_AGE = f"td.{_P}-Yas"
_P_ORIGIN = f"td.{_P}-Baba"
_P_WEIGHT = f"td.{_P}-Kilo"
_P_JOCKEY = f"td.{_P}-JokeAdi"
_P_OWNER = f"td.{_P}-SahipAdi"
_P_TRAINER = f"td.{_P}-AntronorAdi"
_P_GATE = f"td.{_P}-StartId"
_P_HP = f"td.{_P}-Hc"
_P_LAST6 = f"td.{_P}-Son6Yaris"
_P_KGS = f"td.{_P}-KGS"
_P_S20 = f"td.{_P}-s20"
_P_EID = f"td.{_P}-DERECE"
_P_GNY = f"td.{_P}-Gny"
_P_AGF = f"td.{_P}-AGFORAN"

# --- Results column classes ---
_R = "gunluk-GunlukYarisSonuclari"
_R_FINISH = f"td.{_R}-SONUCNO"
_R_NAME = f"td.{_R}-AtAdi3"
_R_AGE = f"td.{_R}-Yas"
_R_ORIGIN = f"td.{_R}-Baba"
_R_WEIGHT = f"td.{_R}-Kilo"
_R_JOCKEY = f"td.{_R}-JokeAdi"
_R_OWNER = f"td.{_R}-SahipAdi"
_R_TRAINER = f"td.{_R}-AntronorAdi"
_R_TIME = f"td.{_R}-Derece"
_R_GNY = f"td.{_R}-Gny"
_R_AGF = f"td.{_R}-AGFORAN"
_R_GATE = f"td.{_R}-StartId"
_R_HP = f"td.{_R}-Hc"

# Endpoints (relative to base_url)
_PROGRAM_PAGE = "/TR/YarisSever/Info/Page/GunlukYarisProgrami"
_PROGRAM_CITY = "/TR/YarisSever/Info/Sehir/GunlukYarisProgrami"
_RESULTS_PAGE = "/TR/YarisSever/Info/Page/GunlukYarisSonuclari"
_RESULTS_CITY = "/TR/YarisSever/Info/Sehir/GunlukYarisSonuclari"

# Historical query endpoints (KosuSorgulama bulk query)
_QUERY_DATA = "/TR/YarisSever/Query/Data/KosuSorgulama"
_QUERY_DATA_ROWS = "/TR/YarisSever/Query/DataRows/KosuSorgulama"

# --- Query result column classes ---
_Q = "sorgu-KosuSorgulama"
_Q_DATE = f"td.{_Q}-Tarih"
_Q_CITY = f"td.{_Q}-Sehir"
_Q_RACE_NUM = f"td.{_Q}-KosuSirasi"
_Q_GROUP = f"td.{_Q}-KosuGrubuAdi"
_Q_RACE_TYPE = f"td.{_Q}-KosuCinsiAdi"
_Q_DISTANCE = f"td.{_Q}-Mesafe"
_Q_SURFACE = f"td.{_Q}-PistAdi"
_Q_WEIGHT = f"td.{_Q}-Kilo"
_Q_ORIGIN = f"td.{_Q}-BabaAnne"
_Q_PRIZE = f"td.{_Q}-IKRAMIYE"
_Q_WINNER = f"td.{_Q}-BirinciAtAdi"
_Q_AGE = f"td.{_Q}-BirinciAtAdiYas"
_Q_TIME = f"td.{_Q}-BirinciAtDerece"
_Q_HP = f"td.{_Q}-HandikapPuani"

_QUERY_RESULTS_PER_PAGE = 50

# Known Turkish domestic track SehirIds (from TJK website navigation)
_DOMESTIC_SEHIR_IDS = {
    1,   # Adana
    2,   # İzmir
    3,   # İstanbul
    4,   # Ankara
    5,   # Bursa
    6,   # Elazığ
    7,   # Diyarbakır
    8,   # Şanlıurfa
    9,   # Kocaeli
    10,  # Antalya
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_int(text: str | None) -> int | None:
    """Parse an integer from text, returning None on failure."""
    if not text:
        return None
    text = text.strip()
    # Remove non-numeric suffixes like "DS" in gate numbers ("7DS")
    cleaned = re.match(r"(\d+)", text)
    if cleaned:
        try:
            return int(cleaned.group(1))
        except ValueError:
            return None
    return None


def _safe_float(text: str | None) -> float | None:
    """Parse a float from text, handling Turkish comma decimals."""
    if not text:
        return None
    text = text.strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _extract_text(td: Tag | None) -> str:
    """Get stripped text from a BeautifulSoup Tag, or empty string."""
    if td is None:
        return ""
    return td.get_text(strip=True)


def _extract_link_text(td: Tag | None) -> str:
    """Extract text from the first <a> link in a cell, ignoring tooltip sups.

    Many cells (jockey, owner, trainer) have the clean text in an <a> link
    followed by <sup> elements containing tooltip text like "APApranti".
    Falls back to full cell text if no link is found.
    """
    if td is None:
        return ""
    link = td.select_one("a")
    if link:
        return link.get_text(strip=True)
    return td.get_text(strip=True)


def _extract_horse_name_program(td: Tag | None) -> str:
    """Extract just the horse name from a program name cell.

    The cell contains the name in an <a> link, followed by <sup> tooltip
    elements for equipment codes (KG, DB, SK, etc.) that should be excluded.
    """
    if td is None:
        return ""
    link = td.select_one("a")
    if link:
        return link.get_text(strip=True)
    return td.get_text(strip=True)


def _extract_horse_name_results(td: Tag | None) -> str:
    """Extract horse name from a results name cell.

    Results cells contain the name followed by (gate_number) in the link text,
    e.g. "FORTHCOMING QUEEN(3)". We strip the trailing gate number.
    """
    if td is None:
        return ""
    link = td.select_one("a")
    raw = link.get_text(strip=True) if link else td.get_text(strip=True)
    # Remove trailing (N) gate number
    cleaned = re.sub(r"\(\d+\)\s*$", "", raw).strip()
    return cleaned


def _extract_eid(td: Tag | None) -> str | None:
    """Extract EID (best time) value from a program EID cell.

    The cell wraps the time in a tooltip div/span. The raw text
    includes a long tooltip description appended after the time.
    We extract just the time portion (e.g. "1.51.55").
    """
    if td is None:
        return None
    span = td.select_one("span#aciklamaFancyDrc")
    if span:
        text = span.get_text(strip=True)
    else:
        text = td.get_text(strip=True)
    if not text:
        return None
    # The time is at the start, followed by description text
    m = re.match(r"([\d.]+)", text)
    return m.group(1) if m else None


def _extract_agf(td: Tag | None) -> float | None:
    """Extract AGF percentage from a cell like '%17(2)' or '-'."""
    text = _extract_text(td)
    if not text or text == "-":
        return None
    m = re.search(r"%(\d+(?:[.,]\d+)?)", text)
    if m:
        return _safe_float(m.group(1))
    return None


def _parse_age(text: str) -> int | None:
    """Parse age from TJK age string like '4y a  a' or '3y d  d'.

    Format: '{age}y {sex_code}  {breed_code}'
    """
    m = re.match(r"(\d+)y", text.strip())
    return int(m.group(1)) if m else None


def _parse_race_number(text: str) -> int | None:
    """Extract race number from text like '1. Kosu:14.00'."""
    m = re.search(r"(\d+)\.\s*Koşu", text)
    if not m:
        m = re.search(r"(\d+)\.", text)
    return int(m.group(1)) if m else None


def _parse_race_config(h3: Tag | None) -> dict:
    """Parse the h3.race-config element into structured fields.

    Example text: 'Maiden/DHOW, 4 Yasli Araplar, 58 kg, 1400 Kum, E.I.D. :1.34.68'

    Returns dict with keys: race_type, horse_type, weight_rule, distance_meters, surface.
    """
    result: dict = {
        "race_type": None,
        "horse_type": None,
        "weight_rule": None,
        "distance_meters": None,
        "surface": None,
    }
    if h3 is None:
        return result

    text = h3.get_text(strip=True)
    if not text:
        return result

    # Race type from the first <a> link
    race_type_a = h3.select_one("a.aciklamaFancy")
    if race_type_a:
        result["race_type"] = race_type_a.get_text(strip=True)

    # Distance and surface: look for pattern like "1400\nKum" or "1400 Kum"
    m = re.search(r"(\d{3,4})\s*(Kum|Çim|Sentetik)", text)
    if m:
        result["distance_meters"] = int(m.group(1))
        result["surface"] = m.group(2)

    # Horse type: typically the second comma-separated part
    # e.g. "4 Yaşlı Araplar" or "3 Yaşlı İngilizler"
    parts = [p.strip() for p in text.split(",")]
    if len(parts) >= 2:
        result["horse_type"] = parts[1].strip()

    # Weight rule: look for "XX kg" pattern
    wm = re.search(r"(\d+)\s*kg", text)
    if wm:
        result["weight_rule"] = f"{wm.group(1)} kg"

    return result


def _format_date(d: date) -> str:
    """Format a date as DD/MM/YYYY for TJK query parameters."""
    return d.strftime("%d/%m/%Y")


# ---------------------------------------------------------------------------
# TJKClient
# ---------------------------------------------------------------------------


class TJKClient:
    """Async HTTP client for the Turkish Jockey Club website.

    Usage::

        async with TJKClient() as client:
            cards = await client.get_race_card(date.today())
    """

    def __init__(
        self,
        base_url: str = "https://www.tjk.org",
        delay: float = 2.0,
        max_retries: int | None = None,
        backoff_base: float | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        # Read module defaults at construction time so tests can monkeypatch.
        self.max_retries = (
            max_retries if max_retries is not None else _DEFAULT_MAX_RETRIES
        )
        self.backoff_base = (
            backoff_base if backoff_base is not None else _DEFAULT_BACKOFF_BASE
        )
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
            timeout=30.0,
        )

    async def _retry(
        self,
        operation: Callable[[], Awaitable[T]],
        label: str,
    ) -> T | None:
        """Instance-bound retry wrapper honouring client-level retry config."""
        return await _with_retry(
            operation,
            label,
            max_retries=self.max_retries,
            backoff_base=self.backoff_base,
        )

    async def __aenter__(self) -> TJKClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_race_card(self, race_date: date) -> list[RawRaceCard]:
        """Fetch race program for *race_date*. Returns list of RawRaceCard."""
        cards, _failures = await self._fetch_races(
            page_url=_PROGRAM_PAGE,
            city_url=_PROGRAM_CITY,
            race_date=race_date,
            is_results=False,
        )
        return cards

    async def get_race_card_with_failures(
        self, race_date: date,
    ) -> tuple[list[RawRaceCard], list[str]]:
        """Fetch race program plus the list of track names that failed.

        Allows callers (e.g. BackfillManager) to know whether a date was
        scraped *completely* or only partially so they can retry failed
        tracks rather than silently marking the date done.
        """
        return await self._fetch_races(
            page_url=_PROGRAM_PAGE,
            city_url=_PROGRAM_CITY,
            race_date=race_date,
            is_results=False,
        )

    async def get_race_results(self, race_date: date) -> list[RawRaceCard]:
        """Fetch race results for *race_date*. Returns list of RawRaceCard
        with finish_position and finish_time populated on each horse."""
        cards, _failures = await self._fetch_races(
            page_url=_RESULTS_PAGE,
            city_url=_RESULTS_CITY,
            race_date=race_date,
            is_results=True,
        )
        return cards

    async def fetch_historical_results(
        self,
        from_date: date,
        to_date: date,
    ) -> list[RawRaceCard]:
        """Fetch historical race results via the KosuSorgulama query endpoint.

        Returns results grouped into RawRaceCard objects (one per race).
        Handles pagination (50 results per page).  Each result row contains
        only the winning horse of that race.

        Parameters
        ----------
        from_date:
            Earliest date (inclusive), DD/MM/YYYY sent to TJK.
        to_date:
            Latest date (inclusive).
        """
        date_fmt = "%d/%m/%Y"
        from_str = from_date.strftime(date_fmt)
        to_str = to_date.strftime(date_fmt)

        all_rows: list[dict] = []

        # --- Page 1 (uses /Query/Data/ endpoint) ---
        async def _fetch_page1() -> httpx.Response:
            r = await self._client.post(
                _QUERY_DATA,
                data={
                    "QueryParameter_Tarih": from_str,
                    "QueryParameter_Tarih_Start": from_str,
                    "QueryParameter_Tarih_End": to_str,
                    "PageNumber": "1",
                },
            )
            r.raise_for_status()
            return r

        resp = await self._retry(_fetch_page1, "historical-query page 1")
        if resp is None:
            return []

        rows, has_more = self._parse_query_page(resp.text)
        all_rows.extend(rows)
        page = 2

        # --- Subsequent pages (uses /Query/DataRows/ endpoint) ---
        while has_more and page <= _MAX_HISTORICAL_PAGES:
            if self.delay > 0:
                await asyncio.sleep(self.delay)

            current_page = page  # bind for closure

            async def _fetch_pageN() -> httpx.Response:
                r = await self._client.post(
                    _QUERY_DATA_ROWS,
                    data={
                        "QueryParameter_Tarih_Start": from_str,
                        "QueryParameter_Tarih_End": to_str,
                        "PageNumber": str(current_page),
                        "Sort": "Tarih desc, Sehir asc, KosuSirasi asc",
                    },
                )
                r.raise_for_status()
                return r

            resp = await self._retry(
                _fetch_pageN, f"historical-query page {current_page}",
            )
            if resp is None:
                break

            rows, has_more = self._parse_query_page(resp.text)
            if not rows:
                break
            all_rows.extend(rows)
            page += 1

        if page > _MAX_HISTORICAL_PAGES:
            logger.warning(
                "Hit max-pages guard (%d) while fetching %s -> %s; truncating",
                _MAX_HISTORICAL_PAGES, from_date, to_date,
            )

        logger.info(
            "Fetched %d historical results for %s -> %s (%d pages)",
            len(all_rows),
            from_date,
            to_date,
            page - 1,
        )

        return self._group_query_rows(all_rows)

    # ------------------------------------------------------------------
    # Internal — historical query helpers
    # ------------------------------------------------------------------

    def _parse_query_page(self, html: str) -> tuple[list[dict], bool]:
        """Parse one page of KosuSorgulama results.

        Returns (list_of_row_dicts, has_more_pages).
        """
        soup = BeautifulSoup(html, "html.parser")
        data_rows = [
            row
            for row in soup.select("tr")
            if row.select_one(_Q_DATE) and "hidable" not in row.get("class", [])
        ]

        results: list[dict] = []
        for row in data_rows:
            date_text = _extract_text(row.select_one(_Q_DATE))
            try:
                race_date = datetime.strptime(date_text, "%d.%m.%Y").date()
            except (ValueError, TypeError):
                continue

            race_num = _safe_int(_extract_text(row.select_one(_Q_RACE_NUM)))
            if race_num is None:
                continue

            age_text = _extract_text(row.select_one(_Q_AGE))

            results.append({
                "date": race_date,
                "city": _extract_text(row.select_one(_Q_CITY)),
                "race_number": race_num,
                "group": _extract_text(row.select_one(_Q_GROUP)) or None,
                "race_type": _extract_text(row.select_one(_Q_RACE_TYPE)) or None,
                "distance": _safe_int(_extract_text(row.select_one(_Q_DISTANCE))),
                "surface": _extract_text(row.select_one(_Q_SURFACE)) or None,
                "weight": _extract_text(row.select_one(_Q_WEIGHT)) or None,
                "origin": _extract_text(row.select_one(_Q_ORIGIN)) or None,
                "prize": _extract_text(row.select_one(_Q_PRIZE)) or None,
                "winner_name": _extract_text(row.select_one(_Q_WINNER)) or None,
                "winner_age": _parse_age(age_text),
                "finish_time": _extract_text(row.select_one(_Q_TIME)) or None,
                "hp": _safe_float(_extract_text(row.select_one(_Q_HP))),
            })

        # Determine if there are more pages by looking for the pager form
        has_more = soup.select_one("form.pagerForm") is not None
        return results, has_more

    @staticmethod
    def _group_query_rows(rows: list[dict]) -> list[RawRaceCard]:
        """Group flat query result rows into RawRaceCard objects.

        Each row represents one race with its winner.  Rows are grouped
        by (date, city, race_number).
        """
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            key = (row["date"], row["city"], row["race_number"])
            grouped[key].append(row)

        cards: list[RawRaceCard] = []
        for (race_date, city, race_number), group in sorted(grouped.items()):
            # Use the first row for race-level metadata
            first = group[0]
            horses: list[RawHorseEntry] = []
            for r in group:
                if r["winner_name"]:
                    horses.append(
                        RawHorseEntry(
                            name=r["winner_name"],
                            age=r["winner_age"],
                            origin=r["origin"],
                            hp=r["hp"],
                            finish_position=1,
                            finish_time=r["finish_time"],
                        )
                    )

            cards.append(
                RawRaceCard(
                    track_name=city,
                    date=race_date,
                    race_number=race_number,
                    distance_meters=first["distance"],
                    surface=first["surface"],
                    race_type=first["race_type"],
                    horse_type=first["group"],
                    weight_rule=f"{first['weight']} kg" if first["weight"] else None,
                    horses=horses,
                )
            )

        return cards

    # ------------------------------------------------------------------
    # Internal — daily race card helpers
    # ------------------------------------------------------------------

    async def _fetch_races(
        self,
        page_url: str,
        city_url: str,
        race_date: date,
        is_results: bool,
    ) -> tuple[list[RawRaceCard], list[str]]:
        """Fetch the main page to discover tracks, then fetch each track.

        Returns ``(cards, failed_track_names)``.  ``failed_track_names`` is
        a list of tracks that did not return any cards due to HTTP errors
        or empty responses — callers can use it to log partial-scrape state
        and retry later.
        """
        date_str = _format_date(race_date)

        async def _fetch_main() -> httpx.Response:
            r = await self._client.get(
                page_url, params={"QueryParameter_Tarih": date_str},
            )
            r.raise_for_status()
            return r

        resp = await self._retry(_fetch_main, f"main-page {page_url}")
        if resp is None:
            return [], []

        soup = BeautifulSoup(resp.text, "html.parser")
        tabs = soup.select(_SEL_TRACK_TABS)
        if not tabs:
            logger.warning("No track tabs found on %s for %s", page_url, date_str)
            return [], []

        # Collect Turkish domestic tracks (filter out international tracks)
        domestic_tracks = []
        for tab in tabs:
            href = tab.get("href", "")
            sehir_id = tab.get("data-sehir-id", "")
            text = tab.get_text(strip=True)
            # Extract track name from tab text, removing the "(N. Y.G.)" suffix
            track_name = re.sub(r"\s*\(\d+\.\s*Y\.G\.\)\s*$", "", text).strip()
            # Only include known Turkish domestic tracks
            try:
                sid = int(sehir_id)
            except (ValueError, TypeError):
                continue
            if sid not in _DOMESTIC_SEHIR_IDS:
                continue
            domestic_tracks.append((track_name, sehir_id, href))

        all_cards: list[RawRaceCard] = []
        failed_tracks: list[str] = []
        for track_name, sehir_id, _href in domestic_tracks:
            if self.delay > 0:
                await asyncio.sleep(self.delay)

            cards = await self._fetch_city_races(
                city_url=city_url,
                sehir_id=sehir_id,
                track_name=track_name,
                race_date=race_date,
                is_results=is_results,
            )
            if not cards:
                failed_tracks.append(track_name)
            all_cards.extend(cards)

        return all_cards, failed_tracks

    async def _fetch_city_races(
        self,
        city_url: str,
        sehir_id: str,
        track_name: str,
        race_date: date,
        is_results: bool,
    ) -> list[RawRaceCard]:
        """Fetch and parse races for a single track/city."""
        date_str = _format_date(race_date)

        async def _fetch_city() -> httpx.Response:
            r = await self._client.get(
                city_url,
                params={
                    "SehirId": sehir_id,
                    "QueryParameter_Tarih": date_str,
                    "SehirAdi": track_name,
                },
            )
            r.raise_for_status()
            return r

        resp = await self._retry(
            _fetch_city, f"city {track_name} (SehirId={sehir_id})",
        )
        if resp is None:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        return self._parse_city_html(soup, track_name, race_date, is_results)

    def _parse_city_html(
        self,
        soup: BeautifulSoup,
        track_name: str,
        race_date: date,
        is_results: bool,
    ) -> list[RawRaceCard]:
        """Parse the city-level HTML fragment into RawRaceCard list."""
        race_details = soup.select("div.race-details")
        tables = soup.select(_SEL_HORSE_TABLE)

        if len(race_details) != len(tables):
            logger.warning(
                "Mismatch: %d race-details vs %d tables for %s",
                len(race_details),
                len(tables),
                track_name,
            )

        cards: list[RawRaceCard] = []
        for idx, detail_div in enumerate(race_details):
            if idx >= len(tables):
                break

            # --- Race header ---
            race_no_h3 = detail_div.select_one(_SEL_RACE_NO)
            race_no_text = _extract_text(race_no_h3)
            race_number = _parse_race_number(race_no_text)
            if race_number is None:
                continue

            race_config_h3 = detail_div.select_one(_SEL_RACE_CONFIG)
            config = _parse_race_config(race_config_h3)

            # --- Horse rows ---
            table = tables[idx]
            rows = table.select("tbody tr")
            horses: list[RawHorseEntry] = []

            for row in rows:
                horse = self._parse_horse_row(row, is_results)
                if horse and horse.name:
                    horses.append(horse)

            card = RawRaceCard(
                track_name=track_name,
                date=race_date,
                race_number=race_number,
                distance_meters=config["distance_meters"],
                surface=config["surface"],
                race_type=config["race_type"],
                horse_type=config["horse_type"],
                weight_rule=config["weight_rule"],
                horses=horses,
            )
            cards.append(card)

        return cards

    def _parse_horse_row(self, row: Tag, is_results: bool) -> RawHorseEntry | None:
        """Parse a single <tr> into a RawHorseEntry."""
        if is_results:
            return self._parse_result_row(row)
        return self._parse_program_row(row)

    def _parse_program_row(self, row: Tag) -> RawHorseEntry | None:
        """Parse a horse row from the race program table."""
        name = _extract_horse_name_program(row.select_one(_P_NAME))
        if not name:
            return None

        age_text = _extract_text(row.select_one(_P_AGE))
        eid_text = _extract_eid(row.select_one(_P_EID))

        return RawHorseEntry(
            name=name,
            age=_parse_age(age_text),
            origin=_extract_text(row.select_one(_P_ORIGIN)) or None,
            owner=_extract_link_text(row.select_one(_P_OWNER)) or None,
            trainer=_extract_link_text(row.select_one(_P_TRAINER)) or None,
            gate_number=_safe_int(_extract_text(row.select_one(_P_GATE))),
            jockey=_extract_link_text(row.select_one(_P_JOCKEY)) or None,
            weight_kg=_safe_float(_extract_text(row.select_one(_P_WEIGHT))),
            hp=_safe_float(_extract_text(row.select_one(_P_HP))),
            kgs=_safe_int(_extract_text(row.select_one(_P_KGS))),
            s20=_safe_float(_extract_text(row.select_one(_P_S20))),
            eid=eid_text,
            gny=_safe_float(_extract_text(row.select_one(_P_GNY))),
            agf=_extract_agf(row.select_one(_P_AGF)),
            last_six=_extract_text(row.select_one(_P_LAST6)) or None,
        )

    def _parse_result_row(self, row: Tag) -> RawHorseEntry | None:
        """Parse a horse row from the race results table."""
        name = _extract_horse_name_results(row.select_one(_R_NAME))
        if not name:
            return None

        age_text = _extract_text(row.select_one(_R_AGE))

        return RawHorseEntry(
            name=name,
            age=_parse_age(age_text),
            origin=_extract_text(row.select_one(_R_ORIGIN)) or None,
            owner=_extract_link_text(row.select_one(_R_OWNER)) or None,
            trainer=_extract_link_text(row.select_one(_R_TRAINER)) or None,
            gate_number=_safe_int(_extract_text(row.select_one(_R_GATE))),
            jockey=_extract_link_text(row.select_one(_R_JOCKEY)) or None,
            weight_kg=_safe_float(_extract_text(row.select_one(_R_WEIGHT))),
            hp=_safe_float(_extract_text(row.select_one(_R_HP))),
            gny=_safe_float(_extract_text(row.select_one(_R_GNY))),
            agf=_extract_agf(row.select_one(_R_AGF)),
            finish_position=_safe_int(_extract_text(row.select_one(_R_FINISH))),
            finish_time=_extract_text(row.select_one(_R_TIME)) or None,
        )
