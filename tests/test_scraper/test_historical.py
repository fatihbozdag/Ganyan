"""Tests for TJK historical race query (KosuSorgulama) fetcher."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Horse, Race, RaceEntry, RaceStatus, ScrapeLog, ScrapeStatus, Track
from ganyan.scraper.parser import parse_race_card
from ganyan.scraper.backfill import BackfillManager, store_historical_race
from ganyan.scraper.tjk_api import TJKClient

# ---------------------------------------------------------------------------
# Realistic HTML fixtures — derived from live TJK KosuSorgulama (2026-04)
# ---------------------------------------------------------------------------

# Page 1: returned by POST /TR/YarisSever/Query/Data/KosuSorgulama
QUERY_PAGE_1_HTML = """
<div class="clear">&nbsp;</div>
<div class="table-loading" id="table-loading">Sorgu Tamamlaniyor</div>
<div class="clear">&nbsp;</div>
<div id="content" style='margin-top:10px;margin-left:10px;'></div>
<script>
    function updateQueryStringParameter(uri, key, value) {
        return uri;
    }
</script>
<div class="program" id="dataDiv">
    <table id="queryTable" summary="Kosular" class="tablesorter">
        <thead>
            <tr>
                <th scope="col"><a name="Tarih">Tarih</a></th>
                <th scope="col"><a name="Sehir">Sehir</a></th>
                <th scope="col"><a name="KosuSirasi">Kosu</a></th>
                <th scope="col"><a name="KosuGrubuAdi">Grup</a></th>
                <th scope="col"><a name="KosuCinsiAdi">Kosu Cinsi</a></th>
                <th scope="col"><a name="AprantiKosuTipi">Apr. Kos. Cinsi</a></th>
                <th scope="col"><a name="Mesafe">Mesafe</a></th>
                <th scope="col"><a name="PistAdi">Pist</a></th>
                <th scope="col"><a name="Kilo">Siklet</a></th>
                <th scope="col"><a name="BabaAnne">Orijin (Baba-Anne)</a></th>
                <th scope="col"><a name="IKRAMIYE">Ikramiye</a></th>
                <th scope="col"><a name="BirinciAtAdi">Birinci</a></th>
                <th scope="col"><a name="BirinciAtAdiYas">Yas</a></th>
                <th scope="col"><a name="BirinciAtDerece">Derece</a></th>
                <th scope="col"><a name="HandikapPuani">H. Puani</a></th>
            </tr>
        </thead>
        <tbody id="tbody0" class="ajaxtbody">
        <tr class="even">
            <td class="sorgu-KosuSorgulama-Tarih" style="">
                <a target="_blank" href="../Page/../../Info/Page/GunlukYarisSonuclari?QueryParameter_Tarih=02%2f03%2f2026&amp;1=#222901">02.03.2026</a>
            </td>
            <td class="sorgu-KosuSorgulama-Sehir" style="">Adana</td>
            <td class="sorgu-KosuSorgulama-KosuSirasi" style="">1</td>
            <td class="sorgu-KosuSorgulama-KosuGrubuAdi" style="">3 Yasli Ingilizler</td>
            <td class="sorgu-KosuSorgulama-KosuCinsiAdi" style="">SARTLI 1</td>
            <td class="sorgu-KosuSorgulama-AprantiKosuTipi" style=""></td>
            <td class="sorgu-KosuSorgulama-Mesafe" style="">1400</td>
            <td class="sorgu-KosuSorgulama-PistAdi" style="">Kum</td>
            <td class="sorgu-KosuSorgulama-Kilo" style="">58</td>
            <td class="sorgu-KosuSorgulama-BabaAnne" style="">TOUCH THE WOLF - LERZAN</td>
            <td class="sorgu-KosuSorgulama-IKRAMIYE" style="">685.000</td>
            <td class="sorgu-KosuSorgulama-BirinciAtAdi" style="">GIRALAMO</td>
            <td class="sorgu-KosuSorgulama-BirinciAtAdiYas" style="">3y d  e</td>
            <td class="sorgu-KosuSorgulama-BirinciAtDerece" style="">1.31.18</td>
            <td class="sorgu-KosuSorgulama-HandikapPuani" style=""></td>
        </tr>
        <tr class="odd">
            <td class="sorgu-KosuSorgulama-Tarih" style="">
                <a target="_blank" href="#">02.03.2026</a>
            </td>
            <td class="sorgu-KosuSorgulama-Sehir" style="">Adana</td>
            <td class="sorgu-KosuSorgulama-KosuSirasi" style="">2</td>
            <td class="sorgu-KosuSorgulama-KosuGrubuAdi" style="">4 Yasli Araplar</td>
            <td class="sorgu-KosuSorgulama-KosuCinsiAdi" style="">Handikap 17 /H1</td>
            <td class="sorgu-KosuSorgulama-AprantiKosuTipi" style=""></td>
            <td class="sorgu-KosuSorgulama-Mesafe" style="">1400</td>
            <td class="sorgu-KosuSorgulama-PistAdi" style="">Kum</td>
            <td class="sorgu-KosuSorgulama-Kilo" style="">58,5</td>
            <td class="sorgu-KosuSorgulama-BabaAnne" style="">TURBO - NESESEL</td>
            <td class="sorgu-KosuSorgulama-IKRAMIYE" style="">680.000</td>
            <td class="sorgu-KosuSorgulama-BirinciAtAdi" style="">OZGUR RUH</td>
            <td class="sorgu-KosuSorgulama-BirinciAtAdiYas" style="">5y k  a</td>
            <td class="sorgu-KosuSorgulama-BirinciAtDerece" style="">1.37.59</td>
            <td class="sorgu-KosuSorgulama-HandikapPuani" style="">85</td>
        </tr>
        <tr class="even">
            <td class="sorgu-KosuSorgulama-Tarih" style="">
                <a target="_blank" href="#">01.03.2026</a>
            </td>
            <td class="sorgu-KosuSorgulama-Sehir" style="">Istanbul</td>
            <td class="sorgu-KosuSorgulama-KosuSirasi" style="">1</td>
            <td class="sorgu-KosuSorgulama-KosuGrubuAdi" style="">3 Yasli Ingilizler</td>
            <td class="sorgu-KosuSorgulama-KosuCinsiAdi" style="">Maiden /Disi</td>
            <td class="sorgu-KosuSorgulama-AprantiKosuTipi" style=""></td>
            <td class="sorgu-KosuSorgulama-Mesafe" style="">1300</td>
            <td class="sorgu-KosuSorgulama-PistAdi" style="">Cim</td>
            <td class="sorgu-KosuSorgulama-Kilo" style="">58</td>
            <td class="sorgu-KosuSorgulama-BabaAnne" style="">KLIMT (USA) - WENONA</td>
            <td class="sorgu-KosuSorgulama-IKRAMIYE" style="">545.000</td>
            <td class="sorgu-KosuSorgulama-BirinciAtAdi" style="">CHARLIE THE FIRST</td>
            <td class="sorgu-KosuSorgulama-BirinciAtAdiYas" style="">3y d  d</td>
            <td class="sorgu-KosuSorgulama-BirinciAtDerece" style="">1.21.67</td>
            <td class="sorgu-KosuSorgulama-HandikapPuani" style="">31</td>
        </tr>
        <tr class="hidable">
            <td colspan="15">
                <div>
                    <form class="pagerForm" action="/TR/YarisSever/Query/DataRows/KosuSorgulama"
                          method="post">
                        <input name="QueryParameter_Tarih_Start" type="hidden" value="" />
                        <input name="QueryParameter_Tarih_End" type="hidden" value="02/03/2026" />
                        <input name="PageNumber" type="hidden" value="1" />
                        <input name="Sort" type="hidden" value="Tarih desc, Sehir asc, KosuSirasi asc" />
                        <button type="submit" class="show-more">Daha Fazla Sonuc Goster</button>
                    </form>
                    <div>Toplam 5 sonuctan 3 tanesi gosteriliyor</div>
                    <a href="#">Basa Don</a>
                </div>
            </td>
        </tr>
        </tbody>
    </table>
</div>
"""

# Page 2: returned by POST /TR/YarisSever/Query/DataRows/KosuSorgulama
QUERY_PAGE_2_HTML = """
<tbody id="tbody2" class="ajaxtbody">
    <tr class="even">
        <td class="sorgu-KosuSorgulama-Tarih" style="">
            <a target="_blank" href="#">01.03.2026</a>
        </td>
        <td class="sorgu-KosuSorgulama-Sehir" style="">Istanbul</td>
        <td class="sorgu-KosuSorgulama-KosuSirasi" style="">2</td>
        <td class="sorgu-KosuSorgulama-KosuGrubuAdi" style="">4 ve Yukari Ingilizler</td>
        <td class="sorgu-KosuSorgulama-KosuCinsiAdi" style="">Handikap 16 /H1</td>
        <td class="sorgu-KosuSorgulama-AprantiKosuTipi" style=""></td>
        <td class="sorgu-KosuSorgulama-Mesafe" style="">1900</td>
        <td class="sorgu-KosuSorgulama-PistAdi" style="">Cim</td>
        <td class="sorgu-KosuSorgulama-Kilo" style="">56</td>
        <td class="sorgu-KosuSorgulama-BabaAnne" style="">PERFECT STORM - LUNA</td>
        <td class="sorgu-KosuSorgulama-IKRAMIYE" style="">900.000</td>
        <td class="sorgu-KosuSorgulama-BirinciAtAdi" style="">STORM CHASER</td>
        <td class="sorgu-KosuSorgulama-BirinciAtAdiYas" style="">4y d  d</td>
        <td class="sorgu-KosuSorgulama-BirinciAtDerece" style="">2.00.15</td>
        <td class="sorgu-KosuSorgulama-HandikapPuani" style="">72</td>
    </tr>
    <tr class="odd">
        <td class="sorgu-KosuSorgulama-Tarih" style="">
            <a target="_blank" href="#">01.03.2026</a>
        </td>
        <td class="sorgu-KosuSorgulama-Sehir" style="">Istanbul</td>
        <td class="sorgu-KosuSorgulama-KosuSirasi" style="">3</td>
        <td class="sorgu-KosuSorgulama-KosuGrubuAdi" style="">3 Yasli Araplar</td>
        <td class="sorgu-KosuSorgulama-KosuCinsiAdi" style="">SARTLI 4</td>
        <td class="sorgu-KosuSorgulama-AprantiKosuTipi" style=""></td>
        <td class="sorgu-KosuSorgulama-Mesafe" style="">1200</td>
        <td class="sorgu-KosuSorgulama-PistAdi" style="">Kum</td>
        <td class="sorgu-KosuSorgulama-Kilo" style="">54</td>
        <td class="sorgu-KosuSorgulama-BabaAnne" style="">DILIRAN - AZERIN</td>
        <td class="sorgu-KosuSorgulama-IKRAMIYE" style="">490.000</td>
        <td class="sorgu-KosuSorgulama-BirinciAtAdi" style="">YILDIZ AT</td>
        <td class="sorgu-KosuSorgulama-BirinciAtAdiYas" style="">3y a  a</td>
        <td class="sorgu-KosuSorgulama-BirinciAtDerece" style="">1.14.50</td>
        <td class="sorgu-KosuSorgulama-HandikapPuani" style="">45</td>
    </tr>
    <tr class="hidable">
        <td colspan="15">
            <div>
                <div>Toplam 5 sonuctan 5 tanesi gosteriliyor</div>
                <a href="#">Basa Don</a>
            </div>
        </td>
    </tr>
</tbody>
"""

# Empty query result (no races in date range)
QUERY_EMPTY_HTML = """
<div class="program" id="dataDiv">
    <table id="queryTable" summary="Kosular" class="tablesorter">
        <thead>
            <tr>
                <th scope="col"><a name="Tarih">Tarih</a></th>
            </tr>
        </thead>
        <tbody id="tbody0" class="ajaxtbody">
        </tbody>
    </table>
</div>
"""

# Page with no "show more" form (last page)
QUERY_LAST_PAGE_HTML = """
<tbody id="tbody2" class="ajaxtbody">
    <tr class="even">
        <td class="sorgu-KosuSorgulama-Tarih" style="">
            <a target="_blank" href="#">15.01.2026</a>
        </td>
        <td class="sorgu-KosuSorgulama-Sehir" style="">Ankara</td>
        <td class="sorgu-KosuSorgulama-KosuSirasi" style="">1</td>
        <td class="sorgu-KosuSorgulama-KosuGrubuAdi" style="">3 Yasli Ingilizler</td>
        <td class="sorgu-KosuSorgulama-KosuCinsiAdi" style="">Maiden</td>
        <td class="sorgu-KosuSorgulama-AprantiKosuTipi" style=""></td>
        <td class="sorgu-KosuSorgulama-Mesafe" style="">1100</td>
        <td class="sorgu-KosuSorgulama-PistAdi" style="">Sentetik</td>
        <td class="sorgu-KosuSorgulama-Kilo" style="">58</td>
        <td class="sorgu-KosuSorgulama-BabaAnne" style="">BOLD - STAR</td>
        <td class="sorgu-KosuSorgulama-IKRAMIYE" style="">400.000</td>
        <td class="sorgu-KosuSorgulama-BirinciAtAdi" style="">BOLD STAR</td>
        <td class="sorgu-KosuSorgulama-BirinciAtAdiYas" style="">3y d  d</td>
        <td class="sorgu-KosuSorgulama-BirinciAtDerece" style="">1.08.25</td>
        <td class="sorgu-KosuSorgulama-HandikapPuani" style="">22</td>
    </tr>
    <tr class="hidable">
        <td colspan="15">
            <div>
                <div>Toplam 1 sonuctan 1 tanesi gosteriliyor</div>
            </div>
        </td>
    </tr>
</tbody>
"""


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Tests — Query page parsing
# ---------------------------------------------------------------------------


class TestParseQueryPage:
    """Tests for TJKClient._parse_query_page."""

    def test_parse_page_1_with_more(self) -> None:
        """Parse first page with 3 data rows and a pager form."""
        client = TJKClient(delay=0)
        rows, has_more = client._parse_query_page(QUERY_PAGE_1_HTML)

        assert len(rows) == 3
        assert has_more is True

        # First row: Adana race 1 on 02.03.2026
        r1 = rows[0]
        assert r1["date"] == date(2026, 3, 2)
        assert r1["city"] == "Adana"
        assert r1["race_number"] == 1
        assert r1["group"] == "3 Yasli Ingilizler"
        assert r1["race_type"] == "SARTLI 1"
        assert r1["distance"] == 1400
        assert r1["surface"] == "Kum"
        assert r1["weight"] == "58"
        assert r1["origin"] == "TOUCH THE WOLF - LERZAN"
        assert r1["winner_name"] == "GIRALAMO"
        assert r1["winner_age"] == 3
        assert r1["finish_time"] == "1.31.18"
        assert r1["hp"] is None  # Empty HP field

        # Second row: Adana race 2
        r2 = rows[1]
        assert r2["city"] == "Adana"
        assert r2["race_number"] == 2
        assert r2["race_type"] == "Handikap 17 /H1"
        assert r2["weight"] == "58,5"
        assert r2["winner_name"] == "OZGUR RUH"
        assert r2["winner_age"] == 5
        assert r2["hp"] == 85.0

        # Third row: Istanbul race 1
        r3 = rows[2]
        assert r3["date"] == date(2026, 3, 1)
        assert r3["city"] == "Istanbul"
        assert r3["race_number"] == 1
        assert r3["distance"] == 1300
        assert r3["surface"] == "Cim"
        assert r3["winner_name"] == "CHARLIE THE FIRST"
        assert r3["hp"] == 31.0

    def test_parse_page_2_last_page(self) -> None:
        """Parse second page: no pager form means no more pages."""
        client = TJKClient(delay=0)
        rows, has_more = client._parse_query_page(QUERY_PAGE_2_HTML)

        assert len(rows) == 2
        # No form.pagerForm present -> last page
        assert has_more is False

        r1 = rows[0]
        assert r1["city"] == "Istanbul"
        assert r1["race_number"] == 2
        assert r1["distance"] == 1900
        assert r1["winner_name"] == "STORM CHASER"

    def test_parse_empty_page(self) -> None:
        """Parse empty query result."""
        client = TJKClient(delay=0)
        rows, has_more = client._parse_query_page(QUERY_EMPTY_HTML)

        assert len(rows) == 0
        assert has_more is False

    def test_parse_single_result_last_page(self) -> None:
        """Parse a page with one result and no pagination form."""
        client = TJKClient(delay=0)
        rows, has_more = client._parse_query_page(QUERY_LAST_PAGE_HTML)

        assert len(rows) == 1
        assert has_more is False
        assert rows[0]["city"] == "Ankara"
        assert rows[0]["surface"] == "Sentetik"


# ---------------------------------------------------------------------------
# Tests — Group query rows into RawRaceCards
# ---------------------------------------------------------------------------


class TestGroupQueryRows:
    """Tests for TJKClient._group_query_rows."""

    def test_group_multiple_races(self) -> None:
        """Multiple rows grouped by (date, city, race_number)."""
        rows = [
            {
                "date": date(2026, 3, 2),
                "city": "Adana",
                "race_number": 1,
                "group": "3 Yasli Ingilizler",
                "race_type": "SARTLI 1",
                "distance": 1400,
                "surface": "Kum",
                "weight": "58",
                "origin": "TOUCH THE WOLF - LERZAN",
                "prize": "685.000",
                "winner_name": "GIRALAMO",
                "winner_age": 3,
                "finish_time": "1.31.18",
                "hp": None,
            },
            {
                "date": date(2026, 3, 2),
                "city": "Adana",
                "race_number": 2,
                "group": "4 Yasli Araplar",
                "race_type": "Handikap 17",
                "distance": 1400,
                "surface": "Kum",
                "weight": "58,5",
                "origin": "TURBO - NESESEL",
                "prize": "680.000",
                "winner_name": "OZGUR RUH",
                "winner_age": 5,
                "finish_time": "1.37.59",
                "hp": 85.0,
            },
            {
                "date": date(2026, 3, 1),
                "city": "Istanbul",
                "race_number": 1,
                "group": "3 Yasli Ingilizler",
                "race_type": "Maiden",
                "distance": 1300,
                "surface": "Cim",
                "weight": "58",
                "origin": "KLIMT (USA) - WENONA",
                "prize": "545.000",
                "winner_name": "CHARLIE THE FIRST",
                "winner_age": 3,
                "finish_time": "1.21.67",
                "hp": 31.0,
            },
        ]

        cards = TJKClient._group_query_rows(rows)

        # Sorted by (date, city, race_number)
        assert len(cards) == 3

        # First card: Istanbul 2026-03-01 race 1
        c1 = cards[0]
        assert c1.track_name == "Istanbul"
        assert c1.date == date(2026, 3, 1)
        assert c1.race_number == 1
        assert c1.distance_meters == 1300
        assert c1.surface == "Cim"
        assert len(c1.horses) == 1
        assert c1.horses[0].name == "CHARLIE THE FIRST"
        assert c1.horses[0].finish_position == 1
        assert c1.horses[0].finish_time == "1.21.67"

        # Second card: Adana 2026-03-02 race 1
        c2 = cards[1]
        assert c2.track_name == "Adana"
        assert c2.date == date(2026, 3, 2)
        assert c2.race_number == 1

        # Third card: Adana 2026-03-02 race 2
        c3 = cards[2]
        assert c3.race_number == 2
        assert c3.race_type == "Handikap 17"
        assert c3.weight_rule == "58,5 kg"
        assert c3.horses[0].name == "OZGUR RUH"
        assert c3.horses[0].hp == 85.0

    def test_group_empty_list(self) -> None:
        """Empty row list returns empty card list."""
        cards = TJKClient._group_query_rows([])
        assert cards == []


# ---------------------------------------------------------------------------
# Tests — Full fetch_historical_results with pagination
# ---------------------------------------------------------------------------


class TestFetchHistoricalResults:
    """Tests for TJKClient.fetch_historical_results with mocked HTTP."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_single_page(self) -> None:
        """Single-page query (no pagination needed)."""
        base = "https://www.tjk.org"

        respx.post(f"{base}/TR/YarisSever/Query/Data/KosuSorgulama").mock(
            return_value=httpx.Response(200, text=QUERY_LAST_PAGE_HTML)
        )

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.fetch_historical_results(
                date(2026, 1, 15), date(2026, 1, 15)
            )

        assert len(cards) == 1
        assert cards[0].track_name == "Ankara"
        assert cards[0].horses[0].name == "BOLD STAR"

    @respx.mock
    @pytest.mark.asyncio
    async def test_two_pages(self) -> None:
        """Multi-page query fetches page 1 and page 2."""
        base = "https://www.tjk.org"

        # Page 1 has pager form -> has_more=True
        respx.post(f"{base}/TR/YarisSever/Query/Data/KosuSorgulama").mock(
            return_value=httpx.Response(200, text=QUERY_PAGE_1_HTML)
        )

        # Page 2 has no pager form -> has_more=False
        respx.post(f"{base}/TR/YarisSever/Query/DataRows/KosuSorgulama").mock(
            return_value=httpx.Response(200, text=QUERY_PAGE_2_HTML)
        )

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.fetch_historical_results(
                date(2026, 3, 1), date(2026, 3, 2)
            )

        # 3 rows from page 1 + 2 rows from page 2 = 5 total
        # Grouped into distinct races: Adana R1, Adana R2 (Mar 2),
        # Istanbul R1, R2, R3 (Mar 1)
        assert len(cards) == 5

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_result(self) -> None:
        """Empty date range returns empty list."""
        base = "https://www.tjk.org"

        respx.post(f"{base}/TR/YarisSever/Query/Data/KosuSorgulama").mock(
            return_value=httpx.Response(200, text=QUERY_EMPTY_HTML)
        )

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.fetch_historical_results(
                date(2099, 1, 1), date(2099, 1, 1)
            )

        assert cards == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_http_error_returns_empty(self) -> None:
        """HTTP error on page 1 returns empty list."""
        base = "https://www.tjk.org"

        respx.post(f"{base}/TR/YarisSever/Query/Data/KosuSorgulama").mock(
            return_value=httpx.Response(500)
        )

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.fetch_historical_results(
                date(2026, 3, 1), date(2026, 3, 2)
            )

        assert cards == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_page2_error_returns_partial(self) -> None:
        """HTTP error on page 2 returns results from page 1 only."""
        base = "https://www.tjk.org"

        respx.post(f"{base}/TR/YarisSever/Query/Data/KosuSorgulama").mock(
            return_value=httpx.Response(200, text=QUERY_PAGE_1_HTML)
        )
        respx.post(f"{base}/TR/YarisSever/Query/DataRows/KosuSorgulama").mock(
            return_value=httpx.Response(500)
        )

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.fetch_historical_results(
                date(2026, 3, 1), date(2026, 3, 2)
            )

        # Only page 1 results (3 rows -> 3 cards)
        assert len(cards) == 3


# ---------------------------------------------------------------------------
# Tests — store_historical_race
# ---------------------------------------------------------------------------


class TestStoreHistoricalRace:
    """Tests for the store_historical_race function."""

    def test_creates_race_as_resulted(self, db_session) -> None:
        """Historical races should be stored with resulted status."""
        from ganyan.scraper.parser import RawRaceCard, RawHorseEntry, parse_race_card

        raw = RawRaceCard(
            track_name="Adana",
            date=date(2026, 3, 2),
            race_number=1,
            distance_meters=1400,
            surface="Kum",
            race_type="SARTLI 1",
            horses=[
                RawHorseEntry(
                    name="GIRALAMO",
                    age=3,
                    origin="TOUCH THE WOLF - LERZAN",
                    finish_position=1,
                    finish_time="1.31.18",
                ),
            ],
        )
        parsed = parse_race_card(raw)
        race = store_historical_race(db_session, parsed)
        db_session.commit()

        assert race.status == RaceStatus.resulted
        assert db_session.query(Race).count() == 1
        assert db_session.query(Horse).count() == 1

        entry = db_session.query(RaceEntry).first()
        assert entry.finish_position == 1
        assert entry.finish_time == "1.31.18"

    def test_idempotent(self, db_session) -> None:
        """Calling store_historical_race twice should not duplicate records."""
        from ganyan.scraper.parser import RawRaceCard, RawHorseEntry, parse_race_card

        raw = RawRaceCard(
            track_name="Istanbul",
            date=date(2026, 3, 1),
            race_number=1,
            distance_meters=1300,
            surface="Cim",
            race_type="Maiden",
            horses=[
                RawHorseEntry(
                    name="CHARLIE THE FIRST",
                    age=3,
                    finish_position=1,
                    finish_time="1.21.67",
                ),
            ],
        )
        parsed = parse_race_card(raw)
        store_historical_race(db_session, parsed)
        db_session.commit()
        store_historical_race(db_session, parsed)
        db_session.commit()

        assert db_session.query(Race).count() == 1
        assert db_session.query(Horse).count() == 1
        assert db_session.query(RaceEntry).count() == 1

    def test_upgrades_scheduled_to_resulted(self, db_session) -> None:
        """If race exists as scheduled, historical store upgrades to resulted."""
        from ganyan.scraper.parser import RawRaceCard, RawHorseEntry, parse_race_card
        from ganyan.scraper.backfill import store_race_card

        raw_scheduled = RawRaceCard(
            track_name="Ankara",
            date=date(2026, 1, 15),
            race_number=1,
            distance_meters=1100,
            surface="Sentetik",
            race_type="Maiden",
            horses=[
                RawHorseEntry(name="BOLD STAR", age=3),
            ],
        )
        parsed = parse_race_card(raw_scheduled)
        store_race_card(db_session, parsed)
        db_session.commit()

        race = db_session.query(Race).first()
        assert race.status == RaceStatus.scheduled

        # Now store historical result for same race
        raw_historical = RawRaceCard(
            track_name="Ankara",
            date=date(2026, 1, 15),
            race_number=1,
            distance_meters=1100,
            surface="Sentetik",
            race_type="Maiden",
            horses=[
                RawHorseEntry(
                    name="BOLD STAR",
                    age=3,
                    finish_position=1,
                    finish_time="1.08.25",
                ),
            ],
        )
        parsed_hist = parse_race_card(raw_historical)
        store_historical_race(db_session, parsed_hist)
        db_session.commit()

        race = db_session.query(Race).first()
        assert race.status == RaceStatus.resulted

        entry = db_session.query(RaceEntry).first()
        assert entry.finish_position == 1


# ---------------------------------------------------------------------------
# Tests — BackfillManager.backfill_historical date chunking
# ---------------------------------------------------------------------------


class TestBackfillHistoricalChunking:
    """Tests for BackfillManager.backfill_historical date range chunking."""

    @pytest.mark.asyncio
    async def test_single_chunk(self, db_session) -> None:
        """Small date range fits in one chunk."""
        fetch_calls = []

        class FakeTJKClient:
            delay = 0

            async def fetch_historical_results(self, from_date, to_date):
                fetch_calls.append((from_date, to_date))
                return []

        mgr = BackfillManager(db_session, FakeTJKClient())
        count = await mgr.backfill_historical(
            date(2026, 3, 1), date(2026, 3, 5), chunk_days=30,
        )

        assert count == 0
        assert len(fetch_calls) == 1
        assert fetch_calls[0] == (date(2026, 3, 1), date(2026, 3, 5))

    @pytest.mark.asyncio
    async def test_multiple_chunks(self, db_session) -> None:
        """Large date range is split into chunks."""
        fetch_calls = []

        class FakeTJKClient:
            delay = 0

            async def fetch_historical_results(self, from_date, to_date):
                fetch_calls.append((from_date, to_date))
                return []

        mgr = BackfillManager(db_session, FakeTJKClient())
        # 60 days with 30-day chunks -> 2 chunks
        count = await mgr.backfill_historical(
            date(2026, 1, 1), date(2026, 3, 1), chunk_days=30,
        )

        assert count == 0
        # Jan 1-30 (30 days), Jan 31 - Mar 1 (30 days)
        assert len(fetch_calls) == 2
        assert fetch_calls[0] == (date(2026, 1, 1), date(2026, 1, 30))
        assert fetch_calls[1] == (date(2026, 1, 31), date(2026, 3, 1))

    @pytest.mark.asyncio
    async def test_stores_races_and_returns_count(self, db_session) -> None:
        """Historical backfill stores races and returns total count."""
        from ganyan.scraper.parser import RawRaceCard, RawHorseEntry

        class FakeTJKClient:
            delay = 0

            async def fetch_historical_results(self, from_date, to_date):
                return [
                    RawRaceCard(
                        track_name="Adana",
                        date=from_date,
                        race_number=1,
                        distance_meters=1400,
                        surface="Kum",
                        race_type="SARTLI 1",
                        horses=[
                            RawHorseEntry(
                                name="GIRALAMO",
                                age=3,
                                finish_position=1,
                                finish_time="1.31.18",
                            ),
                        ],
                    ),
                ]

        mgr = BackfillManager(db_session, FakeTJKClient())
        count = await mgr.backfill_historical(
            date(2026, 3, 1), date(2026, 3, 2), chunk_days=30,
        )

        assert count == 1
        assert db_session.query(Race).count() == 1
        assert db_session.query(Race).first().status == RaceStatus.resulted

    @pytest.mark.asyncio
    async def test_chunk_failure_continues(self, db_session) -> None:
        """Failed chunk is logged but backfill continues with next chunk."""
        fetch_calls = []
        call_count = 0

        class FakeTJKClient:
            delay = 0

            async def fetch_historical_results(self, from_date, to_date):
                nonlocal call_count
                call_count += 1
                fetch_calls.append((from_date, to_date))
                if call_count == 1:
                    raise RuntimeError("Connection lost")
                return []

        mgr = BackfillManager(db_session, FakeTJKClient())
        count = await mgr.backfill_historical(
            date(2026, 1, 1), date(2026, 2, 15), chunk_days=30,
        )

        assert count == 0
        # First chunk failed, second chunk succeeded
        assert len(fetch_calls) == 2

        # Check that failure was logged
        failed_logs = (
            db_session.query(ScrapeLog)
            .filter(ScrapeLog.status == ScrapeStatus.failed)
            .all()
        )
        assert len(failed_logs) == 1
