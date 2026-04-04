"""Tests for TJK API client with mocked HTTP responses."""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from ganyan.scraper.tjk_api import TJKClient

# ---------------------------------------------------------------------------
# Realistic HTML fixtures — derived from live TJK HTML (2026-04)
# ---------------------------------------------------------------------------

# Main page HTML: contains track tabs that the client uses to discover cities
MAIN_PAGE_HTML = """
<html><body>
<ul class="gunluk-tabs">
  <li><a href="/TR/YarisSever/Info/Sehir/GunlukYarisProgrami?SehirId=1&amp;QueryParameter_Tarih=05%2F04%2F2026&amp;SehirAdi=Adana&amp;Era=tomorrow"
         data-sehir-id="1">Adana  (41. Y.G.)</a></li>
  <li><a href="/TR/YarisSever/Info/Sehir/GunlukYarisProgrami?SehirId=3&amp;QueryParameter_Tarih=05%2F04%2F2026&amp;SehirAdi=Istanbul&amp;Era=tomorrow"
         data-sehir-id="3">İstanbul  (27. Y.G.)</a></li>
</ul>
<div class="gunluk-panes">
  <div class="program">
    <input type="hidden" id="DataHash" value="abc123"/>
  </div>
</div>
</body></html>
"""

# City-level program HTML: one race with two horses
CITY_PROGRAM_HTML = """
<div class="races-panes">
  <div>
    <div class="race-details">
      <h3 class="race-no">
        <a id="anc224092">1.                        Koşu:14.00</a>
      </h3>
      <h3 class="race-config">
        <a class="aciklamaFancy" onclick="BultenAciklama(this)" title="Koşu kazanmamış...">Maiden/DHÖW</a>
        , 4 Yaşlı Araplar,

        58 kg,
                    1400

        Kum,E.İ.D. :1.34.68
      </h3>
    </div>
    <table class="tablesorter">
      <thead>
        <tr>
          <th class="formaHeader">Forma</th>
          <th class="aciklamaFancy">N</th>
          <th>At İsmi</th>
          <th class="aciklamaFancy">Yaş</th>
          <th>Orijin(Baba - Anne)</th>
          <th>Sıklet</th>
          <th>Jokey</th>
          <th>Sahip</th>
          <th>Antrenör</th>
          <th class="aciklamaFancy">St</th>
          <th class="aciklamaFancy">HP</th>
          <th>Son 6 Y.</th>
          <th class="aciklamaFancy">KGS</th>
          <th class="aciklamaFancy">s20</th>
          <th class="aciklamaFancy">En İyi D.</th>
          <th>Gny</th>
          <th class="aciklamaFancy">AGF</th>
          <th class="aciklamaFancy">İdm</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="gunluk-GunlukYarisProgrami-FormaKodu"></td>
          <td class="gunluk-GunlukYarisProgrami-SiraId">1</td>
          <td class="gunluk-GunlukYarisProgrami-AtAdi">
            <a href="../../Query/ConnectedPage/AtKosuBilgileri?QueryParameter_AtId=105109" target="_blank">
              ALTUNÇURA<span title=""></span>
            </a>
            <sup class="tooltipp"><span class="aciklamaFancy" id="aciklamaFancyShrt">KG</span><a class="tooltiptextt">Kapalı gözlük.</a></sup>
          </td>
          <td class="gunluk-GunlukYarisProgrami-Yas">4y a  a</td>
          <td class="gunluk-GunlukYarisProgrami-Baba">AYABAKAN-BEYAZ KELEBEK/BİLGİN</td>
          <td class="gunluk-GunlukYarisProgrami-Kilo">55</td>
          <td class="gunluk-GunlukYarisProgrami-JokeAdi">
            <a href="../../Query/Page/JokeyIstatistikleri?QueryParameter_JokeyId=3100" target="_blank" title="MEHMET ÇELİK">M.ÇELİK</a>
            <sup class="tooltipp" id="Apranti"><span class="aciklamaFancy">AP</span><a class="tooltiptextt">Apranti</a></sup>
          </td>
          <td class="gunluk-GunlukYarisProgrami-SahipAdi"><a href="#">AHMET BABACAN</a></td>
          <td class="gunluk-GunlukYarisProgrami-AntronorAdi"><a href="#">M.TEK</a></td>
          <td class="gunluk-GunlukYarisProgrami-StartId">2</td>
          <td class="gunluk-GunlukYarisProgrami-Hc">7</td>
          <td class="gunluk-GunlukYarisProgrami-Son6Yaris">
            <font color="#996633"><b>6</b></font><font color="#996633"><b>0</b></font><font color="#009900"><b>0</b></font>
          </td>
          <td class="gunluk-GunlukYarisProgrami-KGS">22</td>
          <td class="gunluk-GunlukYarisProgrami-s20">17</td>
          <td class="gunluk-GunlukYarisProgrami-DERECE">
            <div class="tooltipp">
              <span id="aciklamaFancyDrc" style="cursor: help;">1.51.55</span>
              <a class="tooltiptextt" id="tlltptxtDrc">Bu derece Adana'da yapılmıştır.</a>
            </div>
          </td>
          <td class="gunluk-GunlukYarisProgrami-Gny"><span>3,50</span></td>
          <td class="gunluk-GunlukYarisProgrami-AGFORAN"><span>%25(1)</span></td>
          <td class="gunluk-GunlukYarisProgrami-idmanpistiFLG"></td>
        </tr>
        <tr>
          <td class="gunluk-GunlukYarisProgrami-FormaKodu"></td>
          <td class="gunluk-GunlukYarisProgrami-SiraId">2</td>
          <td class="gunluk-GunlukYarisProgrami-AtAdi">
            <a href="../../Query/ConnectedPage/AtKosuBilgileri?QueryParameter_AtId=105110" target="_blank">
              STORM RIDER<span title=""></span>
            </a>
          </td>
          <td class="gunluk-GunlukYarisProgrami-Yas">3y d  d</td>
          <td class="gunluk-GunlukYarisProgrami-Baba">KLIMT (USA)-DAYDAY/DEHERE (USA)</td>
          <td class="gunluk-GunlukYarisProgrami-Kilo">57,5</td>
          <td class="gunluk-GunlukYarisProgrami-JokeAdi"><a href="#">E.ATLAMAZ</a></td>
          <td class="gunluk-GunlukYarisProgrami-SahipAdi"><a href="#">ALİ YILMAZ</a></td>
          <td class="gunluk-GunlukYarisProgrami-AntronorAdi"><a href="#">İ.AKKILIÇ</a></td>
          <td class="gunluk-GunlukYarisProgrami-StartId">5DS</td>
          <td class="gunluk-GunlukYarisProgrami-Hc">62</td>
          <td class="gunluk-GunlukYarisProgrami-Son6Yaris">13-4124</td>
          <td class="gunluk-GunlukYarisProgrami-KGS">33</td>
          <td class="gunluk-GunlukYarisProgrami-s20">18</td>
          <td class="gunluk-GunlukYarisProgrami-DERECE"></td>
          <td class="gunluk-GunlukYarisProgrami-Gny"><span></span></td>
          <td class="gunluk-GunlukYarisProgrami-AGFORAN"><span>-</span></td>
          <td class="gunluk-GunlukYarisProgrami-idmanpistiFLG"></td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
"""

# City-level results HTML: one race with two horses (finish order)
CITY_RESULTS_HTML = """
<div class="races-panes">
  <div>
    <div class="race-details">
      <h3 class="race-no">
        <a id="anc223532">1.                        Koşu:14.00</a>
      </h3>
      <h3 class="race-config">
        <a class="aciklamaFancy">Handikap 15/Dişi /H2</a>
        , 3 Yaşlı İngilizler,


                    1500

        Kum,E.İ.D. :1.31.83
      </h3>
    </div>
    <table class="tablesorter">
      <thead>
        <tr>
          <th class="formaHeader">Forma</th>
          <th>S</th>
          <th>At İsmi</th>
          <th class="aciklamaFancy">Yaş</th>
          <th>Orijin(Baba - Anne)</th>
          <th>Sıklet</th>
          <th>Jokey</th>
          <th>Sahip</th>
          <th>Antrenörü</th>
          <th>Derece</th>
          <th>Gny</th>
          <th class="aciklamaFancy">AGF</th>
          <th class="aciklamaFancy">St</th>
          <th>Fark</th>
          <th class="aciklamaFancy">G. Çık.</th>
          <th class="aciklamaFancy">HP</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td class="gunluk-GunlukYarisSonuclari-FormaKodu"></td>
          <td class="gunluk-GunlukYarisSonuclari-SONUCNO">1</td>
          <td class="gunluk-GunlukYarisSonuclari-AtAdi3">
            <a href="../../Query/ConnectedPage/AtKosuBilgileri?QueryParameter_AtId=105109" target="_blank">
              FORTHCOMING QUEEN(3)
            </a>
            <sup class="tooltipp"><span class="aciklamaFancy" id="aciklamaFancyShrt">KG</span></sup>
          </td>
          <td class="gunluk-GunlukYarisSonuclari-Yas">3y d  d</td>
          <td class="gunluk-GunlukYarisSonuclari-Baba">MENDIP (USA)-EPONA/EAGLE EYED (USA)</td>
          <td class="gunluk-GunlukYarisSonuclari-Kilo">57,5</td>
          <td class="gunluk-GunlukYarisSonuclari-JokeAdi">
            <a href="#">ER.CANKILIC</a>
            <sup class="tooltipp" id="Apranti"><span class="aciklamaFancy">AP</span><a class="tooltiptextt">Apranti</a></sup>
          </td>
          <td class="gunluk-GunlukYarisSonuclari-SahipAdi"><a href="#">TAYRAL TUTUMLU</a></td>
          <td class="gunluk-GunlukYarisSonuclari-AntronorAdi"><a href="#">Ş.AYDEMİR</a></td>
          <td class="gunluk-GunlukYarisSonuclari-Derece">1.36.69</td>
          <td class="gunluk-GunlukYarisSonuclari-Gny">3,40</td>
          <td class="gunluk-GunlukYarisSonuclari-AGFORAN">%17(2)</td>
          <td class="gunluk-GunlukYarisSonuclari-StartId">8</td>
          <td class="gunluk-GunlukYarisSonuclari-Fark">5 Boy</td>
          <td class="gunluk-GunlukYarisSonuclari-GecCikis"></td>
          <td class="gunluk-GunlukYarisSonuclari-Hc">52</td>
        </tr>
        <tr>
          <td class="gunluk-GunlukYarisSonuclari-FormaKodu"></td>
          <td class="gunluk-GunlukYarisSonuclari-SONUCNO">2</td>
          <td class="gunluk-GunlukYarisSonuclari-AtAdi3">
            <a href="../../Query/ConnectedPage/AtKosuBilgileri?QueryParameter_AtId=105110" target="_blank">
              KARDAHA(2)
            </a>
          </td>
          <td class="gunluk-GunlukYarisSonuclari-Yas">3y d  a</td>
          <td class="gunluk-GunlukYarisSonuclari-Baba">ABJAR ACADEMY-FAIRY TALE/MOUNTAIN CAT (USA)</td>
          <td class="gunluk-GunlukYarisSonuclari-Kilo">63</td>
          <td class="gunluk-GunlukYarisSonuclari-JokeAdi"><a href="#">M.AKYAVUZ</a></td>
          <td class="gunluk-GunlukYarisSonuclari-SahipAdi"><a href="#">F.SEDAT DAĞYUDAN</a></td>
          <td class="gunluk-GunlukYarisSonuclari-AntronorAdi"><a href="#">M.KORKMAZ</a></td>
          <td class="gunluk-GunlukYarisSonuclari-Derece">1.37.61</td>
          <td class="gunluk-GunlukYarisSonuclari-Gny">5,10</td>
          <td class="gunluk-GunlukYarisSonuclari-AGFORAN">%12(3)</td>
          <td class="gunluk-GunlukYarisSonuclari-StartId">7</td>
          <td class="gunluk-GunlukYarisSonuclari-Fark">1 Boy</td>
          <td class="gunluk-GunlukYarisSonuclari-GecCikis"></td>
          <td class="gunluk-GunlukYarisSonuclari-Hc">75</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>
"""

# Results main page (same structure, different endpoint)
MAIN_RESULTS_PAGE_HTML = """
<html><body>
<ul class="gunluk-tabs">
  <li><a href="/TR/YarisSever/Info/Sehir/GunlukYarisSonuclari?SehirId=1&amp;QueryParameter_Tarih=04%2F04%2F2026&amp;SehirAdi=Adana&amp;Era=today"
         data-sehir-id="1">Adana  (40. Y.G.)</a></li>
</ul>
<div class="gunluk-panes">
  <div class="program">
    <input type="hidden" id="DataHash" value="xyz789"/>
  </div>
</div>
</body></html>
"""

# Empty page (no races for date)
EMPTY_PAGE_HTML = """
<html><body>
<ul class="gunluk-tabs">
</ul>
<div class="gunluk-panes">
  <div class="program"></div>
</div>
</body></html>
"""


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_date() -> date:
    return date(2026, 4, 5)


@pytest.fixture
def results_date() -> date:
    return date(2026, 4, 4)


# ---------------------------------------------------------------------------
# Tests — Race Program
# ---------------------------------------------------------------------------


class TestGetRaceCard:
    """Tests for TJKClient.get_race_card."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_parse(self, test_date: date) -> None:
        """Parse a race card with two horses from mocked HTML."""
        base = "https://www.tjk.org"

        # Mock the main page (returns track tabs)
        respx.get(
            f"{base}/TR/YarisSever/Info/Page/GunlukYarisProgrami"
        ).mock(return_value=httpx.Response(200, text=MAIN_PAGE_HTML))

        # Mock city pages (Adana and Istanbul)
        respx.get(
            f"{base}/TR/YarisSever/Info/Sehir/GunlukYarisProgrami",
            params__contains={"SehirId": "1"},
        ).mock(return_value=httpx.Response(200, text=CITY_PROGRAM_HTML))

        respx.get(
            f"{base}/TR/YarisSever/Info/Sehir/GunlukYarisProgrami",
            params__contains={"SehirId": "3"},
        ).mock(return_value=httpx.Response(200, text=CITY_PROGRAM_HTML))

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.get_race_card(test_date)

        # Two tracks, each with one race
        assert len(cards) == 2

        card = cards[0]
        assert card.track_name == "Adana"
        assert card.date == test_date
        assert card.race_number == 1
        assert card.distance_meters == 1400
        assert card.surface == "Kum"
        assert card.race_type == "Maiden/DHÖW"
        assert card.horse_type == "4 Yaşlı Araplar"
        assert card.weight_rule == "58 kg"

        # Check horses
        assert len(card.horses) == 2

        h1 = card.horses[0]
        assert h1.name == "ALTUNÇURA"
        assert h1.age == 4
        assert h1.origin == "AYABAKAN-BEYAZ KELEBEK/BİLGİN"
        assert h1.weight_kg == 55.0
        assert h1.jockey == "M.ÇELİK"
        assert h1.owner == "AHMET BABACAN"
        assert h1.trainer == "M.TEK"
        assert h1.gate_number == 2
        assert h1.hp == 7.0
        assert h1.kgs == 22
        assert h1.s20 == 17.0
        assert h1.eid == "1.51.55"
        assert h1.gny == 3.5
        assert h1.agf == 25.0
        assert h1.finish_position is None
        assert h1.finish_time is None

        h2 = card.horses[1]
        assert h2.name == "STORM RIDER"
        assert h2.age == 3
        assert h2.weight_kg == 57.5
        assert h2.gate_number == 5  # "5DS" -> 5
        assert h2.eid is None
        assert h2.gny is None
        assert h2.agf is None

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_response(self, test_date: date) -> None:
        """No track tabs returns empty list."""
        base = "https://www.tjk.org"
        respx.get(
            f"{base}/TR/YarisSever/Info/Page/GunlukYarisProgrami"
        ).mock(return_value=httpx.Response(200, text=EMPTY_PAGE_HTML))

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.get_race_card(test_date)

        assert cards == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_http_error_returns_empty(self, test_date: date) -> None:
        """HTTP 500 returns empty list without raising."""
        base = "https://www.tjk.org"
        respx.get(
            f"{base}/TR/YarisSever/Info/Page/GunlukYarisProgrami"
        ).mock(return_value=httpx.Response(500))

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.get_race_card(test_date)

        assert cards == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_connection_error_returns_empty(self, test_date: date) -> None:
        """Network errors return empty list without raising."""
        base = "https://www.tjk.org"
        respx.get(
            f"{base}/TR/YarisSever/Info/Page/GunlukYarisProgrami"
        ).mock(side_effect=httpx.ConnectError("Connection refused"))

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.get_race_card(test_date)

        assert cards == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_city_error_skipped(self, test_date: date) -> None:
        """If one city fails, others still returned."""
        base = "https://www.tjk.org"

        respx.get(
            f"{base}/TR/YarisSever/Info/Page/GunlukYarisProgrami"
        ).mock(return_value=httpx.Response(200, text=MAIN_PAGE_HTML))

        # Adana fails
        respx.get(
            f"{base}/TR/YarisSever/Info/Sehir/GunlukYarisProgrami",
            params__contains={"SehirId": "1"},
        ).mock(return_value=httpx.Response(500))

        # Istanbul succeeds
        respx.get(
            f"{base}/TR/YarisSever/Info/Sehir/GunlukYarisProgrami",
            params__contains={"SehirId": "3"},
        ).mock(return_value=httpx.Response(200, text=CITY_PROGRAM_HTML))

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.get_race_card(test_date)

        # Only Istanbul's race comes through
        assert len(cards) == 1
        assert cards[0].track_name == "İstanbul"

    @respx.mock
    @pytest.mark.asyncio
    async def test_international_tracks_filtered(self, test_date: date) -> None:
        """International tracks (SehirId > 100) are skipped."""
        base = "https://www.tjk.org"
        intl_html = """
        <html><body>
        <ul class="gunluk-tabs">
          <li><a href="..." data-sehir-id="85">Gulfstream Park ABD</a></li>
          <li><a href="..." data-sehir-id="119">Scottsville Guney Afrika</a></li>
        </ul>
        </body></html>
        """
        respx.get(
            f"{base}/TR/YarisSever/Info/Page/GunlukYarisProgrami"
        ).mock(return_value=httpx.Response(200, text=intl_html))

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.get_race_card(test_date)

        assert cards == []


# ---------------------------------------------------------------------------
# Tests — Race Results
# ---------------------------------------------------------------------------


class TestGetRaceResults:
    """Tests for TJKClient.get_race_results."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_results_parse(self, results_date: date) -> None:
        """Parse race results with finish positions and times."""
        base = "https://www.tjk.org"

        respx.get(
            f"{base}/TR/YarisSever/Info/Page/GunlukYarisSonuclari"
        ).mock(return_value=httpx.Response(200, text=MAIN_RESULTS_PAGE_HTML))

        respx.get(
            f"{base}/TR/YarisSever/Info/Sehir/GunlukYarisSonuclari",
            params__contains={"SehirId": "1"},
        ).mock(return_value=httpx.Response(200, text=CITY_RESULTS_HTML))

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.get_race_results(results_date)

        assert len(cards) == 1
        card = cards[0]
        assert card.track_name == "Adana"
        assert card.race_number == 1
        assert card.distance_meters == 1500
        assert card.surface == "Kum"
        assert card.race_type == "Handikap 15/Dişi /H2"

        assert len(card.horses) == 2

        h1 = card.horses[0]
        assert h1.name == "FORTHCOMING QUEEN"
        assert h1.finish_position == 1
        assert h1.finish_time == "1.36.69"
        assert h1.age == 3
        assert h1.weight_kg == 57.5
        assert h1.gate_number == 8
        assert h1.gny == 3.4
        assert h1.agf == 17.0
        assert h1.hp == 52.0

        h2 = card.horses[1]
        assert h2.name == "KARDAHA"
        assert h2.finish_position == 2
        assert h2.finish_time == "1.37.61"
        assert h2.gny == 5.1
        assert h2.agf == 12.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_results_empty_date(self, results_date: date) -> None:
        """No results returns empty list."""
        base = "https://www.tjk.org"
        respx.get(
            f"{base}/TR/YarisSever/Info/Page/GunlukYarisSonuclari"
        ).mock(return_value=httpx.Response(200, text=EMPTY_PAGE_HTML))

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.get_race_results(results_date)

        assert cards == []

    @respx.mock
    @pytest.mark.asyncio
    async def test_results_http_error(self, results_date: date) -> None:
        """HTTP error returns empty list."""
        base = "https://www.tjk.org"
        respx.get(
            f"{base}/TR/YarisSever/Info/Page/GunlukYarisSonuclari"
        ).mock(return_value=httpx.Response(503))

        async with TJKClient(base_url=base, delay=0) as client:
            cards = await client.get_race_results(results_date)

        assert cards == []


# ---------------------------------------------------------------------------
# Tests — Client lifecycle
# ---------------------------------------------------------------------------


class TestClientLifecycle:
    """Tests for client open/close behaviour."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Client works as an async context manager."""
        async with TJKClient(delay=0) as client:
            assert client._client is not None

    @respx.mock
    @pytest.mark.asyncio
    async def test_manual_close(self) -> None:
        """Manual close works."""
        client = TJKClient(delay=0)
        await client.close()

    def test_default_config(self) -> None:
        """Default configuration matches expected values."""
        client = TJKClient()
        assert client.base_url == "https://www.tjk.org"
        assert client.delay == 2.0

    def test_custom_config(self) -> None:
        """Custom configuration is accepted."""
        client = TJKClient(base_url="https://test.tjk.org", delay=0.5)
        assert client.base_url == "https://test.tjk.org"
        assert client.delay == 0.5


# ---------------------------------------------------------------------------
# Tests — Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper/parsing functions."""

    def test_parse_race_config_maiden(self) -> None:
        from bs4 import BeautifulSoup
        from ganyan.scraper.tjk_api import _parse_race_config

        html = """
        <h3 class="race-config">
          <a class="aciklamaFancy">Maiden/DHÖW</a>
          , 4 Yaşlı Araplar,
          58 kg,
                      1400
          Kum,E.İ.D. :1.34.68
        </h3>
        """
        soup = BeautifulSoup(html, "html.parser")
        h3 = soup.select_one("h3.race-config")
        result = _parse_race_config(h3)

        assert result["race_type"] == "Maiden/DHÖW"
        assert result["distance_meters"] == 1400
        assert result["surface"] == "Kum"
        assert result["horse_type"] == "4 Yaşlı Araplar"
        assert result["weight_rule"] == "58 kg"

    def test_parse_race_config_handicap(self) -> None:
        from bs4 import BeautifulSoup
        from ganyan.scraper.tjk_api import _parse_race_config

        html = """
        <h3 class="race-config">
          <a class="aciklamaFancy">Handikap 16 /H1</a>
          , 4 ve Yukarı İngilizler,
                      1900
          Çim,E.İ.D. :2.00.00
        </h3>
        """
        soup = BeautifulSoup(html, "html.parser")
        h3 = soup.select_one("h3.race-config")
        result = _parse_race_config(h3)

        assert result["race_type"] == "Handikap 16 /H1"
        assert result["distance_meters"] == 1900
        assert result["surface"] == "Çim"
        assert result["weight_rule"] is None  # Handicaps don't have fixed weight

    def test_parse_race_config_none(self) -> None:
        from ganyan.scraper.tjk_api import _parse_race_config

        result = _parse_race_config(None)
        assert result["race_type"] is None
        assert result["distance_meters"] is None

    def test_safe_int_with_suffix(self) -> None:
        from ganyan.scraper.tjk_api import _safe_int

        assert _safe_int("5DS") == 5
        assert _safe_int("7") == 7
        assert _safe_int("") is None
        assert _safe_int(None) is None

    def test_safe_float_turkish_comma(self) -> None:
        from ganyan.scraper.tjk_api import _safe_float

        assert _safe_float("57,5") == 57.5
        assert _safe_float("3.40") == 3.4
        assert _safe_float("") is None
        assert _safe_float(None) is None

    def test_extract_agf(self) -> None:
        from bs4 import BeautifulSoup
        from ganyan.scraper.tjk_api import _extract_agf

        soup = BeautifulSoup('<td><span>%25(1)</span></td>', "html.parser")
        assert _extract_agf(soup.select_one("td")) == 25.0

        soup = BeautifulSoup('<td><span>-</span></td>', "html.parser")
        assert _extract_agf(soup.select_one("td")) is None

        assert _extract_agf(None) is None

    def test_parse_age(self) -> None:
        from ganyan.scraper.tjk_api import _parse_age

        assert _parse_age("4y a  a") == 4
        assert _parse_age("3y d  d") == 3
        assert _parse_age("") is None

    def test_extract_horse_name_results(self) -> None:
        from bs4 import BeautifulSoup
        from ganyan.scraper.tjk_api import _extract_horse_name_results

        html = '<td><a>FORTHCOMING QUEEN(3)</a></td>'
        soup = BeautifulSoup(html, "html.parser")
        assert _extract_horse_name_results(soup.select_one("td")) == "FORTHCOMING QUEEN"

        html = '<td><a>KARDAHA(2)</a></td>'
        soup = BeautifulSoup(html, "html.parser")
        assert _extract_horse_name_results(soup.select_one("td")) == "KARDAHA"

    def test_extract_eid(self) -> None:
        from bs4 import BeautifulSoup
        from ganyan.scraper.tjk_api import _extract_eid

        html = """
        <td>
          <div class="tooltipp">
            <span id="aciklamaFancyDrc">1.51.55</span>
            <a class="tooltiptextt">Bu derece...</a>
          </div>
        </td>
        """
        soup = BeautifulSoup(html, "html.parser")
        assert _extract_eid(soup.select_one("td")) == "1.51.55"

        # Empty cell
        html = "<td></td>"
        soup = BeautifulSoup(html, "html.parser")
        assert _extract_eid(soup.select_one("td")) is None
