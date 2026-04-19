"""Tests for the horse detail crawler — pedigree extraction only."""

from datetime import date

import httpx
import pytest
import respx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Horse
from ganyan.scraper.horse_crawler import (
    HorseCrawler, HorseProfile, _parse_birth_date, _parse_kunye,
)


# Minimal HTML shaped like the real kunye block.  Only the spans inside
# `div.kunye` matter for the parser.
_SAMPLE_PAGE = """
<html><body>
  <h2>ÇELİK ANSELMO</h2>
  <div class="grid_8 alpha omega kunye">
    <span class="key">İsim</span><span>ÇELİK ANSELMO</span>
    <span class="key">Yaş</span><span>3 y  ae</span>
    <span class="key">Doğ. Trh</span><span>5.04.2023</span>
    <span class="key">Handikap P.</span><span>40</span>
    <span class="key">Baba</span><span>GELİBOLU</span>
    <span class="key">Anne</span><span>ALTIN TURBO/TURBO</span>
    <span class="key">Antrenör</span><span>F.KİP</span>
  </div>
</body></html>
"""

_EMPTY_PAGE = "<html><body><p>Horse not found</p></body></html>"


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_parse_birth_date():
    assert _parse_birth_date("5.04.2023") == date(2023, 4, 5)
    assert _parse_birth_date("15.12.2022") == date(2022, 12, 15)
    assert _parse_birth_date("") is None
    assert _parse_birth_date("not-a-date") is None


def test_parse_kunye_success():
    profile = _parse_kunye(_SAMPLE_PAGE, at_id=109699)
    assert profile is not None
    assert profile.at_id == 109699
    assert profile.name == "ÇELİK ANSELMO"
    assert profile.sire == "GELİBOLU"
    assert profile.dam == "ALTIN TURBO/TURBO"
    assert profile.birth_date == date(2023, 4, 5)


def test_parse_kunye_missing_block():
    assert _parse_kunye(_EMPTY_PAGE, at_id=1) is None


@respx.mock
@pytest.mark.asyncio
async def test_crawl_populates_horse(db_session):
    horse = Horse(name="ÇELİK ANSELMO", tjk_at_id=109699)
    db_session.add(horse)
    db_session.commit()

    respx.get("https://www.tjk.org/TR/YarisSever/Query/ConnectedPage/AtKosuBilgileri").mock(
        return_value=httpx.Response(200, text=_SAMPLE_PAGE),
    )

    async with HorseCrawler(
        db_session, delay=0, concurrency=1,
    ) as crawler:
        stored = await crawler.crawl_missing_profiles()

    db_session.expire_all()
    refreshed = db_session.query(Horse).filter_by(name="ÇELİK ANSELMO").one()
    assert stored == 1
    assert refreshed.sire == "GELİBOLU"
    assert refreshed.dam == "ALTIN TURBO/TURBO"
    assert refreshed.birth_date == date(2023, 4, 5)
    assert refreshed.profile_crawled_at is not None


@respx.mock
@pytest.mark.asyncio
async def test_crawl_skips_horses_without_at_id(db_session):
    db_session.add(Horse(name="No Id Horse"))
    db_session.commit()

    async with HorseCrawler(db_session, delay=0, concurrency=1) as crawler:
        stored = await crawler.crawl_missing_profiles()

    assert stored == 0


@respx.mock
@pytest.mark.asyncio
async def test_crawl_skips_already_profiled(db_session):
    """Horses with profile_crawled_at set are not re-fetched."""
    from datetime import datetime as dt

    horse = Horse(
        name="Already Done",
        tjk_at_id=42,
        profile_crawled_at=dt.utcnow(),
        sire="Old Sire",
    )
    db_session.add(horse)
    db_session.commit()

    async with HorseCrawler(db_session, delay=0, concurrency=1) as crawler:
        stored = await crawler.crawl_missing_profiles()

    assert stored == 0
