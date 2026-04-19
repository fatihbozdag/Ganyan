import pytest
from ganyan.config import Settings


@pytest.fixture(autouse=True)
def _fast_tjk_retries(monkeypatch):
    """Keep failure-path scraper tests quick by shrinking the retry backoff.

    Production defaults (3 retries, exp base 2s) would add ~14s of sleeps per
    mocked 5xx response.  Tests verify behaviour, not real-world latency.
    """
    monkeypatch.setattr(
        "ganyan.scraper.tjk_api._DEFAULT_MAX_RETRIES", 1, raising=False,
    )
    monkeypatch.setattr(
        "ganyan.scraper.tjk_api._DEFAULT_BACKOFF_BASE", 0.0, raising=False,
    )


@pytest.fixture
def settings():
    return Settings(
        database_url="postgresql+psycopg://ganyan:ganyan@localhost:5432/ganyan_test"
    )


def test_settings_defaults(settings):
    assert settings.tjk_base_url == "https://www.tjk.org"
    assert settings.scrape_delay == 2.0
    assert settings.flask_port == 5003
    assert "ganyan_test" in settings.database_url
