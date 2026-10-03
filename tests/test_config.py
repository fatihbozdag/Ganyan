"""Tests for ganyan.config.Settings defaults."""
import pytest

from ganyan.config import Settings

_ENV_KEYS = ("TJK_BASE_URL", "SCRAPE_DELAY", "FLASK_PORT", "DATABASE_URL")


@pytest.fixture
def settings(monkeypatch):
    # Hermetic: ignore the developer's .env and any exported overrides so the
    # assertions below check the code defaults, not the local environment.
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return Settings(
        _env_file=None,
        database_url="postgresql+psycopg://ganyan:ganyan@localhost:5432/ganyan_test",
    )


def test_settings_defaults(settings):
    assert settings.tjk_base_url == "https://www.tjk.org"
    assert settings.scrape_delay == 2.0
    assert settings.flask_port == 5003
    assert "ganyan_test" in settings.database_url
