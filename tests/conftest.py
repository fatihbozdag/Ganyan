import pytest
from ganyan.config import Settings


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
