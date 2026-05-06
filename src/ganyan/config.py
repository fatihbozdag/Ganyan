from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite:///data/ganyan.db"
    tjk_base_url: str = "https://www.tjk.org"
    scrape_delay: float = 2.0
    log_level: str = "INFO"
    flask_port: int = 5003
    flask_debug: bool = False
    show_backfill_ui: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    return Settings()
