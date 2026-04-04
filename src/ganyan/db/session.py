from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ganyan.config import get_settings


def get_engine(database_url: str | None = None):
    url = database_url or get_settings().database_url
    return create_engine(url)


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    engine = get_engine(database_url)
    return sessionmaker(bind=engine)


def get_session(database_url: str | None = None) -> Session:
    factory = get_session_factory(database_url)
    return factory()
