# Ganyan Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild Ganyan as a working Turkish horse racing prediction system with TJK data scraping, Bayesian predictions, CLI, and web interface.

**Architecture:** Service-oriented monorepo — scraper, predictor, and web/CLI layers share a PostgreSQL database. Single Python package managed by `uv` with clean module boundaries.

**Tech Stack:** Python 3.12+, SQLAlchemy 2.0, Alembic, httpx, BeautifulSoup4, Flask, HTMX, Typer, Pydantic Settings, PostgreSQL 16, Docker Compose, NumPy

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Package definition, dependencies, CLI entry point |
| `docker-compose.yml` | PostgreSQL 16 container |
| `.env.example` | Template for environment variables |
| `alembic.ini` | Alembic config pointing to migrations |
| `alembic/env.py` | Migration environment (reads DATABASE_URL from config) |
| `src/ganyan/__init__.py` | Package root, version |
| `src/ganyan/config.py` | `Settings` class via pydantic-settings |
| `src/ganyan/db/__init__.py` | Re-exports engine, Session, Base |
| `src/ganyan/db/models.py` | ORM models: Track, Race, Horse, RaceEntry, ScrapeLog |
| `src/ganyan/db/session.py` | Engine creation, sessionmaker, get_session |
| `src/ganyan/scraper/__init__.py` | Re-exports TJKClient |
| `src/ganyan/scraper/tjk_api.py` | TJK API/HTML client — fetch race cards and results |
| `src/ganyan/scraper/parser.py` | Raw HTML/JSON → validated dataclasses |
| `src/ganyan/scraper/backfill.py` | Incremental historical data loader |
| `src/ganyan/predictor/__init__.py` | Re-exports BayesianPredictor, Prediction |
| `src/ganyan/predictor/features.py` | Feature extraction from race history |
| `src/ganyan/predictor/bayesian.py` | Empirical Bayesian model |
| `src/ganyan/web/__init__.py` | Empty |
| `src/ganyan/web/app.py` | Flask factory: `create_app()` |
| `src/ganyan/web/routes.py` | All routes (dashboard, races, predict, scrape) |
| `src/ganyan/web/templates/base.html` | Base layout with Bootstrap 5 + HTMX |
| `src/ganyan/web/templates/index.html` | Dashboard |
| `src/ganyan/web/templates/races.html` | Race cards for a date |
| `src/ganyan/web/templates/predict.html` | Prediction results |
| `src/ganyan/web/templates/history.html` | Prediction accuracy history |
| `src/ganyan/cli/__init__.py` | Empty |
| `src/ganyan/cli/main.py` | Typer app with scrape, predict, races, db commands |
| `tests/conftest.py` | Shared fixtures: test DB, session, sample data |
| `tests/test_scraper/test_parser.py` | Parser unit tests |
| `tests/test_scraper/test_tjk_api.py` | TJK client tests (mocked HTTP) |
| `tests/test_scraper/test_backfill.py` | Backfill logic tests |
| `tests/test_predictor/test_features.py` | Feature extraction tests |
| `tests/test_predictor/test_bayesian.py` | Bayesian model tests |
| `tests/test_web/test_routes.py` | Flask route tests |
| `tests/test_cli/test_main.py` | CLI invocation tests |

---

### Task 1: Project Scaffold + PostgreSQL + Config

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `src/ganyan/__init__.py`
- Create: `src/ganyan/config.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Remove old project files**

Delete all existing source files but keep `.git/`, `docs/`, `.claude/`, `data/processed/` (historical CSVs for later import), and `CLAUDE.md`.

```bash
# From project root
rm -f app.py bayesian_predictor.py race_analyzer.py current_race_prediction.py test_prediction.py
rm -f setup.py requirements.txt startup.sh __init__.py debug_response.html current_race.json scrapy.cfg
rm -rf scrapers/ tjk_scraper/ src/ utils/ analysis/ templates/ scripts/
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
services:
  db:
    image: postgres:16
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: ganyan
      POSTGRES_USER: ganyan
      POSTGRES_PASSWORD: ganyan
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

- [ ] **Step 3: Create .env.example**

```
DATABASE_URL=postgresql+psycopg://ganyan:ganyan@localhost:5432/ganyan
TJK_BASE_URL=https://www.tjk.org
SCRAPE_DELAY=2.0
LOG_LEVEL=INFO
FLASK_PORT=5003
FLASK_DEBUG=false
```

- [ ] **Step 4: Create pyproject.toml**

```toml
[project]
name = "ganyan"
version = "0.1.0"
description = "Turkish horse racing prediction system"
requires-python = ">=3.12"
dependencies = [
    "sqlalchemy[asyncio]>=2.0",
    "alembic>=1.13",
    "psycopg[binary]>=3.1",
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "flask>=3.0",
    "typer>=0.12",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "respx>=0.21",
    "factory-boy>=3.3",
]

[project.scripts]
ganyan = "ganyan.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ganyan"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 5: Create package init**

```python
# src/ganyan/__init__.py
__version__ = "0.1.0"
```

- [ ] **Step 6: Create config module**

```python
# src/ganyan/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://ganyan:ganyan@localhost:5432/ganyan"
    tjk_base_url: str = "https://www.tjk.org"
    scrape_delay: float = 2.0
    log_level: str = "INFO"
    flask_port: int = 5003
    flask_debug: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 7: Create directory structure and empty __init__.py files**

```bash
mkdir -p src/ganyan/db src/ganyan/scraper src/ganyan/predictor src/ganyan/web/templates src/ganyan/web/static src/ganyan/cli
mkdir -p tests/test_scraper tests/test_predictor tests/test_web tests/test_cli
touch src/ganyan/db/__init__.py src/ganyan/scraper/__init__.py src/ganyan/predictor/__init__.py
touch src/ganyan/web/__init__.py src/ganyan/cli/__init__.py
touch tests/__init__.py tests/test_scraper/__init__.py tests/test_predictor/__init__.py
touch tests/test_web/__init__.py tests/test_cli/__init__.py
```

- [ ] **Step 8: Create tests/conftest.py with config test**

```python
# tests/conftest.py
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
```

- [ ] **Step 9: Start PostgreSQL and install dependencies**

```bash
docker compose up -d
cp .env.example .env
uv sync --all-extras
```

- [ ] **Step 10: Run test to verify setup**

Run: `uv run pytest tests/conftest.py -v`
Expected: PASS — `test_settings_defaults` passes

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "feat: project scaffold with pyproject.toml, docker-compose, config"
```

---

### Task 2: Database Models + Alembic Migrations

**Files:**
- Create: `src/ganyan/db/session.py`
- Create: `src/ganyan/db/models.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`
- Modify: `src/ganyan/db/__init__.py`

- [ ] **Step 1: Write test for database models**

```python
# tests/test_db_models.py
import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Track, Race, Horse, RaceEntry, ScrapeLog, RaceStatus, ScrapeStatus


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_create_track(db_session):
    track = Track(name="İstanbul", city="İstanbul", surface_types=["çim", "kum"])
    db_session.add(track)
    db_session.commit()
    assert track.id is not None
    assert track.name == "İstanbul"


def test_create_race_with_track(db_session):
    track = Track(name="Ankara", city="Ankara", surface_types=["sentetik"])
    db_session.add(track)
    db_session.flush()

    race = Race(
        track_id=track.id,
        date=date(2026, 4, 5),
        race_number=1,
        distance_meters=1200,
        surface="sentetik",
        status=RaceStatus.scheduled,
    )
    db_session.add(race)
    db_session.commit()
    assert race.id is not None
    assert race.track.name == "Ankara"


def test_create_horse_and_entry(db_session):
    track = Track(name="İzmir", city="İzmir", surface_types=["çim"])
    db_session.add(track)
    db_session.flush()

    race = Race(
        track_id=track.id,
        date=date(2026, 4, 5),
        race_number=2,
        distance_meters=1400,
        surface="çim",
        status=RaceStatus.scheduled,
    )
    db_session.add(race)
    db_session.flush()

    horse = Horse(name="Karayel", age=4, origin="TR")
    db_session.add(horse)
    db_session.flush()

    entry = RaceEntry(
        race_id=race.id,
        horse_id=horse.id,
        gate_number=3,
        jockey="Ahmet Çelik",
        weight_kg=57.0,
        hp=85.5,
        kgs=21,
        last_six="1 3 2 4 1 2",
    )
    db_session.add(entry)
    db_session.commit()

    assert entry.id is not None
    assert entry.horse.name == "Karayel"
    assert entry.race.race_number == 2
    assert entry.finish_position is None  # pre-race


def test_race_unique_constraint(db_session):
    track = Track(name="Bursa", city="Bursa", surface_types=["çim"])
    db_session.add(track)
    db_session.flush()

    race1 = Race(
        track_id=track.id, date=date(2026, 4, 5), race_number=1,
        distance_meters=1200, surface="çim", status=RaceStatus.scheduled,
    )
    race2 = Race(
        track_id=track.id, date=date(2026, 4, 5), race_number=1,
        distance_meters=1600, surface="çim", status=RaceStatus.scheduled,
    )
    db_session.add(race1)
    db_session.commit()
    db_session.add(race2)
    with pytest.raises(Exception):  # IntegrityError
        db_session.commit()


def test_scrape_log(db_session):
    log = ScrapeLog(
        date=date(2026, 4, 5),
        track="İstanbul",
        status=ScrapeStatus.success,
    )
    db_session.add(log)
    db_session.commit()
    assert log.id is not None
    assert log.scraped_at is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ganyan.db.models'`

- [ ] **Step 3: Implement database models**

```python
# src/ganyan/db/models.py
import enum
from datetime import date as date_type, datetime

from sqlalchemy import (
    String, SmallInteger, Integer, Numeric, Date, DateTime, Enum,
    ForeignKey, UniqueConstraint, Index, func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RaceStatus(enum.Enum):
    scheduled = "scheduled"
    resulted = "resulted"
    cancelled = "cancelled"


class ScrapeStatus(enum.Enum):
    success = "success"
    failed = "failed"
    skipped = "skipped"


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    city: Mapped[str | None] = mapped_column(String(100))
    surface_types: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)

    races: Mapped[list["Race"]] = relationship(back_populates="track")


class Race(Base):
    __tablename__ = "races"
    __table_args__ = (
        UniqueConstraint("track_id", "date", "race_number", name="uq_race_track_date_num"),
        Index("ix_races_date", "date"),
        Index("ix_races_track_date", "track_id", "date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"))
    date: Mapped[date_type] = mapped_column(Date)
    race_number: Mapped[int] = mapped_column(SmallInteger)
    distance_meters: Mapped[int | None] = mapped_column(Integer)
    surface: Mapped[str | None] = mapped_column(String(50))
    race_type: Mapped[str | None] = mapped_column(String(100))
    horse_type: Mapped[str | None] = mapped_column(String(100))
    weight_rule: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[RaceStatus] = mapped_column(Enum(RaceStatus), default=RaceStatus.scheduled)

    track: Mapped["Track"] = relationship(back_populates="races")
    entries: Mapped[list["RaceEntry"]] = relationship(back_populates="race")


class Horse(Base):
    __tablename__ = "horses"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    age: Mapped[int | None] = mapped_column(SmallInteger)
    origin: Mapped[str | None] = mapped_column(String(100))
    owner: Mapped[str | None] = mapped_column(String(200))
    trainer: Mapped[str | None] = mapped_column(String(200))

    entries: Mapped[list["RaceEntry"]] = relationship(back_populates="horse")


class RaceEntry(Base):
    __tablename__ = "race_entries"
    __table_args__ = (
        Index("ix_race_entries_race_id", "race_id"),
        Index("ix_race_entries_horse_id", "horse_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id"))
    horse_id: Mapped[int] = mapped_column(ForeignKey("horses.id"))
    gate_number: Mapped[int | None] = mapped_column(SmallInteger)
    jockey: Mapped[str | None] = mapped_column(String(200))
    weight_kg: Mapped[float | None] = mapped_column(Numeric(4, 1))
    hp: Mapped[float | None] = mapped_column(Numeric(5, 1))
    kgs: Mapped[int | None] = mapped_column(SmallInteger)
    s20: Mapped[float | None] = mapped_column(Numeric(5, 2))
    eid: Mapped[str | None] = mapped_column(String(20))
    gny: Mapped[float | None] = mapped_column(Numeric(5, 2))
    agf: Mapped[float | None] = mapped_column(Numeric(5, 2))
    last_six: Mapped[str | None] = mapped_column(String(50))
    finish_position: Mapped[int | None] = mapped_column(SmallInteger)
    finish_time: Mapped[str | None] = mapped_column(String(20))
    performance_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    predicted_probability: Mapped[float | None] = mapped_column(Numeric(5, 2))

    race: Mapped["Race"] = relationship(back_populates="entries")
    horse: Mapped["Horse"] = relationship(back_populates="entries")


class ScrapeLog(Base):
    __tablename__ = "scrape_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date_type] = mapped_column(Date)
    track: Mapped[str] = mapped_column(String(100))
    status: Mapped[ScrapeStatus] = mapped_column(Enum(ScrapeStatus))
    scraped_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db_models.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Implement session module**

```python
# src/ganyan/db/session.py
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
```

- [ ] **Step 6: Update db __init__.py**

```python
# src/ganyan/db/__init__.py
from ganyan.db.models import Base, Track, Race, Horse, RaceEntry, ScrapeLog, RaceStatus, ScrapeStatus
from ganyan.db.session import get_engine, get_session_factory, get_session

__all__ = [
    "Base", "Track", "Race", "Horse", "RaceEntry", "ScrapeLog",
    "RaceStatus", "ScrapeStatus",
    "get_engine", "get_session_factory", "get_session",
]
```

- [ ] **Step 7: Set up Alembic**

```bash
uv run alembic init alembic
```

Then replace `alembic/env.py`:

```python
# alembic/env.py
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from ganyan.config import get_settings
from ganyan.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url():
    return get_settings().database_url


def run_migrations_offline():
    url = get_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(configuration, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 8: Generate and run initial migration**

```bash
uv run alembic revision --autogenerate -m "initial schema"
uv run alembic upgrade head
```

- [ ] **Step 9: Verify database tables exist**

```bash
docker compose exec db psql -U ganyan -d ganyan -c "\dt"
```

Expected: tables `tracks`, `races`, `horses`, `race_entries`, `scrape_log`, `alembic_version`

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "feat: database models, session management, alembic migrations"
```

---

### Task 3: TJK Scraper — Parser

**Files:**
- Create: `src/ganyan/scraper/parser.py`
- Create: `tests/test_scraper/__init__.py` (already exists from Task 1)
- Create: `tests/test_scraper/test_parser.py`

- [ ] **Step 1: Write parser tests**

```python
# tests/test_scraper/test_parser.py
from datetime import date

from ganyan.scraper.parser import (
    parse_eid_to_seconds,
    parse_last_six,
    normalize_track_name,
    RawRaceCard,
    RawHorseEntry,
    ParsedRaceCard,
    parse_race_card,
)


def test_parse_eid_to_seconds_standard():
    assert parse_eid_to_seconds("1.30.45") == 90.45


def test_parse_eid_to_seconds_short():
    assert parse_eid_to_seconds("58.20") == 58.20


def test_parse_eid_to_seconds_empty():
    assert parse_eid_to_seconds("") is None
    assert parse_eid_to_seconds(None) is None


def test_parse_last_six():
    assert parse_last_six("2 4 4 5 2 7") == [2, 4, 4, 5, 2, 7]


def test_parse_last_six_with_missing():
    assert parse_last_six("1 3 - 2 - 4") == [1, 3, None, 2, None, 4]


def test_parse_last_six_empty():
    assert parse_last_six("") == []
    assert parse_last_six(None) == []


def test_normalize_track_name():
    assert normalize_track_name("İstanbul") == "İstanbul"
    assert normalize_track_name("istanbul") == "İstanbul"
    assert normalize_track_name("ISTANBUL") == "İstanbul"
    assert normalize_track_name(" İstanbul ") == "İstanbul"


def test_parse_race_card():
    raw = RawRaceCard(
        track_name="İstanbul",
        date=date(2026, 4, 5),
        race_number=3,
        distance_meters=1400,
        surface="Çim",
        race_type="Handikap",
        horse_type="İngiliz",
        weight_rule="Handikap",
        horses=[
            RawHorseEntry(
                name="Karayel",
                age=4,
                origin="TR",
                owner="Ali Kaya",
                trainer="Mehmet Demir",
                gate_number=3,
                jockey="Ahmet Çelik",
                weight_kg=57.0,
                hp=85.5,
                kgs=21,
                s20=12.50,
                eid="1.30.45",
                gny=8.30,
                agf=5.20,
                last_six="1 3 2 4 1 2",
            ),
        ],
    )
    parsed = parse_race_card(raw)
    assert parsed.track_name == "İstanbul"
    assert parsed.race_number == 3
    assert len(parsed.horses) == 1
    assert parsed.horses[0].name == "Karayel"
    assert parsed.horses[0].eid_seconds == 90.45
    assert parsed.horses[0].last_six_parsed == [1, 3, 2, 4, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scraper/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement parser**

```python
# src/ganyan/scraper/parser.py
from dataclasses import dataclass, field
from datetime import date


# --- Raw data structures (from scraper) ---

@dataclass
class RawHorseEntry:
    name: str
    age: int | None = None
    origin: str | None = None
    owner: str | None = None
    trainer: str | None = None
    gate_number: int | None = None
    jockey: str | None = None
    weight_kg: float | None = None
    hp: float | None = None
    kgs: int | None = None
    s20: float | None = None
    eid: str | None = None
    gny: float | None = None
    agf: float | None = None
    last_six: str | None = None
    finish_position: int | None = None
    finish_time: str | None = None


@dataclass
class RawRaceCard:
    track_name: str
    date: date
    race_number: int
    distance_meters: int | None = None
    surface: str | None = None
    race_type: str | None = None
    horse_type: str | None = None
    weight_rule: str | None = None
    horses: list[RawHorseEntry] = field(default_factory=list)


# --- Parsed data structures (validated, enriched) ---

@dataclass
class ParsedHorseEntry:
    name: str
    age: int | None = None
    origin: str | None = None
    owner: str | None = None
    trainer: str | None = None
    gate_number: int | None = None
    jockey: str | None = None
    weight_kg: float | None = None
    hp: float | None = None
    kgs: int | None = None
    s20: float | None = None
    eid: str | None = None
    eid_seconds: float | None = None
    gny: float | None = None
    agf: float | None = None
    last_six: str | None = None
    last_six_parsed: list[int | None] = field(default_factory=list)
    finish_position: int | None = None
    finish_time: str | None = None


@dataclass
class ParsedRaceCard:
    track_name: str
    date: date
    race_number: int
    distance_meters: int | None = None
    surface: str | None = None
    race_type: str | None = None
    horse_type: str | None = None
    weight_rule: str | None = None
    horses: list[ParsedHorseEntry] = field(default_factory=list)


# --- Parsing functions ---

# Canonical track names (Turkish-correct capitalization)
TRACK_NAMES = {
    "istanbul": "İstanbul",
    "İstanbul": "İstanbul",
    "ankara": "Ankara",
    "izmir": "İzmir",
    "İzmir": "İzmir",
    "bursa": "Bursa",
    "adana": "Adana",
    "antalya": "Antalya",
    "elazığ": "Elazığ",
    "Elazığ": "Elazığ",
    "diyarbakır": "Diyarbakır",
    "Diyarbakır": "Diyarbakır",
    "kocaeli": "Kocaeli",
    "şanlıurfa": "Şanlıurfa",
    "Şanlıurfa": "Şanlıurfa",
}


def parse_eid_to_seconds(eid: str | None) -> float | None:
    """Convert EİD time string to total seconds.

    Formats: '1.30.45' → 90.45 (min.sec.hundredths)
             '58.20' → 58.20 (sec.hundredths)
    """
    if not eid or not eid.strip():
        return None
    parts = eid.strip().split(".")
    if len(parts) == 3:
        minutes, seconds, hundredths = int(parts[0]), int(parts[1]), int(parts[2])
        return minutes * 60 + seconds + hundredths / 100
    elif len(parts) == 2:
        seconds, hundredths = int(parts[0]), int(parts[1])
        return seconds + hundredths / 100
    return None


def parse_last_six(last_six: str | None) -> list[int | None]:
    """Parse last six race positions. '-' means no result."""
    if not last_six or not last_six.strip():
        return []
    result = []
    for part in last_six.strip().split():
        if part == "-":
            result.append(None)
        else:
            try:
                result.append(int(part))
            except ValueError:
                result.append(None)
    return result


def normalize_track_name(name: str) -> str:
    """Normalize track name to canonical Turkish form."""
    stripped = name.strip()
    # Try exact match first
    if stripped in TRACK_NAMES:
        return TRACK_NAMES[stripped]
    # Try case-insensitive
    lower = stripped.lower()
    if lower in TRACK_NAMES:
        return TRACK_NAMES[lower]
    # Return title-cased if unknown
    return stripped.title()


def parse_race_card(raw: RawRaceCard) -> ParsedRaceCard:
    """Parse a raw race card into validated, enriched form."""
    horses = []
    for h in raw.horses:
        horses.append(ParsedHorseEntry(
            name=h.name.strip(),
            age=h.age,
            origin=h.origin,
            owner=h.owner,
            trainer=h.trainer,
            gate_number=h.gate_number,
            jockey=h.jockey,
            weight_kg=h.weight_kg,
            hp=h.hp,
            kgs=h.kgs,
            s20=h.s20,
            eid=h.eid,
            eid_seconds=parse_eid_to_seconds(h.eid),
            gny=h.gny,
            agf=h.agf,
            last_six=h.last_six,
            last_six_parsed=parse_last_six(h.last_six),
            finish_position=h.finish_position,
            finish_time=h.finish_time,
        ))

    return ParsedRaceCard(
        track_name=normalize_track_name(raw.track_name),
        date=raw.date,
        race_number=raw.race_number,
        distance_meters=raw.distance_meters,
        surface=raw.surface.lower() if raw.surface else None,
        race_type=raw.race_type,
        horse_type=raw.horse_type,
        weight_rule=raw.weight_rule,
        horses=horses,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scraper/test_parser.py -v`
Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ganyan/scraper/parser.py tests/test_scraper/test_parser.py
git commit -m "feat: TJK data parser with EİD, last_six, track normalization"
```

---

### Task 4: TJK Scraper — API/HTML Client

**Files:**
- Create: `src/ganyan/scraper/tjk_api.py`
- Create: `tests/test_scraper/test_tjk_api.py`
- Modify: `src/ganyan/scraper/__init__.py`

**Important note:** This task requires live investigation of TJK's website to find API endpoints. The implementation below uses HTML scraping as the baseline — if JSON APIs are discovered during development, `tjk_api.py` should be updated to prefer those.

- [ ] **Step 1: Investigate TJK website for API endpoints**

Before writing tests, manually investigate TJK's website:

```bash
# Check if TJK has a JSON API behind their race program pages
# Open browser dev tools on tjk.org, go to race program page, check Network tab for XHR/Fetch requests
# Common patterns to look for:
# - /api/* endpoints returning JSON
# - /services/* SOAP/REST endpoints
# - GraphQL at /graphql
# Look at the race program page: https://www.tjk.org/TR/YarisSever/Query/Page/GunlukYarisProgrami
```

Document what you find. The implementation below assumes HTML scraping — update if APIs are found.

- [ ] **Step 2: Write TJK client tests (mocked HTTP)**

```python
# tests/test_scraper/test_tjk_api.py
import pytest
import httpx
import respx
from datetime import date

from ganyan.scraper.tjk_api import TJKClient


SAMPLE_RACE_PROGRAM_HTML = """
<html><body>
<div class="race-program">
  <div class="race-header">
    <span class="race-no">1. Koşu</span>
    <span class="distance">1200m</span>
    <span class="surface">Çim</span>
    <span class="race-type">Handikap</span>
  </div>
  <table class="race-table">
    <tr class="horse-row">
      <td class="gate">1</td>
      <td class="horse-name">Karayel</td>
      <td class="age">4</td>
      <td class="origin">TR</td>
      <td class="jockey">Ahmet Çelik</td>
      <td class="weight">57.0</td>
      <td class="hp">85.5</td>
      <td class="kgs">21</td>
      <td class="s20">12.50</td>
      <td class="eid">1.30.45</td>
      <td class="gny">8.30</td>
      <td class="agf">5.20</td>
      <td class="last-six">1 3 2 4 1 2</td>
      <td class="owner">Ali Kaya</td>
      <td class="trainer">Mehmet Demir</td>
    </tr>
    <tr class="horse-row">
      <td class="gate">2</td>
      <td class="horse-name">Rüzgar</td>
      <td class="age">3</td>
      <td class="origin">IRE</td>
      <td class="jockey">Halis Karataş</td>
      <td class="weight">55.5</td>
      <td class="hp">78.0</td>
      <td class="kgs">14</td>
      <td class="s20">10.80</td>
      <td class="eid">1.31.20</td>
      <td class="gny">7.10</td>
      <td class="agf">4.80</td>
      <td class="last-six">3 2 1 5 4 3</td>
      <td class="owner">Veli Yılmaz</td>
      <td class="trainer">Can Öz</td>
    </tr>
  </table>
</div>
</body></html>
"""


@pytest.fixture
def client():
    return TJKClient(base_url="https://www.tjk.org", delay=0)


@respx.mock
@pytest.mark.asyncio
async def test_get_race_card_parses_html(client):
    respx.get("https://www.tjk.org/TR/YarisSever/Query/Page/GunlukYarisProgrami").mock(
        return_value=httpx.Response(200, html=SAMPLE_RACE_PROGRAM_HTML)
    )
    cards = await client.get_race_card(date(2026, 4, 5))
    assert len(cards) >= 1
    card = cards[0]
    assert card.race_number == 1
    assert card.distance_meters == 1200
    assert len(card.horses) == 2
    assert card.horses[0].name == "Karayel"
    assert card.horses[1].name == "Rüzgar"


@respx.mock
@pytest.mark.asyncio
async def test_get_race_card_handles_empty(client):
    respx.get("https://www.tjk.org/TR/YarisSever/Query/Page/GunlukYarisProgrami").mock(
        return_value=httpx.Response(200, html="<html><body>Yarış bulunamadı</body></html>")
    )
    cards = await client.get_race_card(date(2026, 4, 5))
    assert cards == []


@respx.mock
@pytest.mark.asyncio
async def test_get_race_card_handles_http_error(client):
    respx.get("https://www.tjk.org/TR/YarisSever/Query/Page/GunlukYarisProgrami").mock(
        return_value=httpx.Response(500)
    )
    cards = await client.get_race_card(date(2026, 4, 5))
    assert cards == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_scraper/test_tjk_api.py -v`
Expected: FAIL — `ModuleNotFoundError`

Note: You will also need `pytest-asyncio` — add it to dev dependencies:

```bash
uv add --dev pytest-asyncio
```

- [ ] **Step 4: Implement TJK client**

```python
# src/ganyan/scraper/tjk_api.py
import asyncio
import logging
from datetime import date

import httpx
from bs4 import BeautifulSoup

from ganyan.scraper.parser import RawRaceCard, RawHorseEntry

logger = logging.getLogger(__name__)


class TJKClient:
    """Client for fetching race data from TJK website.

    Tries JSON API first (if discovered), falls back to HTML scraping.
    """

    def __init__(self, base_url: str = "https://www.tjk.org", delay: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    "Accept-Language": "tr-TR,tr;q=0.9",
                },
                timeout=30.0,
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_race_card(self, race_date: date) -> list[RawRaceCard]:
        """Fetch race cards for a given date."""
        try:
            client = await self._get_client()
            url = f"{self.base_url}/TR/YarisSever/Query/Page/GunlukYarisProgrami"
            params = {"QueryParameter_Tarih": race_date.strftime("%d/%m/%Y")}
            response = await client.get(url, params=params)
            response.raise_for_status()

            if self.delay > 0:
                await asyncio.sleep(self.delay)

            return self._parse_race_program_html(response.text, race_date)
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching race card for {race_date}: {e}")
            return []

    async def get_race_results(self, race_date: date) -> list[RawRaceCard]:
        """Fetch race results for a given date."""
        try:
            client = await self._get_client()
            url = f"{self.base_url}/TR/YarisSever/Query/Page/GunlukYarisSonuclari"
            params = {"QueryParameter_Tarih": race_date.strftime("%d/%m/%Y")}
            response = await client.get(url, params=params)
            response.raise_for_status()

            if self.delay > 0:
                await asyncio.sleep(self.delay)

            return self._parse_race_results_html(response.text, race_date)
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching results for {race_date}: {e}")
            return []

    def _parse_race_program_html(self, html: str, race_date: date) -> list[RawRaceCard]:
        """Parse the daily race program HTML page into RawRaceCard objects."""
        soup = BeautifulSoup(html, "html.parser")
        cards = []

        race_divs = soup.select("div.race-program")
        if not race_divs:
            # Try alternative selectors — TJK may use different class names
            race_divs = soup.select("[class*='race']")

        for race_div in race_divs:
            try:
                card = self._parse_single_race_div(race_div, race_date)
                if card:
                    cards.append(card)
            except Exception as e:
                logger.warning(f"Error parsing race div: {e}")
                continue

        return cards

    def _parse_single_race_div(self, div, race_date: date) -> RawRaceCard | None:
        """Parse a single race div into a RawRaceCard."""
        # Extract race header info
        race_no_el = div.select_one(".race-no, [class*='race-no']")
        if not race_no_el:
            return None

        race_no_text = race_no_el.get_text(strip=True)
        race_number = self._extract_number(race_no_text)
        if not race_number:
            return None

        distance_el = div.select_one(".distance, [class*='distance']")
        distance = self._extract_number(distance_el.get_text(strip=True)) if distance_el else None

        surface_el = div.select_one(".surface, [class*='surface']")
        surface = surface_el.get_text(strip=True) if surface_el else None

        race_type_el = div.select_one(".race-type, [class*='race-type']")
        race_type = race_type_el.get_text(strip=True) if race_type_el else None

        # Extract track name from page context (may be in parent or header)
        track_name = self._extract_track_name(div) or "Unknown"

        # Parse horses
        horses = []
        for row in div.select("tr.horse-row, [class*='horse-row']"):
            horse = self._parse_horse_row(row)
            if horse:
                horses.append(horse)

        return RawRaceCard(
            track_name=track_name,
            date=race_date,
            race_number=race_number,
            distance_meters=distance,
            surface=surface,
            race_type=race_type,
            horses=horses,
        )

    def _parse_horse_row(self, row) -> RawHorseEntry | None:
        """Parse a single horse row from a race table."""

        def get_text(cls: str) -> str | None:
            el = row.select_one(f".{cls}, [class*='{cls}']")
            return el.get_text(strip=True) if el else None

        def get_float(cls: str) -> float | None:
            text = get_text(cls)
            if not text:
                return None
            try:
                return float(text.replace(",", "."))
            except ValueError:
                return None

        def get_int(cls: str) -> int | None:
            text = get_text(cls)
            if not text:
                return None
            try:
                return int(text)
            except ValueError:
                return None

        name = get_text("horse-name")
        if not name:
            return None

        return RawHorseEntry(
            name=name,
            age=get_int("age"),
            origin=get_text("origin"),
            owner=get_text("owner"),
            trainer=get_text("trainer"),
            gate_number=get_int("gate"),
            jockey=get_text("jockey"),
            weight_kg=get_float("weight"),
            hp=get_float("hp"),
            kgs=get_int("kgs"),
            s20=get_float("s20"),
            eid=get_text("eid"),
            gny=get_float("gny"),
            agf=get_float("agf"),
            last_six=get_text("last-six"),
        )

    def _parse_race_results_html(self, html: str, race_date: date) -> list[RawRaceCard]:
        """Parse race results page. Same structure as program but with finish data."""
        # Results page has similar structure — reuse program parser, results are in extra columns
        return self._parse_race_program_html(html, race_date)

    def _extract_track_name(self, div) -> str | None:
        """Try to find track name from context."""
        track_el = div.select_one(".track-name, [class*='track'], [class*='hipodrom']")
        if track_el:
            return track_el.get_text(strip=True)
        return None

    @staticmethod
    def _extract_number(text: str) -> int | None:
        """Extract first integer from text like '1. Koşu' → 1 or '1200m' → 1200."""
        import re
        match = re.search(r"\d+", text)
        return int(match.group()) if match else None
```

- [ ] **Step 5: Update scraper __init__.py**

```python
# src/ganyan/scraper/__init__.py
from ganyan.scraper.tjk_api import TJKClient
from ganyan.scraper.parser import (
    RawRaceCard, RawHorseEntry, ParsedRaceCard, ParsedHorseEntry,
    parse_race_card,
)

__all__ = [
    "TJKClient",
    "RawRaceCard", "RawHorseEntry", "ParsedRaceCard", "ParsedHorseEntry",
    "parse_race_card",
]
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_scraper/ -v`
Expected: all tests PASS

- [ ] **Step 7: Test against live TJK site**

Write a quick manual test to validate the scraper works against the real site. Run it interactively — do NOT commit this file:

```python
# scratch_test_live.py (do not commit)
import asyncio
from datetime import date
from ganyan.scraper.tjk_api import TJKClient

async def main():
    client = TJKClient(delay=2.0)
    cards = await client.get_race_card(date.today())
    print(f"Found {len(cards)} races")
    for card in cards:
        print(f"  Race {card.race_number}: {len(card.horses)} horses at {card.track_name}")
        for h in card.horses[:2]:
            print(f"    {h.name} - jockey: {h.jockey}, weight: {h.weight_kg}")
    await client.close()

asyncio.run(main())
```

```bash
uv run python scratch_test_live.py
```

**Expect:** This will likely need adjustments to CSS selectors based on actual TJK HTML structure. Update `_parse_race_program_html`, `_parse_single_race_div`, and `_parse_horse_row` selectors based on what you see. This is the critical step — the mock HTML in tests is a placeholder for the real structure.

- [ ] **Step 8: Update tests to match real TJK HTML structure**

After Step 7, update `SAMPLE_RACE_PROGRAM_HTML` in `test_tjk_api.py` to match the actual HTML structure from TJK. Update CSS selectors in `tjk_api.py` accordingly.

- [ ] **Step 9: Run all tests again**

Run: `uv run pytest tests/ -v`
Expected: all PASS

- [ ] **Step 10: Commit**

```bash
git add src/ganyan/scraper/ tests/test_scraper/
git commit -m "feat: TJK scraper client with HTML parsing and mocked tests"
```

---

### Task 5: Scraper — Database Integration + Backfill

**Files:**
- Create: `src/ganyan/scraper/backfill.py`
- Create: `tests/test_scraper/test_backfill.py`

- [ ] **Step 1: Write backfill/storage tests**

```python
# tests/test_scraper/test_backfill.py
import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Track, Race, Horse, RaceEntry, ScrapeLog, RaceStatus, ScrapeStatus
from ganyan.scraper.parser import RawRaceCard, RawHorseEntry, parse_race_card
from ganyan.scraper.backfill import store_race_card, get_scraped_dates, BackfillManager


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _make_raw_card(track="İstanbul", race_num=1, horse_name="Karayel"):
    return RawRaceCard(
        track_name=track,
        date=date(2026, 4, 5),
        race_number=race_num,
        distance_meters=1400,
        surface="Çim",
        race_type="Handikap",
        horses=[
            RawHorseEntry(
                name=horse_name, age=4, origin="TR",
                gate_number=1, jockey="Ahmet Çelik", weight_kg=57.0,
                hp=85.5, kgs=21, s20=12.5, eid="1.30.45",
                gny=8.3, agf=5.2, last_six="1 3 2 4 1 2",
            ),
        ],
    )


def test_store_race_card_creates_all_records(db_session):
    raw = _make_raw_card()
    parsed = parse_race_card(raw)
    store_race_card(db_session, parsed)
    db_session.commit()

    tracks = db_session.query(Track).all()
    assert len(tracks) == 1
    assert tracks[0].name == "İstanbul"

    races = db_session.query(Race).all()
    assert len(races) == 1
    assert races[0].race_number == 1

    horses = db_session.query(Horse).all()
    assert len(horses) == 1
    assert horses[0].name == "Karayel"

    entries = db_session.query(RaceEntry).all()
    assert len(entries) == 1
    assert float(entries[0].hp) == 85.5


def test_store_race_card_is_idempotent(db_session):
    raw = _make_raw_card()
    parsed = parse_race_card(raw)
    store_race_card(db_session, parsed)
    db_session.commit()
    store_race_card(db_session, parsed)
    db_session.commit()

    assert db_session.query(Track).count() == 1
    assert db_session.query(Race).count() == 1
    assert db_session.query(Horse).count() == 1
    assert db_session.query(RaceEntry).count() == 1


def test_store_race_card_reuses_existing_horse(db_session):
    raw1 = _make_raw_card(race_num=1, horse_name="Karayel")
    raw2 = _make_raw_card(race_num=2, horse_name="Karayel")
    store_race_card(db_session, parse_race_card(raw1))
    store_race_card(db_session, parse_race_card(raw2))
    db_session.commit()

    assert db_session.query(Horse).count() == 1
    assert db_session.query(RaceEntry).count() == 2


def test_get_scraped_dates(db_session):
    log = ScrapeLog(date=date(2026, 4, 5), track="İstanbul", status=ScrapeStatus.success)
    db_session.add(log)
    db_session.commit()

    scraped = get_scraped_dates(db_session)
    assert date(2026, 4, 5) in scraped
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scraper/test_backfill.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement backfill module**

```python
# src/ganyan/scraper/backfill.py
import logging
from datetime import date, timedelta

from sqlalchemy.orm import Session

from ganyan.db.models import Track, Race, Horse, RaceEntry, ScrapeLog, RaceStatus, ScrapeStatus
from ganyan.scraper.parser import ParsedRaceCard

logger = logging.getLogger(__name__)


def get_or_create_track(session: Session, name: str) -> Track:
    """Get existing track or create new one."""
    track = session.query(Track).filter(Track.name == name).first()
    if not track:
        track = Track(name=name)
        session.add(track)
        session.flush()
    return track


def get_or_create_horse(session: Session, name: str, **kwargs) -> Horse:
    """Get existing horse or create new one."""
    horse = session.query(Horse).filter(Horse.name == name).first()
    if not horse:
        horse = Horse(name=name, **kwargs)
        session.add(horse)
        session.flush()
    else:
        # Update mutable fields if provided
        for key in ("age", "owner", "trainer"):
            if key in kwargs and kwargs[key] is not None:
                setattr(horse, key, kwargs[key])
    return horse


def store_race_card(session: Session, parsed: ParsedRaceCard) -> Race:
    """Store a parsed race card in the database. Idempotent."""
    track = get_or_create_track(session, parsed.track_name)

    # Check if race already exists
    race = session.query(Race).filter(
        Race.track_id == track.id,
        Race.date == parsed.date,
        Race.race_number == parsed.race_number,
    ).first()

    if not race:
        race = Race(
            track_id=track.id,
            date=parsed.date,
            race_number=parsed.race_number,
            distance_meters=parsed.distance_meters,
            surface=parsed.surface,
            race_type=parsed.race_type,
            horse_type=parsed.horse_type,
            weight_rule=parsed.weight_rule,
            status=RaceStatus.scheduled,
        )
        session.add(race)
        session.flush()

    for h in parsed.horses:
        horse = get_or_create_horse(
            session, h.name, age=h.age, origin=h.origin,
            owner=h.owner, trainer=h.trainer,
        )

        # Check if entry already exists
        existing = session.query(RaceEntry).filter(
            RaceEntry.race_id == race.id,
            RaceEntry.horse_id == horse.id,
        ).first()

        if not existing:
            entry = RaceEntry(
                race_id=race.id,
                horse_id=horse.id,
                gate_number=h.gate_number,
                jockey=h.jockey,
                weight_kg=h.weight_kg,
                hp=h.hp,
                kgs=h.kgs,
                s20=h.s20,
                eid=h.eid,
                gny=h.gny,
                agf=h.agf,
                last_six=h.last_six,
                finish_position=h.finish_position,
            )
            session.add(entry)

    return race


def update_race_results(session: Session, parsed: ParsedRaceCard) -> Race | None:
    """Update an existing race with results. Returns None if race not found."""
    track = session.query(Track).filter(Track.name == parsed.track_name).first()
    if not track:
        return None

    race = session.query(Race).filter(
        Race.track_id == track.id,
        Race.date == parsed.date,
        Race.race_number == parsed.race_number,
    ).first()

    if not race:
        return None

    for h in parsed.horses:
        horse = session.query(Horse).filter(Horse.name == h.name).first()
        if not horse:
            continue

        entry = session.query(RaceEntry).filter(
            RaceEntry.race_id == race.id,
            RaceEntry.horse_id == horse.id,
        ).first()

        if entry and h.finish_position is not None:
            entry.finish_position = h.finish_position
            entry.finish_time = h.finish_time

    race.status = RaceStatus.resulted
    return race


def get_scraped_dates(session: Session) -> set[date]:
    """Get set of dates that have been successfully scraped."""
    logs = session.query(ScrapeLog.date).filter(
        ScrapeLog.status == ScrapeStatus.success
    ).distinct().all()
    return {log[0] for log in logs}


def log_scrape(session: Session, scrape_date: date, track: str, status: ScrapeStatus):
    """Record a scrape attempt."""
    log = ScrapeLog(date=scrape_date, track=track, status=status)
    session.add(log)


class BackfillManager:
    """Manages incremental historical data loading."""

    def __init__(self, session: Session, tjk_client):
        self.session = session
        self.client = tjk_client

    async def backfill(self, from_date: date, to_date: date | None = None):
        """Backfill race data from from_date to to_date (default: yesterday).

        Processes in reverse chronological order. Skips already-scraped dates.
        """
        if to_date is None:
            to_date = date.today() - timedelta(days=1)

        scraped = get_scraped_dates(self.session)

        # Generate dates in reverse order (recent first)
        current = to_date
        while current >= from_date:
            if current not in scraped:
                await self._scrape_date(current)
            current -= timedelta(days=1)

    async def _scrape_date(self, scrape_date: date):
        """Scrape all races for a single date."""
        logger.info(f"Scraping {scrape_date}")
        try:
            cards = await self.client.get_race_card(scrape_date)
            if not cards:
                log_scrape(self.session, scrape_date, "all", ScrapeStatus.skipped)
                self.session.commit()
                return

            for raw_card in cards:
                from ganyan.scraper.parser import parse_race_card
                parsed = parse_race_card(raw_card)
                store_race_card(self.session, parsed)
                log_scrape(self.session, scrape_date, parsed.track_name, ScrapeStatus.success)

            self.session.commit()
            logger.info(f"Scraped {len(cards)} races for {scrape_date}")

        except Exception as e:
            logger.error(f"Failed to scrape {scrape_date}: {e}")
            self.session.rollback()
            log_scrape(self.session, scrape_date, "all", ScrapeStatus.failed)
            self.session.commit()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_scraper/test_backfill.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ganyan/scraper/backfill.py tests/test_scraper/test_backfill.py
git commit -m "feat: race data storage and incremental backfill manager"
```

---

### Task 6: Prediction Engine — Feature Extraction

**Files:**
- Create: `src/ganyan/predictor/features.py`
- Create: `tests/test_predictor/test_features.py`

- [ ] **Step 1: Write feature extraction tests**

```python
# tests/test_predictor/test_features.py
import pytest
from ganyan.predictor.features import (
    compute_speed_figure,
    compute_form_cycle,
    compute_weight_delta,
    compute_rest_fitness,
    compute_class_indicator,
    extract_features,
    HorseFeatures,
)


def test_compute_speed_figure():
    # 1400m in 90.45s = ~15.48 m/s
    speed = compute_speed_figure(eid_seconds=90.45, distance_meters=1400)
    assert 15.0 < speed < 16.0


def test_compute_speed_figure_none():
    assert compute_speed_figure(eid_seconds=None, distance_meters=1400) is None
    assert compute_speed_figure(eid_seconds=90.0, distance_meters=None) is None


def test_compute_form_cycle_improving():
    # Positions getting better (lower = better), recent has highest weight
    positions = [6, 5, 4, 3, 2, 1]  # oldest to newest
    score = compute_form_cycle(positions)
    assert score > 0.7  # strong improving form


def test_compute_form_cycle_declining():
    positions = [1, 2, 3, 4, 5, 6]  # getting worse
    score = compute_form_cycle(positions)
    assert score < 0.4  # poor form


def test_compute_form_cycle_empty():
    assert compute_form_cycle([]) is None
    assert compute_form_cycle(None) is None


def test_compute_form_cycle_with_nones():
    positions = [2, None, 3, None, 1, 4]
    score = compute_form_cycle(positions)
    assert score is not None  # should handle missing values


def test_compute_weight_delta():
    # Horse at 55kg, field average 58kg → lighter = positive
    delta = compute_weight_delta(horse_weight=55.0, field_avg_weight=58.0)
    assert delta > 0  # positive = advantage


def test_compute_weight_delta_heavy():
    delta = compute_weight_delta(horse_weight=62.0, field_avg_weight=58.0)
    assert delta < 0  # negative = disadvantage


def test_compute_rest_fitness_optimal():
    # 21 days is optimal per spec
    fitness = compute_rest_fitness(kgs=21)
    assert fitness > 0.8


def test_compute_rest_fitness_too_long():
    fitness = compute_rest_fitness(kgs=120)
    assert fitness < 0.5


def test_compute_rest_fitness_too_short():
    fitness = compute_rest_fitness(kgs=3)
    assert fitness < 0.6


def test_compute_rest_fitness_none():
    assert compute_rest_fitness(None) is None


def test_compute_class_indicator():
    # HP=85 in a race with avg HP=80 → stepping down (advantage)
    indicator = compute_class_indicator(hp=85.0, field_avg_hp=80.0)
    assert indicator > 0


def test_extract_features():
    features = extract_features(
        eid_seconds=90.45,
        distance_meters=1400,
        last_six_parsed=[2, 4, 1, 3, 2, 1],
        weight_kg=57.0,
        field_avg_weight=58.0,
        kgs=21,
        hp=85.0,
        field_avg_hp=80.0,
    )
    assert isinstance(features, HorseFeatures)
    assert features.speed_figure is not None
    assert features.form_cycle is not None
    assert features.weight_delta is not None
    assert features.rest_fitness is not None
    assert features.class_indicator is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_predictor/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement feature extraction**

```python
# src/ganyan/predictor/features.py
from dataclasses import dataclass

import numpy as np


@dataclass
class HorseFeatures:
    speed_figure: float | None = None
    form_cycle: float | None = None
    weight_delta: float | None = None
    rest_fitness: float | None = None
    class_indicator: float | None = None
    track_affinity: float | None = None  # requires history, filled later


def compute_speed_figure(eid_seconds: float | None, distance_meters: int | None) -> float | None:
    """Normalize EİD into a speed figure (meters per second)."""
    if eid_seconds is None or distance_meters is None or eid_seconds <= 0:
        return None
    return distance_meters / eid_seconds


def compute_form_cycle(positions: list[int | None] | None) -> float | None:
    """Compute form score from recent finishing positions.

    Uses exponential decay weighting (most recent = highest weight).
    Returns 0-1 where 1 = best form.
    """
    if not positions:
        return None

    # Filter out None values while keeping indices for weighting
    valid = [(i, p) for i, p in enumerate(positions) if p is not None]
    if not valid:
        return None

    n = len(positions)
    scores = []
    weights = []
    for i, pos in valid:
        # Weight: exponential decay from oldest (index 0) to newest (index n-1)
        weight = np.exp((i - n + 1) * 0.5)  # newest gets weight ~1.0
        # Convert position to score: 1st → 1.0, 2nd → 0.85, ..., 10th+ → low
        score = max(0.0, 1.0 - (pos - 1) * 0.15)
        scores.append(score)
        weights.append(weight)

    weights = np.array(weights)
    scores = np.array(scores)
    return float(np.average(scores, weights=weights))


def compute_weight_delta(horse_weight: float | None, field_avg_weight: float | None) -> float | None:
    """Compute weight advantage/disadvantage relative to field.

    Positive = lighter than average (advantage).
    """
    if horse_weight is None or field_avg_weight is None:
        return None
    return (field_avg_weight - horse_weight) / field_avg_weight


def compute_rest_fitness(kgs: int | None) -> float | None:
    """Compute fitness score based on days since last race.

    Peak fitness around 14-28 days. Penalize extremes.
    Uses a Gaussian-like curve centered on 21 days.
    """
    if kgs is None:
        return None
    optimal = 21.0
    sigma = 20.0
    return float(np.exp(-((kgs - optimal) ** 2) / (2 * sigma ** 2)))


def compute_class_indicator(hp: float | None, field_avg_hp: float | None) -> float | None:
    """Compute class advantage. Positive = horse has higher HP than field average (dropping in class)."""
    if hp is None or field_avg_hp is None or field_avg_hp == 0:
        return None
    return (hp - field_avg_hp) / field_avg_hp


def extract_features(
    eid_seconds: float | None = None,
    distance_meters: int | None = None,
    last_six_parsed: list[int | None] | None = None,
    weight_kg: float | None = None,
    field_avg_weight: float | None = None,
    kgs: int | None = None,
    hp: float | None = None,
    field_avg_hp: float | None = None,
) -> HorseFeatures:
    """Extract all features for a single horse."""
    return HorseFeatures(
        speed_figure=compute_speed_figure(eid_seconds, distance_meters),
        form_cycle=compute_form_cycle(last_six_parsed),
        weight_delta=compute_weight_delta(weight_kg, field_avg_weight),
        rest_fitness=compute_rest_fitness(kgs),
        class_indicator=compute_class_indicator(hp, field_avg_hp),
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_predictor/test_features.py -v`
Expected: all 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/ganyan/predictor/features.py tests/test_predictor/test_features.py
git commit -m "feat: feature extraction — speed, form, weight, rest, class"
```

---

### Task 7: Prediction Engine — Bayesian Model

**Files:**
- Create: `src/ganyan/predictor/bayesian.py`
- Create: `tests/test_predictor/test_bayesian.py`
- Modify: `src/ganyan/predictor/__init__.py`

- [ ] **Step 1: Write Bayesian predictor tests**

```python
# tests/test_predictor/test_bayesian.py
import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import Base, Track, Race, Horse, RaceEntry, RaceStatus
from ganyan.predictor.bayesian import BayesianPredictor, Prediction


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def race_with_entries(db_session):
    """Create a race with 4 horses."""
    track = Track(name="İstanbul", city="İstanbul")
    db_session.add(track)
    db_session.flush()

    race = Race(
        track_id=track.id, date=date(2026, 4, 5), race_number=1,
        distance_meters=1400, surface="çim", status=RaceStatus.scheduled,
    )
    db_session.add(race)
    db_session.flush()

    horses_data = [
        ("Karayel", 4, 57.0, 85.5, 21, "1.30.45", "1 3 2 4 1 2", 12.5),
        ("Rüzgar", 3, 55.5, 78.0, 14, "1.31.20", "3 2 1 5 4 3", 10.8),
        ("Fırtına", 5, 59.0, 90.0, 35, "1.29.80", "2 1 1 2 3 1", 14.0),
        ("Yıldız", 4, 56.0, 82.0, 7, "1.32.00", "5 6 4 3 5 4", 9.5),
    ]

    for name, age, weight, hp, kgs, eid, last_six, s20 in horses_data:
        horse = Horse(name=name, age=age)
        db_session.add(horse)
        db_session.flush()
        entry = RaceEntry(
            race_id=race.id, horse_id=horse.id,
            gate_number=horses_data.index((name, age, weight, hp, kgs, eid, last_six, s20)) + 1,
            jockey=f"Jokey {name}", weight_kg=weight, hp=hp, kgs=kgs,
            eid=eid, last_six=last_six, s20=s20,
        )
        db_session.add(entry)

    db_session.commit()
    return race.id


def test_predict_returns_predictions_for_all_horses(db_session, race_with_entries):
    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race_with_entries)
    assert len(predictions) == 4
    assert all(isinstance(p, Prediction) for p in predictions)


def test_probabilities_sum_to_100(db_session, race_with_entries):
    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race_with_entries)
    total = sum(p.probability for p in predictions)
    assert abs(total - 100.0) < 0.01


def test_predictions_are_sorted_by_probability(db_session, race_with_entries):
    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race_with_entries)
    probs = [p.probability for p in predictions]
    assert probs == sorted(probs, reverse=True)


def test_predictions_have_contributing_factors(db_session, race_with_entries):
    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race_with_entries)
    for p in predictions:
        assert isinstance(p.contributing_factors, dict)
        assert len(p.contributing_factors) > 0


def test_predictions_have_confidence(db_session, race_with_entries):
    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race_with_entries)
    for p in predictions:
        assert 0.0 <= p.confidence <= 1.0


def test_predict_empty_race(db_session):
    """Race with no entries should return empty list."""
    track = Track(name="Ankara", city="Ankara")
    db_session.add(track)
    db_session.flush()
    race = Race(
        track_id=track.id, date=date(2026, 4, 5), race_number=1,
        distance_meters=1200, surface="kum", status=RaceStatus.scheduled,
    )
    db_session.add(race)
    db_session.commit()

    predictor = BayesianPredictor(db_session)
    predictions = predictor.predict(race.id)
    assert predictions == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_predictor/test_bayesian.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Bayesian predictor**

```python
# src/ganyan/predictor/bayesian.py
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ganyan.db.models import Race, RaceEntry
from ganyan.predictor.features import extract_features, HorseFeatures
from ganyan.scraper.parser import parse_eid_to_seconds, parse_last_six


@dataclass
class Prediction:
    horse_id: int
    horse_name: str
    probability: float  # 0-100
    confidence: float  # 0-1
    contributing_factors: dict = field(default_factory=dict)


class BayesianPredictor:
    """Empirical Bayesian prediction model.

    Prior: 1/N (uniform across field).
    Likelihood: feature-based adjustments.
    Posterior: normalized to sum to 100%.
    """

    # Feature weights for likelihood computation
    FEATURE_WEIGHTS = {
        "speed_figure": 0.30,
        "form_cycle": 0.25,
        "weight_delta": 0.15,
        "rest_fitness": 0.15,
        "class_indicator": 0.15,
    }

    def __init__(self, session: Session):
        self.session = session

    def predict(self, race_id: int) -> list[Prediction]:
        """Generate predictions for all horses in a race."""
        race = self.session.get(Race, race_id)
        if not race:
            return []

        entries = self.session.query(RaceEntry).filter(RaceEntry.race_id == race_id).all()
        if not entries:
            return []

        n = len(entries)
        prior = 1.0 / n

        # Compute field averages for relative features
        weights = [float(e.weight_kg) for e in entries if e.weight_kg is not None]
        hps = [float(e.hp) for e in entries if e.hp is not None]
        field_avg_weight = sum(weights) / len(weights) if weights else None
        field_avg_hp = sum(hps) / len(hps) if hps else None

        # Extract features and compute raw scores
        raw_scores = []
        for entry in entries:
            features = extract_features(
                eid_seconds=parse_eid_to_seconds(entry.eid),
                distance_meters=race.distance_meters,
                last_six_parsed=parse_last_six(entry.last_six),
                weight_kg=float(entry.weight_kg) if entry.weight_kg is not None else None,
                field_avg_weight=field_avg_weight,
                kgs=entry.kgs,
                hp=float(entry.hp) if entry.hp is not None else None,
                field_avg_hp=field_avg_hp,
            )

            likelihood, factors, confidence = self._compute_likelihood(features)
            posterior = prior * likelihood
            raw_scores.append((entry, posterior, factors, confidence))

        # Normalize to sum to 100%
        total = sum(score for _, score, _, _ in raw_scores)
        if total <= 0:
            # Fallback: uniform distribution
            total = n
            raw_scores = [(e, 1.0, f, c) for e, _, f, c in raw_scores]

        predictions = []
        for entry, score, factors, confidence in raw_scores:
            prob = (score / total) * 100.0
            predictions.append(Prediction(
                horse_id=entry.horse_id,
                horse_name=entry.horse.name,
                probability=round(prob, 2),
                confidence=round(confidence, 2),
                contributing_factors=factors,
            ))

        # Sort by probability descending
        predictions.sort(key=lambda p: p.probability, reverse=True)
        return predictions

    def _compute_likelihood(self, features: HorseFeatures) -> tuple[float, dict, float]:
        """Compute likelihood ratio from features.

        Returns (likelihood, contributing_factors, confidence).
        """
        likelihood = 1.0
        factors = {}
        available_features = 0
        total_features = len(self.FEATURE_WEIGHTS)

        for feature_name, weight in self.FEATURE_WEIGHTS.items():
            value = getattr(features, feature_name, None)
            if value is None:
                continue

            available_features += 1

            # Convert feature value to a likelihood multiplier
            # Values > 0 boost, values < 0 penalize, 0 = neutral
            if feature_name == "speed_figure":
                # Speed: normalize to 0-1 range (typical range 12-18 m/s)
                normalized = min(1.0, max(0.0, (value - 12.0) / 6.0))
                multiplier = 0.5 + normalized  # range 0.5 to 1.5
            elif feature_name == "form_cycle":
                # Already 0-1
                multiplier = 0.5 + value  # range 0.5 to 1.5
            elif feature_name in ("weight_delta", "class_indicator"):
                # Can be negative or positive, typically -0.1 to 0.1
                multiplier = 1.0 + value * 3.0  # amplify small differences
            elif feature_name == "rest_fitness":
                # Already 0-1
                multiplier = 0.5 + value  # range 0.5 to 1.5
            else:
                multiplier = 1.0

            # Apply weight
            weighted_multiplier = 1.0 + (multiplier - 1.0) * weight
            likelihood *= weighted_multiplier
            factors[feature_name] = round(multiplier - 1.0, 3)  # impact relative to neutral

        # Confidence based on data availability
        confidence = available_features / total_features if total_features > 0 else 0.0

        return likelihood, factors, confidence
```

- [ ] **Step 4: Update predictor __init__.py**

```python
# src/ganyan/predictor/__init__.py
from ganyan.predictor.bayesian import BayesianPredictor, Prediction
from ganyan.predictor.features import extract_features, HorseFeatures

__all__ = ["BayesianPredictor", "Prediction", "extract_features", "HorseFeatures"]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_predictor/ -v`
Expected: all tests PASS (features + bayesian)

- [ ] **Step 6: Commit**

```bash
git add src/ganyan/predictor/ tests/test_predictor/
git commit -m "feat: empirical Bayesian prediction model with feature-based likelihoods"
```

---

### Task 8: CLI

**Files:**
- Create: `src/ganyan/cli/main.py`
- Create: `tests/test_cli/test_main.py`

- [ ] **Step 1: Write CLI tests**

```python
# tests/test_cli/test_main.py
from typer.testing import CliRunner

from ganyan.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ganyan" in result.output.lower() or "scrape" in result.output.lower()


def test_scrape_help():
    result = runner.invoke(app, ["scrape", "--help"])
    assert result.exit_code == 0
    assert "--today" in result.output


def test_predict_help():
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0


def test_races_help():
    result = runner.invoke(app, ["races", "--help"])
    assert result.exit_code == 0


def test_db_help():
    result = runner.invoke(app, ["db", "--help"])
    assert result.exit_code == 0
    assert "init" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CLI**

```python
# src/ganyan/cli/main.py
import asyncio
import logging
from datetime import date, datetime

import typer

from ganyan.config import get_settings

app = typer.Typer(name="ganyan", help="Turkish horse racing prediction system")
scrape_app = typer.Typer(help="Scrape race data from TJK")
predict_app = typer.Typer(help="Generate race predictions")
races_app = typer.Typer(help="View race information")
db_app = typer.Typer(help="Database management")

app.add_typer(scrape_app, name="scrape")
app.add_typer(predict_app, name="predict")
app.add_typer(races_app, name="races")
app.add_typer(db_app, name="db")


def _setup_logging():
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@scrape_app.callback(invoke_without_command=True)
def scrape(
    today: bool = typer.Option(False, "--today", help="Fetch today's race cards"),
    results: bool = typer.Option(False, "--results", help="Fetch today's results"),
    backfill: bool = typer.Option(False, "--backfill", help="Backfill historical data"),
    from_date: str = typer.Option(None, "--from", help="Start date for backfill (YYYY-MM-DD)"),
):
    """Scrape race data from TJK."""
    _setup_logging()
    from ganyan.db import get_session
    from ganyan.scraper import TJKClient
    from ganyan.scraper.backfill import BackfillManager, store_race_card, log_scrape
    from ganyan.scraper.parser import parse_race_card
    from ganyan.db.models import ScrapeStatus

    settings = get_settings()
    session = get_session()
    client = TJKClient(base_url=settings.tjk_base_url, delay=settings.scrape_delay)

    async def _run():
        try:
            if today:
                typer.echo(f"Fetching today's race cards ({date.today()})...")
                cards = await client.get_race_card(date.today())
                for raw in cards:
                    parsed = parse_race_card(raw)
                    store_race_card(session, parsed)
                    log_scrape(session, date.today(), parsed.track_name, ScrapeStatus.success)
                session.commit()
                typer.echo(f"Stored {len(cards)} races.")

            elif results:
                typer.echo(f"Fetching today's results ({date.today()})...")
                result_cards = await client.get_race_results(date.today())
                from ganyan.scraper.backfill import update_race_results
                for raw in result_cards:
                    parsed = parse_race_card(raw)
                    update_race_results(session, parsed)
                session.commit()
                typer.echo(f"Updated {len(result_cards)} race results.")

            elif backfill:
                if not from_date:
                    typer.echo("Error: --from is required with --backfill", err=True)
                    raise typer.Exit(1)
                start = datetime.strptime(from_date, "%Y-%m-%d").date()
                typer.echo(f"Backfilling from {start} to yesterday...")
                manager = BackfillManager(session, client)
                await manager.backfill(start)
                typer.echo("Backfill complete.")

            else:
                typer.echo("Specify --today, --results, or --backfill. Use --help for details.")

        finally:
            await client.close()
            session.close()

    asyncio.run(_run())


@predict_app.callback(invoke_without_command=True)
def predict(
    race_id: int = typer.Argument(None, help="Race ID to predict"),
    today: bool = typer.Option(False, "--today", help="Predict all today's races"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Generate predictions for races."""
    _setup_logging()
    from ganyan.db import get_session
    from ganyan.db.models import Race
    from ganyan.predictor import BayesianPredictor

    session = get_session()

    try:
        if today:
            races = session.query(Race).filter(Race.date == date.today()).all()
            if not races:
                typer.echo("No races found for today.")
                return
            race_ids = [r.id for r in races]
        elif race_id:
            race_ids = [race_id]
        else:
            typer.echo("Specify a race_id or --today. Use --help for details.")
            return

        predictor = BayesianPredictor(session)

        for rid in race_ids:
            race = session.get(Race, rid)
            predictions = predictor.predict(rid)

            if json_output:
                import json
                data = [
                    {
                        "horse": p.horse_name,
                        "probability": p.probability,
                        "confidence": p.confidence,
                        "factors": p.contributing_factors,
                    }
                    for p in predictions
                ]
                typer.echo(json.dumps({"race_id": rid, "predictions": data}, indent=2, ensure_ascii=False))
            else:
                track_name = race.track.name if race and race.track else "?"
                typer.echo(f"\n{'='*50}")
                typer.echo(f"Race {rid}: {track_name} - Race {race.race_number} ({race.date})")
                typer.echo(f"{'='*50}")
                for i, p in enumerate(predictions, 1):
                    typer.echo(
                        f"  {i}. {p.horse_name:20s} "
                        f"Prob: {p.probability:5.1f}%  "
                        f"Conf: {p.confidence:.0%}"
                    )

    finally:
        session.close()


@races_app.callback(invoke_without_command=True)
def races(
    today: bool = typer.Option(False, "--today", help="Show today's races"),
    race_date: str = typer.Option(None, "--date", help="Show races for date (YYYY-MM-DD)"),
):
    """View race information."""
    from ganyan.db import get_session
    from ganyan.db.models import Race

    session = get_session()

    try:
        if today:
            target = date.today()
        elif race_date:
            target = datetime.strptime(race_date, "%Y-%m-%d").date()
        else:
            typer.echo("Specify --today or --date. Use --help for details.")
            return

        race_list = session.query(Race).filter(Race.date == target).order_by(Race.race_number).all()

        if not race_list:
            typer.echo(f"No races found for {target}.")
            return

        typer.echo(f"\nRaces for {target}:")
        typer.echo("-" * 60)
        for r in race_list:
            track_name = r.track.name if r.track else "?"
            entry_count = len(r.entries)
            typer.echo(
                f"  ID {r.id:4d} | Race {r.race_number:2d} | {track_name:12s} | "
                f"{r.distance_meters or '?':>5}m | {r.surface or '?':8s} | "
                f"{entry_count} horses | {r.status.value}"
            )

    finally:
        session.close()


@db_app.command("init")
def db_init():
    """Run database migrations."""
    import subprocess
    typer.echo("Running alembic upgrade head...")
    result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    if result.returncode == 0:
        typer.echo("Database initialized successfully.")
    else:
        typer.echo(f"Error: {result.stderr}", err=True)
        raise typer.Exit(1)


@db_app.command("reset")
def db_reset():
    """Drop all tables and recreate. WARNING: destroys all data."""
    confirm = typer.confirm("This will destroy ALL data. Are you sure?")
    if not confirm:
        typer.echo("Aborted.")
        return

    import subprocess
    typer.echo("Running alembic downgrade base...")
    subprocess.run(["alembic", "downgrade", "base"], capture_output=True, text=True)
    typer.echo("Running alembic upgrade head...")
    subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)
    typer.echo("Database reset complete.")
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli/test_main.py -v`
Expected: all 5 tests PASS

- [ ] **Step 5: Verify CLI entry point works**

```bash
uv run ganyan --help
uv run ganyan scrape --help
uv run ganyan predict --help
```

Expected: help text displayed for each command

- [ ] **Step 6: Commit**

```bash
git add src/ganyan/cli/ tests/test_cli/
git commit -m "feat: typer CLI with scrape, predict, races, db commands"
```

---

### Task 9: Web App — Flask + HTMX

**Files:**
- Create: `src/ganyan/web/app.py`
- Create: `src/ganyan/web/routes.py`
- Create: `src/ganyan/web/templates/base.html`
- Create: `src/ganyan/web/templates/index.html`
- Create: `src/ganyan/web/templates/races.html`
- Create: `src/ganyan/web/templates/predict.html`
- Create: `src/ganyan/web/templates/history.html`
- Create: `tests/test_web/test_routes.py`

- [ ] **Step 1: Write Flask route tests**

```python
# tests/test_web/test_routes.py
import pytest
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ganyan.db.models import Base, Track, Race, Horse, RaceEntry, RaceStatus
from ganyan.web.app import create_app


@pytest.fixture
def app():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)

    # Seed test data
    with factory() as session:
        track = Track(name="İstanbul", city="İstanbul")
        session.add(track)
        session.flush()
        race = Race(
            track_id=track.id, date=date.today(), race_number=1,
            distance_meters=1400, surface="çim", status=RaceStatus.scheduled,
        )
        session.add(race)
        session.flush()
        horse = Horse(name="Karayel", age=4)
        session.add(horse)
        session.flush()
        entry = RaceEntry(
            race_id=race.id, horse_id=horse.id, gate_number=1,
            jockey="Ahmet Çelik", weight_kg=57.0, hp=85.5, kgs=21,
            eid="1.30.45", last_six="1 3 2 4 1 2",
        )
        session.add(entry)
        session.commit()

    flask_app = create_app(session_factory=factory)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Ganyan" in response.data.decode()


def test_races_today(client):
    response = client.get(f"/races/{date.today().isoformat()}")
    assert response.status_code == 200


def test_races_json(client):
    response = client.get(
        f"/races/{date.today().isoformat()}",
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_predict_race(client):
    response = client.get("/races/1/predict")
    assert response.status_code == 200


def test_predict_race_json(client):
    response = client.get(
        "/races/1/predict",
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data


def test_predict_nonexistent_race(client):
    response = client.get("/races/999/predict")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web/test_routes.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Flask app factory**

```python
# src/ganyan/web/app.py
from flask import Flask
from sqlalchemy.orm import sessionmaker

from ganyan.config import get_settings
from ganyan.db.session import get_session_factory


def create_app(session_factory: sessionmaker | None = None) -> Flask:
    """Flask application factory."""
    app = Flask(__name__)
    settings = get_settings()

    app.config["SECRET_KEY"] = "dev"  # Override via env in production

    # Store session factory for routes to use
    if session_factory is None:
        session_factory = get_session_factory()
    app.config["SESSION_FACTORY"] = session_factory

    # Register routes
    from ganyan.web.routes import bp
    app.register_blueprint(bp)

    return app


def run():
    """Run the Flask development server."""
    settings = get_settings()
    app = create_app()
    app.run(host="0.0.0.0", port=settings.flask_port, debug=settings.flask_debug)
```

- [ ] **Step 4: Implement routes**

```python
# src/ganyan/web/routes.py
from datetime import date, datetime

from flask import Blueprint, render_template, request, jsonify, abort

from ganyan.db.models import Race, RaceStatus
from ganyan.predictor import BayesianPredictor

bp = Blueprint("main", __name__)


def _get_session():
    from flask import current_app
    factory = current_app.config["SESSION_FACTORY"]
    return factory()


def _wants_json():
    return request.accept_mimetypes.best_match(["application/json", "text/html"]) == "application/json"


@bp.route("/")
def index():
    session = _get_session()
    try:
        today_races = session.query(Race).filter(Race.date == date.today()).order_by(Race.race_number).all()
        recent_races = (
            session.query(Race)
            .filter(Race.status == RaceStatus.resulted)
            .order_by(Race.date.desc(), Race.race_number.desc())
            .limit(10)
            .all()
        )
        return render_template("index.html", today_races=today_races, recent_races=recent_races)
    finally:
        session.close()


@bp.route("/races/<race_date>")
def races_by_date(race_date: str):
    session = _get_session()
    try:
        target = datetime.strptime(race_date, "%Y-%m-%d").date()
        race_list = session.query(Race).filter(Race.date == target).order_by(Race.race_number).all()

        if _wants_json():
            return jsonify([
                {
                    "id": r.id,
                    "race_number": r.race_number,
                    "track": r.track.name if r.track else None,
                    "distance": r.distance_meters,
                    "surface": r.surface,
                    "status": r.status.value,
                    "entries": len(r.entries),
                }
                for r in race_list
            ])

        return render_template("races.html", races=race_list, race_date=target)
    finally:
        session.close()


@bp.route("/races/<int:race_id>/predict")
def predict_race(race_id: int):
    session = _get_session()
    try:
        race = session.get(Race, race_id)
        if not race:
            if _wants_json():
                return jsonify({"error": "Race not found"}), 404
            abort(404)

        predictor = BayesianPredictor(session)
        predictions = predictor.predict(race_id)

        if _wants_json():
            return jsonify({
                "race_id": race_id,
                "predictions": [
                    {
                        "horse": p.horse_name,
                        "probability": p.probability,
                        "confidence": p.confidence,
                        "factors": p.contributing_factors,
                    }
                    for p in predictions
                ],
            })

        return render_template("predict.html", race=race, predictions=predictions)
    finally:
        session.close()


@bp.route("/history")
def history():
    session = _get_session()
    try:
        resulted = (
            session.query(Race)
            .filter(Race.status == RaceStatus.resulted)
            .order_by(Race.date.desc())
            .limit(50)
            .all()
        )
        return render_template("history.html", races=resulted)
    finally:
        session.close()


@bp.route("/scrape/today", methods=["POST"])
def scrape_today():
    """Trigger a manual scrape for today's races."""
    import asyncio
    from ganyan.config import get_settings
    from ganyan.scraper import TJKClient, parse_race_card
    from ganyan.scraper.backfill import store_race_card, log_scrape
    from ganyan.db.models import ScrapeStatus

    settings = get_settings()
    session = _get_session()

    async def _do_scrape():
        client = TJKClient(base_url=settings.tjk_base_url, delay=settings.scrape_delay)
        try:
            cards = await client.get_race_card(date.today())
            for raw in cards:
                parsed = parse_race_card(raw)
                store_race_card(session, parsed)
                log_scrape(session, date.today(), parsed.track_name, ScrapeStatus.success)
            session.commit()
            return len(cards)
        finally:
            await client.close()
            session.close()

    try:
        count = asyncio.run(_do_scrape())
        if _wants_json():
            return jsonify({"success": True, "races_scraped": count})
        return render_template("index.html", flash_message=f"{count} yarış güncellendi.")
    except Exception as e:
        if _wants_json():
            return jsonify({"success": False, "error": str(e)}), 500
        return render_template("index.html", flash_message=f"Hata: {e}")
```

- [ ] **Step 5: Create base template**

```html
<!-- src/ganyan/web/templates/base.html -->
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Ganyan{% endblock %} - At Yarışı Tahmin</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://unpkg.com/htmx.org@1.9.12"></script>
    <style>
        .prediction-bar {
            height: 24px;
            background: linear-gradient(90deg, #198754, #ffc107, #dc3545);
            border-radius: 4px;
        }
        .confidence-badge {
            font-size: 0.75rem;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">Ganyan</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/">Ana Sayfa</a>
                <a class="nav-link" href="/races/{{ today }}">Bugünkü Yarışlar</a>
                <a class="nav-link" href="/history">Geçmiş</a>
            </div>
        </div>
    </nav>
    <div class="container mt-4">
        {% if flash_message %}
        <div class="alert alert-info alert-dismissible fade show">
            {{ flash_message }}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
        {% endif %}
        {% block content %}{% endblock %}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

- [ ] **Step 6: Create page templates**

```html
<!-- src/ganyan/web/templates/index.html -->
{% extends "base.html" %}
{% block content %}
<div class="row">
    <div class="col-md-8">
        <h2>Bugünkü Yarışlar</h2>
        {% if today_races %}
        <div class="list-group">
            {% for race in today_races %}
            <a href="/races/{{ race.id }}/predict" class="list-group-item list-group-item-action">
                <div class="d-flex justify-content-between">
                    <strong>{{ race.race_number }}. Koşu - {{ race.track.name }}</strong>
                    <span class="badge bg-secondary">{{ race.entries|length }} at</span>
                </div>
                <small>{{ race.distance_meters }}m | {{ race.surface or '-' }} | {{ race.status.value }}</small>
            </a>
            {% endfor %}
        </div>
        {% else %}
        <p class="text-muted">Bugün için yarış bulunamadı.</p>
        <form hx-post="/scrape/today" hx-swap="innerHTML" hx-target="#scrape-result">
            <button class="btn btn-primary" type="submit">TJK'dan Veri Çek</button>
        </form>
        <div id="scrape-result" class="mt-2"></div>
        {% endif %}
    </div>
    <div class="col-md-4">
        <h4>Son Sonuçlar</h4>
        {% for race in recent_races %}
        <div class="card mb-2">
            <div class="card-body p-2">
                <small>{{ race.date }} - {{ race.track.name }} - {{ race.race_number }}. Koşu</small>
            </div>
        </div>
        {% endfor %}
    </div>
</div>
{% endblock %}
```

```html
<!-- src/ganyan/web/templates/races.html -->
{% extends "base.html" %}
{% block title %}{{ race_date }} Yarışları{% endblock %}
{% block content %}
<h2>{{ race_date }} Yarışları</h2>
{% if races %}
<div class="table-responsive">
    <table class="table table-hover">
        <thead>
            <tr>
                <th>Koşu</th><th>Pist</th><th>Mesafe</th><th>Zemin</th><th>At Sayısı</th><th>Durum</th><th></th>
            </tr>
        </thead>
        <tbody>
            {% for race in races %}
            <tr>
                <td>{{ race.race_number }}</td>
                <td>{{ race.track.name }}</td>
                <td>{{ race.distance_meters }}m</td>
                <td>{{ race.surface or '-' }}</td>
                <td>{{ race.entries|length }}</td>
                <td><span class="badge bg-{{ 'success' if race.status.value == 'resulted' else 'warning' }}">{{ race.status.value }}</span></td>
                <td><a href="/races/{{ race.id }}/predict" class="btn btn-sm btn-outline-primary">Tahmin</a></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% else %}
<p class="text-muted">Bu tarih için yarış bulunamadı.</p>
{% endif %}
{% endblock %}
```

```html
<!-- src/ganyan/web/templates/predict.html -->
{% extends "base.html" %}
{% block title %}Tahmin - {{ race.track.name }} {{ race.race_number }}. Koşu{% endblock %}
{% block content %}
<h2>{{ race.track.name }} - {{ race.race_number }}. Koşu ({{ race.date }})</h2>
<p>{{ race.distance_meters }}m | {{ race.surface or '-' }}</p>

{% if predictions %}
<div class="table-responsive">
    <table class="table">
        <thead>
            <tr>
                <th>#</th><th>At</th><th>Olasılık</th><th>Güven</th><th>Faktörler</th>
            </tr>
        </thead>
        <tbody>
            {% for p in predictions %}
            <tr>
                <td>{{ loop.index }}</td>
                <td><strong>{{ p.horse_name }}</strong></td>
                <td>
                    <div class="d-flex align-items-center gap-2">
                        <div class="progress flex-grow-1" style="height: 20px;">
                            <div class="progress-bar bg-success" style="width: {{ p.probability }}%">
                                {{ "%.1f"|format(p.probability) }}%
                            </div>
                        </div>
                    </div>
                </td>
                <td>
                    <span class="badge confidence-badge bg-{{ 'success' if p.confidence > 0.7 else 'warning' if p.confidence > 0.4 else 'secondary' }}">
                        {{ "%.0f"|format(p.confidence * 100) }}%
                    </span>
                </td>
                <td>
                    {% for factor, impact in p.contributing_factors.items() %}
                    <span class="badge bg-{{ 'success' if impact > 0 else 'danger' }} me-1">
                        {{ factor }}: {{ "%+.2f"|format(impact) }}
                    </span>
                    {% endfor %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% else %}
<p class="text-muted">Bu yarış için tahmin yapılamıyor.</p>
{% endif %}
{% endblock %}
```

```html
<!-- src/ganyan/web/templates/history.html -->
{% extends "base.html" %}
{% block title %}Geçmiş{% endblock %}
{% block content %}
<h2>Geçmiş Yarışlar</h2>
{% if races %}
<div class="list-group">
    {% for race in races %}
    <a href="/races/{{ race.id }}/predict" class="list-group-item list-group-item-action">
        <div class="d-flex justify-content-between">
            <span>{{ race.date }} - {{ race.track.name }} - {{ race.race_number }}. Koşu</span>
            <span class="badge bg-info">{{ race.entries|length }} at</span>
        </div>
    </a>
    {% endfor %}
</div>
{% else %}
<p class="text-muted">Henüz sonuçlanmış yarış yok.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_web/test_routes.py -v`
Expected: all 6 tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/ganyan/web/ tests/test_web/
git commit -m "feat: Flask web app with HTMX — dashboard, races, predictions, history"
```

---

### Task 10: Integration Test + CLAUDE.md Update

**Files:**
- Modify: `CLAUDE.md`
- Create: `Dockerfile`

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: all tests PASS

- [ ] **Step 2: Create Dockerfile (ready for cloud, not used locally)**

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
RUN uv sync --no-dev

COPY alembic.ini .
COPY alembic/ alembic/
COPY src/ src/

EXPOSE 5003

CMD ["uv", "run", "python", "-m", "ganyan.web.app"]
```

- [ ] **Step 3: Smoke test the full stack locally**

```bash
# 1. Ensure Postgres is running
docker compose up -d

# 2. Run migrations
uv run ganyan db init

# 3. Try scraping today's races
uv run ganyan scrape --today

# 4. List today's races
uv run ganyan races --today

# 5. Start web app
uv run python -c "from ganyan.web.app import run; run()"
# Visit http://localhost:5003 in browser
```

- [ ] **Step 4: Update CLAUDE.md**

Replace CLAUDE.md with updated content reflecting the new project structure:

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ganyan is a Turkish horse racing prediction system. It scrapes race data from TJK (Türkiye Jokey Kulübü), stores it in PostgreSQL, and generates Bayesian predictions served via CLI and Flask web app.

## Commands

\```bash
# Start PostgreSQL
docker compose up -d

# Install dependencies
uv sync --all-extras

# Run database migrations
uv run ganyan db init

# Scrape today's race cards
uv run ganyan scrape --today

# Scrape today's results
uv run ganyan scrape --results

# Backfill historical data
uv run ganyan scrape --backfill --from 2024-01-01

# Predict a specific race
uv run ganyan predict <race_id>

# Predict all today's races
uv run ganyan predict --today
uv run ganyan predict --today --json

# List races
uv run ganyan races --today
uv run ganyan races --date 2024-03-15

# Start web app (port 5003)
uv run python -c "from ganyan.web.app import run; run()"

# Run tests
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_predictor/test_bayesian.py::test_probabilities_sum_to_100 -v
\```

## Architecture

Three-layer service-oriented monorepo sharing PostgreSQL:

1. **Scraper** (`src/ganyan/scraper/`) — TJK website client, HTML parser, incremental backfill. `TJKClient` fetches race cards and results. `parser.py` normalizes raw data. `backfill.py` handles idempotent storage and historical loading.

2. **Predictor** (`src/ganyan/predictor/`) — Empirical Bayesian model. `features.py` extracts speed, form, weight, rest, class features. `bayesian.py` computes prior × likelihood → normalized probabilities with confidence scores.

3. **Web + CLI** (`src/ganyan/web/`, `src/ganyan/cli/`) — Flask app with HTMX (Bootstrap 5, Turkish UI). Typer CLI for terminal use. Both consume predictor and scraper directly.

### Data Flow

\```
TJK website → scraper/tjk_api.py → scraper/parser.py → scraper/backfill.py → PostgreSQL
                                                                                    ↓
CLI (ganyan predict) ← predictor/bayesian.py ← predictor/features.py ← race_entries
Flask (/races/<id>/predict) ←────────────────┘
\```

### Key Turkish Racing Metrics

- **HP** — Handikap Puanı (handicap points)
- **KGS** — Koşmama Gün Sayısı (days since last race; 14-28 optimal)
- **S20** — Son 20 yarış performansı (last 20 races performance)
- **EİD** — En İyi Derece (best time, stored as string, converted to seconds for computation)
- **GNY** — Günlük Nispi Yarış puanı (daily relative race score)
- **AGF** — Ağırlıklı Galibiyet Faktörü (weighted win factor)

### Database

PostgreSQL 16 via Docker Compose. SQLAlchemy 2.0 ORM + Alembic migrations. Tables: `tracks`, `races`, `horses`, `race_entries` (pre-race + post-race fields in one row), `scrape_log`.

### Config

`pydantic-settings` reads from `.env` file or environment variables. See `.env.example`.
\```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md Dockerfile
git commit -m "feat: Dockerfile, updated CLAUDE.md, integration smoke test"
```

- [ ] **Step 6: Run full test suite one final time**

```bash
uv run pytest tests/ -v
```

Expected: all tests PASS
