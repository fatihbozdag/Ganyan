# Ganyan Rebuild — Design Spec

**Date:** 2026-04-05
**Status:** Approved

## Goal

Rebuild Ganyan from scratch as a working Turkish horse racing prediction system that fetches live race card data from TJK before races start and generates Bayesian predictions. Scrape historical data incrementally for model improvement.

## Requirements

- Fetch today's race cards from TJK (pre-race: horses, stats, jockey, weight, metrics)
- Scrape historical race data incrementally (recent first, backfill over time)
- Bayesian prediction engine (start simple, evolve)
- CLI for quick terminal use, web app for detailed analysis
- PostgreSQL database
- Local-first, structured for cloud deployment later

## Architecture: Service-Oriented Monorepo

Three layers sharing a PostgreSQL database, all in one Python package:

1. **Scraper** — TJK API client + HTML fallback, parser, backfill logic
2. **Core** — Prediction engine, feature extraction, data access
3. **Web + CLI** — Flask app (HTMX) and typer CLI, both consume core

### Project Structure

```
ganyan/
├── pyproject.toml              # uv-managed, single package
├── alembic.ini
├── alembic/
│   └── versions/
├── src/
│   └── ganyan/
│       ├── __init__.py
│       ├── config.py           # pydantic-settings, env-driven
│       ├── db/
│       │   ├── __init__.py
│       │   ├── models.py       # SQLAlchemy ORM models
│       │   └── session.py      # Engine + session factory
│       ├── scraper/
│       │   ├── __init__.py
│       │   ├── tjk_api.py      # TJK API client (discover + fetch)
│       │   ├── tjk_html.py     # HTML fallback scraper
│       │   ├── parser.py       # Raw data → domain objects
│       │   └── backfill.py     # Incremental historical loader
│       ├── predictor/
│       │   ├── __init__.py
│       │   ├── bayesian.py     # Bayesian prediction model
│       │   └── features.py     # Feature extraction
│       ├── web/
│       │   ├── __init__.py
│       │   ├── app.py          # Flask factory
│       │   ├── routes.py       # API + page routes
│       │   ├── templates/
│       │   └── static/
│       └── cli/
│           ├── __init__.py
│           └── main.py         # typer CLI
├── tests/
│   ├── test_scraper/
│   ├── test_predictor/
│   └── test_web/
├── docker-compose.yml          # PostgreSQL only
├── Dockerfile                  # Not used locally, ready for cloud
├── .env.example
└── README.md
```

## Database Schema

PostgreSQL via SQLAlchemy ORM. Alembic for migrations.

### Tables

**`tracks`**
| Column | Type | Notes |
|--------|------|-------|
| id | serial PK | |
| name | varchar, unique | e.g. "İstanbul" |
| city | varchar | |
| surface_types | text[] | e.g. {"çim", "kum", "sentetik"} |

**`races`**
| Column | Type | Notes |
|--------|------|-------|
| id | serial PK | |
| track_id | FK → tracks | |
| date | date | |
| race_number | smallint | |
| distance_meters | int | |
| surface | varchar | çim/kum/sentetik |
| race_type | varchar | |
| horse_type | varchar | |
| weight_rule | varchar | |
| status | enum | scheduled / resulted / cancelled |
| | unique | (track_id, date, race_number) |

**`horses`**
| Column | Type | Notes |
|--------|------|-------|
| id | serial PK | |
| name | varchar, unique | TJK names are unique in active registry |
| age | smallint | nullable, updated per season |
| origin | varchar | sire/dam country |
| owner | varchar | |
| trainer | varchar | |

**`race_entries`**
| Column | Type | Notes |
|--------|------|-------|
| id | serial PK | |
| race_id | FK → races | |
| horse_id | FK → horses | |
| gate_number | smallint | Barrier/start position |
| jockey | varchar | |
| weight_kg | decimal(4,1) | |
| hp | decimal(5,1) | Handikap Puanı |
| kgs | smallint | Koşmama Gün Sayısı |
| s20 | decimal(5,2) | Son 20 yarış performansı |
| eid | varchar | En İyi Derece (time string) |
| gny | decimal(5,2) | Günlük Nispi Yarış puanı |
| agf | decimal(5,2) | Ağırlıklı Galibiyet Faktörü |
| last_six | varchar | e.g. "2 4 4 5 2 7" |
| finish_position | smallint | NULL before race |
| finish_time | varchar | NULL before race |
| performance_score | decimal(5,2) | NULL before race |
| predicted_probability | decimal(5,2) | NULL before prediction |

**Indexes:** `races(date)`, `races(track_id, date)`, `race_entries(race_id)`, `race_entries(horse_id)`

**`scrape_log`**
| Column | Type | Notes |
|--------|------|-------|
| id | serial PK | |
| date | date | |
| track | varchar | |
| status | enum | success / failed / skipped |
| scraped_at | timestamp | |

## Scraper

### Strategy

1. **API discovery first**: Probe TJK website for undocumented JSON/XML endpoints (XHR calls, GraphQL, REST patterns). Wrap in `TJKClient` class.
2. **HTML fallback**: If no usable API, scrape HTML with httpx + BeautifulSoup. Same interface as API client.

### Interface

```python
class TJKClient:
    async def get_race_card(self, date: date) -> list[RawRaceData]
    async def get_race_results(self, date: date) -> list[RawResultData]
    async def get_horse_stats(self, horse_name: str) -> RawHorseData
```

### Parser

Raw data (JSON or HTML) → validated dataclasses → database insertion. Handles:
- Turkish character normalization
- EİD time string → seconds conversion
- last_six string → list parsing
- Field validation

### Backfill

- Processes dates in reverse chronological order (recent first)
- Tracks progress in `scrape_log` table — idempotent, safe to re-run
- Rate limited: 1-2 second delays between requests
- CLI: `ganyan scrape --backfill --from 2024-01-01`

### Daily Flow

1. `ganyan scrape --today` — fetch race cards → insert races + race_entries (pre-race fields)
2. `ganyan scrape --results` — fetch results → update race_entries (post-race fields)

### HTTP

All requests via `httpx` (async-capable). Respect rate limits. Rotate user-agent if needed.

## Prediction Engine

### Feature Extraction (`features.py`)

From a horse's `race_entries` history, compute:

| Feature | Source | Logic |
|---------|--------|-------|
| Speed figure | eid | Normalize by distance and surface for cross-race comparison |
| Form cycle | last_six | Exponential decay over last 6 positions; captures improving vs declining trend |
| Weight delta | weight_kg | Difference from field average; lighter relative to field = advantage |
| Rest days | kgs | Fitness curve: peak 14-28 days, penalize extremes |
| Class indicator | hp | Normalized by race class; stepping up or down? |
| Track affinity | history | Win rate / place rate at this track + surface combo |

### Bayesian Model (`bayesian.py`)

Empirical Bayesian approach (no MCMC initially):

1. **Prior**: Base probability = 1/N (field size)
2. **Likelihood updates**: Multiply prior by likelihood ratios from each feature
3. **Normalize**: Sum to 100%
4. **Confidence**: Wide band with little history, narrow with 20+ races

```python
class BayesianPredictor:
    def predict(self, race_id: int) -> list[Prediction]

@dataclass
class Prediction:
    horse_id: int
    horse_name: str
    probability: float        # 0-100
    confidence: float         # 0-1
    contributing_factors: dict # feature → impact
```

`contributing_factors` shows which features pushed probability up/down — displayed in web UI.

**Evolution path**: Swap in PyMC model with learned priors once enough data. Interface stays the same.

## Web App

### Stack

- Flask with factory pattern (`create_app()`)
- Jinja2 templates + HTMX for dynamic updates
- Bootstrap 5 + Turkish language interface
- No JavaScript framework

### Routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Dashboard: today's races, recent predictions |
| GET | `/races/<date>` | Race cards for a date |
| GET | `/races/<race_id>/predict` | Run prediction, show results |
| GET | `/history` | Past predictions vs actual results |
| POST | `/scrape/today` | Trigger manual scrape |

JSON responses when `Accept: application/json`, HTML otherwise.

## CLI

Built with `typer`.

```
ganyan scrape --today                      # Fetch today's race cards
ganyan scrape --results                    # Fetch today's results
ganyan scrape --backfill --from 2024-01-01 # Historical backfill

ganyan predict <race_id>                   # Predict specific race
ganyan predict --today                     # Predict all today's races
ganyan predict --today --json              # JSON output

ganyan races --today                       # List today's races
ganyan races --date 2024-03-15             # List races for a date

ganyan db init                             # Run migrations
ganyan db reset                            # Drop and recreate
```

CLI imports core modules directly — no HTTP roundtrip.

## Configuration

`pydantic-settings` based. Reads env vars, falls back to `.env`.

```python
class Settings:
    database_url: str = "postgresql://ganyan:ganyan@localhost:5432/ganyan"
    tjk_base_url: str = "https://www.tjk.org"
    scrape_delay: float = 2.0
    log_level: str = "INFO"
    flask_port: int = 5003
    flask_debug: bool = False
```

## Infrastructure

### Local Development

- PostgreSQL via `docker-compose.yml` (only service containerized)
- Python app runs natively on Mac
- `uv` for dependency management

### Docker Compose

```yaml
services:
  db:
    image: postgres:16
    ports: ["5432:5432"]
    environment:
      POSTGRES_DB: ganyan
      POSTGRES_USER: ganyan
      POSTGRES_PASSWORD: ganyan
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

### Cloud Readiness (not built now)

- `Dockerfile` in repo (ready, not actively used)
- Alembic migrations bootstrap any Postgres instance
- Env-var config — swap `DATABASE_URL` for cloud DB
- Scraper runs as cron job / cloud scheduler
- Web app deploys behind gunicorn

## Out of Scope

- Authentication (single user)
- Redis / caching
- Async workers (Celery)
- CI/CD pipeline
- Monitoring / alerting
- ML model (Bayesian only for now; ML added later)
- Real-time odds / during-race updates

## Key Dependencies

- `sqlalchemy` + `alembic` — ORM and migrations
- `httpx` — async HTTP client
- `beautifulsoup4` — HTML parsing fallback
- `pydantic` + `pydantic-settings` — validation and config
- `flask` — web framework
- `typer` — CLI framework
- `numpy` — numerical computation for predictor
- `psycopg` — PostgreSQL driver
