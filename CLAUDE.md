# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ganyan is a Turkish horse racing prediction system. It scrapes race data from TJK (Türkiye Jokey Kulübü), stores it in PostgreSQL, and generates Bayesian predictions served via CLI and Flask web app.

## Commands

```bash
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
```

## Architecture

Three-layer service-oriented monorepo sharing PostgreSQL:

1. **Scraper** (`src/ganyan/scraper/`) — TJK website client using AJAX endpoints at `/TR/YarisSever/Info/Sehir/GunlukYarisProgrami`. `tjk_api.py` fetches race cards and results per city via `SehirId` parameters. `parser.py` normalizes raw HTML data into dataclasses. `backfill.py` handles idempotent storage and incremental historical loading.

2. **Predictor** (`src/ganyan/predictor/`) — Empirical Bayesian model. `features.py` extracts speed figure, form cycle (exponential decay), weight delta, rest fitness (Gaussian curve), and class indicator. `bayesian.py` computes prior (1/N) x feature likelihoods → normalized probabilities with confidence scores and contributing factors.

3. **Web + CLI** (`src/ganyan/web/`, `src/ganyan/cli/`) — Flask app with HTMX (Bootstrap 5, Turkish UI). Typer CLI for terminal use. Both consume predictor and scraper directly.

### Data Flow

```
TJK website (AJAX per city) → scraper/tjk_api.py → scraper/parser.py → scraper/backfill.py → PostgreSQL
                                                                                                    ↓
CLI (ganyan predict) ← predictor/bayesian.py ← predictor/features.py ← race_entries table
Flask (/races/<id>/predict) ←────────────────┘
```

### Key Turkish Racing Metrics

- **HP** — Handikap Puanı (handicap points)
- **KGS** — Koşmama Gün Sayısı (days since last race; 14-28 optimal)
- **S20** — Son 20 yarış performansı (last 20 races performance)
- **EİD** — En İyi Derece (best time, stored as string "1.30.45", converted to seconds for computation)
- **GNY** — Günlük Nispi Yarış puanı (daily relative race score)
- **AGF** — Ağırlıklı Galibiyet Faktörü (weighted win factor)

### Database

PostgreSQL 16 via Docker Compose. SQLAlchemy 2.0 ORM + Alembic migrations. Tables: `tracks`, `races` (unique on track+date+race_number), `horses` (unique on name), `race_entries` (pre-race + post-race fields in one row), `scrape_log`.

### Config

`pydantic-settings` reads from `.env` file or environment variables. See `.env.example`. Key: `DATABASE_URL`, `TJK_BASE_URL`, `SCRAPE_DELAY`, `FLASK_PORT`.
