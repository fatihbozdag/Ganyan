# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ganyan is a Turkish horse racing (at yarışı) prediction system that scrapes race data from TJK (Türkiye Jokey Kulübü), stores it in SQLite, and generates predictions via a dual ML + Bayesian engine served through a Flask web app.

## Commands

```bash
# Install dependencies (use uv per global preference)
uv pip install -r requirements.txt

# Run the Flask web app (serves on port 5003)
python app.py

# Run prediction simulation
python test_prediction.py

# Build database from processed CSVs
python scripts/create_db_from_processed.py

# Run the TJK scraper
python scripts/run_scraper.py

# Run tests
pytest
```

## Architecture

### Dual Prediction Pipeline

The system runs two independent prediction models and combines their outputs:

1. **ML Predictor** (`race_analyzer.py` → `RaceAnalyzer`): Weighted scoring across speed (30%), form (25%), weight (20%), track fit (15%), class (10%). Applies multiplicative adjustments for EİD, recent form, weight, KGS, and S20.

2. **Bayesian Predictor** (`bayesian_predictor.py` → `BayesianPredictor`): Prior-based probability estimation with speed (35%), form (30%), weight (20%), class (15%) priors. Uses exponential decay for form cycles, normalizes to sum=100%.

3. **Combined**: 60% ML + 40% Bayesian → final ranking.

### Data Flow

```
TJK website → scrapers/ (Selenium/Safari) → data/processed/*.csv
    → scripts/create_db_from_processed.py → data/races_new.db (SQLite)
    → RaceAnalyzer + BayesianPredictor → Flask API → Web UI
```

### Flask App (`app.py`)

- `GET /` — Main UI (Bootstrap 5, Turkish language)
- `GET /get_predictions` — Returns ML, Bayesian, and combined predictions as JSON
- `POST /add_horse` — Add horse entry to `current_race.json`
- `POST /clear_race` — Reset current race
- `POST /update_race_info` — Update race metadata
- `GET /get_race_data` — Current race JSON

### Key Turkish Racing Metrics

- **HP** — Handikap Puanı (handicap points)
- **KGS** — Koşmama Gün Sayısı (days since last race; 21 days optimal)
- **S20** — Son 20 yarış performansı (last 20 races performance)
- **EİD** — En İyi Derece (best time/performance)
- **GNY** — Günlük Nispi Yarış puanı (daily relative race score)
- **AGF** — Ağırlıklı Galibiyet Faktörü (weighted win factor)
- **Last Six** — Recent race finishing positions (e.g., "2 4 4 5 2 7")

### Database Schema (SQLite)

Three tables: `races` (date, venue, race_no, distance_track), `horses` (name, age, origin), `race_results` (FK to both, with jockey, weight, performance_score, last_6_races, score_1-6). Utility class: `src/utils/db_utils.py` → `DatabaseManager`.

### Scrapers

Two implementations targeting `medya-cdn.tjk.org` CSV endpoints:
- **Selenium-based** (`scrapers/tjk_scraper.py`): Safari WebDriver, handles dynamic pages
- **Scrapy spider** (`tjk_scraper/`): Polite crawling (3s delay, single concurrent request)

CSV URL pattern: `https://medya-cdn.tjk.org/raporftp/TJKPDF{YYYY-MM-DD}/{DD.MM.YYYY}-{TRACK}-GunlukYarisSonuclari-TR.csv`

### Scripts Directory

25 utility scripts in `scripts/` for database creation, scraping, analysis, prediction, and a GUI (`enhanced_race_gui.py`). Entry points: `run_scraper.py`, `predict_race.py`, `analyze_races.py`, `create_db_from_processed.py`.

## Conventions

- Variable names mix Turkish and English (metric names are Turkish abbreviations)
- Data files use Turkish date format: `DD.MM.YYYY-{TrackName}.csv`
- Track names include Turkish characters (İstanbul, İzmir, Şanlıurfa, etc.)
- Processed data lives in `data/processed/`, database in `data/races_new.db`
- Race state persisted in `current_race.json` at project root
