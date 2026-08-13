# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ganyan is a Turkish horse racing prediction system. It scrapes race data from TJK (Türkiye Jokey Kulübü), stores it in PostgreSQL, and generates Bayesian predictions served via CLI and Flask web app.

## Commands

```bash
# Start PostgreSQL (Homebrew on this machine — NOT Docker)
brew services start postgresql@15
# After a reboot a stale lock can block startup; if `pg_isready` fails:
#   rm /opt/homebrew/var/postgresql@15/postmaster.pid && brew services restart postgresql@15
# If brew services itself errors (Homebrew internal bug, seen 2026-08-12), bypass it:
#   /opt/homebrew/opt/postgresql@15/bin/pg_ctl -D /opt/homebrew/var/postgresql@15 -l /opt/homebrew/var/log/postgresql@15-server.log start

# Install dependencies
uv sync --all-extras

# Run database migrations
uv run ganyan db init

# Scrape today's race cards
uv run ganyan scrape --today

# Scrape today's results
uv run ganyan scrape --results

# Backfill historical data (cards), then full-field results for training data.
# GOTCHA: both passes share one completion namespace in scrape_log, so a results
# pass run after a card backfill silently skips every date — pass --rescrape.
uv run ganyan scrape --backfill --from 2024-01-01
uv run ganyan scrape --results-range --from 2024-01-01 --rescrape

# Predict a specific race
uv run ganyan predict <race_id>

# Predict all today's races
uv run ganyan predict --today
uv run ganyan predict --today --json

# Read the daemon's stored ensemble predictions (the invariant #6 tool)
uv run ganyan predictions

# Full command list (~19 commands; only a subset is documented here)
uv run ganyan --help

# List races
uv run ganyan races --today
uv run ganyan races --date 2024-03-15

# Start web app (port 5003)
uv run python -c "from ganyan.web.app import run; run()"

# Restart the in-process scheduler/daemon after code or model changes
launchctl kickstart -k gui/$(id -u)/com.ganyan.web
# CAVEAT 2026-08-12: the launchd path hangs in TCC until /opt/homebrew/bin/uv is
# re-granted Full Disk Access (System Settings → Privacy & Security; uv upgrades
# invalidate the grant). Until then the daemon runs as a detached user process —
# check `lsof -nP -iTCP:5003 -sTCP:LISTEN`; restart by killing it and re-running
# `nohup uv run python -c "from ganyan.web.app import run; run()" >>/tmp/ganyan-web.log 2>&1 &`
# from a terminal shell inside the repo.

# Launch the desktop GUI (CustomTkinter, GanyanGUI/ — untracked)
# GOTCHA: GanyanGUI/main.py auto-pip-installs missing deps via a Tk dialog, including
# psycopg2-binary which conflicts with the project's psycopg3 — it can mutate the venv.
# Proper GUI deps live in the `gui` extra (customtkinter, matplotlib, openpyxl).
bash run_gui.sh

# Run tests
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_predictor/test_bayesian.py::test_probabilities_sum_to_100 -v
```

## Project goal (2026-05-03)

**Pick winners across all bet types.** Tek, İkili, Üçlü, 4'lü, 5'lı, 6'lı, 7'li — every structure. The model and pipeline exist to serve this goal, not to satisfy academic process rules. Hit rate is the primary metric; engineering work serves picking winners, not the other way around.

## Critical invariants (read before consulting /advice for a real bet)

1. **Model beats chance ≠ bets have edge.** The 2026-05-02 chance-hypothesis settlement proved the model has 3-17× lift over random (p≈0). The same date's OOS retest proved real kept-race ROI is **−20% to −30% on uclu_box6 and sirali_ikili_top1** — the takeout floor. These are independent claims. The advice gate filters for confidence, not edge after takeout.

2. **The halt flag is authoritative.** `/tmp/ganyan-halt.flag` (or `$GANYAN_HALT_FLAG_PATH`) is set by canaries (rolling-PnL, uniformity guard, heartbeat, scrape integrity, regime monitor). When set, `/advice` and `ganyan advice` suppress Kelly stakes. Manually clear with `rm /tmp/ganyan-halt.flag` only after investigating the reason.

3. **OOS bar: ≥365 days AND ≥1500 races.** Set by the V2 retraction (2026-05-02). Enforced in `logs/discordance_oos_backtest.py:assert_min_window`. Do not bypass.

4. **Frame around winning horse and winning bet, not payout/ROI** — primary metric is top-1 hit rate. (Existing memory; reaffirmed here.)

5. **No tested single-race structure is positive-EV; multi-race exotics are the only remaining path.** Original version (2026-05-04 "Tek+Plase ≥30% default") was based on limited data + a buggy ROI calc using `SUM(net_tl)/SUM(stake_tl)` that excluded uncaptured plase wins (memory `feedback_roi_calculation_rule.md`). 2026-05-11 corrected pivot analysis on n=1,899 picks, filtered to races where plase pool actually formed:
   - Pure Plase: **−10.8% to −25% ROI** across thresholds (best at ≥0%)
   - Pure Tek: **−20% to −37%** across thresholds
   - Combined Tek+Plase: **−17% to −36%** (Tek always drags it down)
   - Üçlü K-3 / box6: −33%
   - Sırali İkili: −14%
   - Multi-race 6'lı/7'lı: mathematical paths to positive EV but **untested in ledger** (1 pick total)

   Plase pool doesn't form on ~45% of races even at field≥8 (TJK rules + betting volume), so plase_top1 picks generated on those races can never pay. Picks generator now skips plase_top1 when fewer than 8 horses carry win probabilities — the gate is `len(win_probs) >= 8` at `src/ganyan/predictor/picks.py:218`, counting post-scratch runners, NOT a `field_size` column; the additional 45% "no pool" rate is structural and unfixable from our side.

   **Single-race betting is bleed-only. Multi-race exotics require small-stake live validation before commitment.** The validation harness exists since 2026-05: `morning_card` generates a daily 6'lı paper-trade coupon per track (≤144 tickets, written to `multi_race_picks`), graded via `uv run ganyan multi-grade` — monitor it, don't rebuild it. Never recover losses with bigger exotics — recovery comes from the next clean signal.

6. **Pull live ensemble before any bet recommendation.** `ganyan predict --today` invokes the single-head MLPredictor by default; the daemon's scheduled inference runs the multi-head EnsemblePredictor (currently 11 heads — see Architecture) and writes to the `Prediction` table. These can disagree by 10pp. Read the stored rows with `uv run ganyan predictions` (read-only view of the latest `Prediction` rows), or force ensemble inference with `ganyan predict --today --model ensemble`. Re-pull within 30 minutes of post when stakes are >100 TL — and note the daemon's `repredict_upcoming` job stops at 20:35 while races run to ~23:00, so evening races ride stale predictions unless you re-pull manually.

7. **Never overwrite the live model file directly.** `ganyan train` (default invocation) writes to `models/lightgbm_ranker.{txt,meta.json}` — i.e. THE LIVE MODEL the daemon reads every 30 min. A bare `uv run ganyan train` will silently replace the in-production weights with whatever the random seed + this morning's data window produced, even when in-sample top-1 is 2-5pp WORSE than what's already deployed (observed: pedigree_v1 42.94% top-1 was overwritten by a routine retrain that landed at 40.25%). Same applies to any external tool, agent, or one-off script that writes under `models/`. The daemon itself is the worst offender: the `monthly_retrain` job (`src/ganyan/scheduler.py`, cron day 1 03:30) bare-trains straight onto `lightgbm_ranker.*` and `lightgbm_value.*` with no OOS gate — it fired 2026-06-01 and replaced the 42.94% top-1 weights with 40.92%. Remediated 2026-08-12: live files reverted to the HEAD weights (June output preserved in `models/_auto_retrain_20260601_backup/`) and the job registration commented out in `scheduler.py`. Re-enable only behind a candidate-name + OOS-gate flow. The discipline is:

   ```bash
   # 1. Train under a non-production name
   uv run ganyan train --model-name lightgbm_ranker_test

   # 2. OOS validate against the project's window bar (≥365d, ≥1500 races).
   # Use the PER-MODEL gate. (discordance_oos_backtest.py IGNORES --model —
   # it always scores the live ensemble; discovered 2026-08-13 when a
   # candidate "gate" run returned bit-identical numbers to the live run.)
   uv run python logs/oos_model_gate.py --candidate candidates/lightgbm_ranker_test

   # 3. ONLY swap if OOS top-1 lift ≥ +1pp vs current production
   mv models/lightgbm_ranker_test.txt models/lightgbm_ranker.txt
   mv models/lightgbm_ranker_test.meta.json models/lightgbm_ranker.meta.json
   ```

   If the live model has been overwritten without OOS, revert via `git checkout HEAD -- models/lightgbm_ranker.{txt,meta.json}` before the next 30-min daemon tick reads it.

## Architecture

Three-layer service-oriented monorepo sharing PostgreSQL:

1. **Scraper** (`src/ganyan/scraper/`) — TJK website client using AJAX endpoints at `/TR/YarisSever/Info/Sehir/GunlukYarisProgrami`. `tjk_api.py` fetches race cards and results per city via `SehirId` parameters. `parser.py` normalizes raw HTML data into dataclasses. `backfill.py` handles idempotent storage and incremental historical loading. TJK serves an incomplete TLS chain, so `tjk_api.py` injects the OS trust store via `truststore`; without it scraping fails on macOS with `CERTIFICATE_VERIFY_FAILED`.

2. **Predictor** (`src/ganyan/predictor/`) — two coexisting model families. The **production inference path** is the LightGBM stack in `ml/`: `ml/trainer.py` trains the ranker (`models/lightgbm_ranker.*`), `ml/predictor.py` is the single-head `MLPredictor` (used by `ganyan predict`), and `ml/ensemble.py` is the `EnsemblePredictor` the daemon runs on schedule and writes to the `Prediction` table (see invariants #6/#7). The ensemble's head count is NOT fixed: it loads every `*.meta.json` in the `models/` root (non-recursive glob), so dropping a model file there silently adds a voting head, while subdirectories (`plase_v1/`, `pedigree_v1/`, …) are quarantined. Currently 11 heads (the failed-OOS `lightgbm_ranker_wx_v1` head was quarantined to `models/_quarantine_wx_v1/` on 2026-08-12; it had been voting on 5 all-NaN weather features that `features.py` no longer produces — missing columns reindex silently instead of raising). Live ranker schema: 42 features. `features.py` builds the shared feature matrix (speed figure, form cycle, AGF edge, pedigree, etc.). The **Bayesian model** (`bayes/`, `bayesian.py`) powers the `/advice` skip-gate (v3 default) — prior (1/N) × feature likelihoods → normalized probabilities with confidence + contributing factors.

3. **Web + CLI** (`src/ganyan/web/`, `src/ganyan/cli/`) — Flask app with HTMX (Bootstrap 5, Turkish UI). Typer CLI for terminal use. Both consume predictor and scraper directly.

### Data Flow

```
TJK website (AJAX per city) → scraper/tjk_api.py → scraper/parser.py → scraper/backfill.py → PostgreSQL
                                                                                                    ↓
daemon (scheduled)   → predictor/ml/ensemble.py  (multi-head) → Prediction table  ←┐
CLI (ganyan predict) → predictor/ml/predictor.py (single-head)                      ├ predictor/features.py ← race_entries
/advice (CLI + web)  → predictor/bayes/          (skip-gate v3)                     ┘
```

### Key Turkish Racing Metrics

- **HP** — Handikap Puanı (handicap points)
- **KGS** — Koşmama Gün Sayısı (days since last race; 14-28 optimal)
- **S20** — Son 20 yarış performansı (last 20 races performance)
- **EİD** — En İyi Derece (best time, stored as string "1.30.45", converted to seconds for computation)
- **GNY** — Günlük Nispi Yarış puanı (daily relative race score)
- **AGF** — Ağırlıklı Galibiyet Faktörü (weighted win factor)

### Database

PostgreSQL 15 via Homebrew (`brew services`; `docker-compose.yml` exists but is not the path used on this machine — it pins `postgres:16`, an unreconciled version skew, and `GanyanGUI/KULLANIM.md`'s "install Docker Desktop" instructions are likewise wrong for this machine). SQLAlchemy 2.0 ORM + Alembic migrations. Tables: `tracks`, `races` (unique on track+date+race_number), `horses` (unique on name), `race_entries` (pre-race + post-race fields in one row), `scrape_log`, plus `predictions`, `picks`, `multi_race_picks`, `multi_race_pools`, `agf_snapshots`, `external_signals`.

### Config

`pydantic-settings` reads from `.env` file or environment variables. See `.env.example`. Key: `DATABASE_URL`, `TJK_BASE_URL`, `SCRAPE_DELAY`, `FLASK_PORT`.

### Repo layout traps

- `.worktrees/premortem-fixes/` is a stale git worktree (May 2026) carrying its own OLDER CLAUDE.md — never take guidance from or edit files under it.
- `claude-code-ask-gemini/` is a separately-cloned git repo nested inside this one; `git status` shows it as a single untracked directory and hides its contents.
- `logs/discordance_oos_backtest.py` is tracked and load-bearing (it enforces invariant #3) amid ~20 disposable untracked `logs/*.py` experiment scripts — never bulk-clean `logs/`.

### Reporting Conventions

When discussing model or strategy performance, frame around the **winning horse** and **winning bet** — not the payout or ROI.

- **Primary metric**: top-1 hit rate (did we pick the winning horse?)
- **Secondary**: top-3 hit rate; binary strategy hit rate (`ganyan_top1`, `sirali_ikili_top1`, `uclu_top1`, `uclu_box6`)
- **Tertiary** (only when sizing strategy or explicitly asked): payout/ROI/net-TL

Payout reflects TJK pool dynamics (takeout, retail behavior, "devren" carryovers) more than model quality. A 33% top-1 day on short-priced favorites shows −25% ROI because the math doesn't work at 1.9× average odds — but the model is doing its job. Don't anchor model-quality reports on money.

## Maintenance routines

**Liveness first.** Nothing auto-starts. Before interpreting any canary or halt-flag state, verify the stack is alive: `launchctl list | grep com.ganyan` (expect web + heartbeat + integrity + regime) and `pg_isready`. An absent `/tmp/ganyan-halt.flag` under a DEAD daemon means "nothing is running to set it", not "all clear" — invariant #2 only holds while the daemon lives.

| Cadence | Task | How |
| --- | --- | --- |
| Daily 12:00 | Heartbeat (liveness + uniformity) | `launchctl list \| grep com.ganyan.heartbeat` |
| Daily 12:30 | Scrape integrity (AGF drift) | `launchctl list \| grep com.ganyan.integrity` |
| Daily 23:30 | Regime monitor (takeout drift) | `launchctl list \| grep com.ganyan.regime` |
| Weekly | Commit-ratio audit (ganyan vs linguistic) | `git log --since="7 days ago" --oneline \| wc -l` in each repo; if ganyan > 5× linguistic for 2 consecutive weeks, force a Ganyan freeze week |

In-process APScheduler jobs inside `com.ganyan.web` (`src/ganyan/scheduler.py`, Europe/Istanbul): `morning_card` 08:30 (cards + picks + daily 6'lı coupons), `results_poll` every 20 min 13–23h, `external_signals` 09:15 + 18:15, `agf_snapshot` :00/:30 11–22h, `repredict_upcoming` :05/:35 11–20h, `pedigree_refresh` Sun 03:00. An eighth job, `monthly_retrain` (day 1 03:30), was DISARMED 2026-08-12 — registration commented out in `scheduler.py`; see invariant #7 before re-enabling.

