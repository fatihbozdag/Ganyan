# Ganyan — Always-online setup (macOS / launchd)

Two ways to keep the system running 24/7.  Pick **one**, not both.

## Option 1 — Web app + scheduler in one process (recommended)

Serves the Flask dashboard on port 5003 *and* runs all four scheduled
jobs in the same Python process.

```bash
cp ops/com.ganyan.web.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ganyan.web.plist

# Check status
launchctl print gui/$(id -u)/com.ganyan.web

# Tail logs
tail -f /tmp/ganyan-web.log /tmp/ganyan-web.err
```

To stop:

```bash
launchctl bootout gui/$(id -u)/com.ganyan.web
rm ~/Library/LaunchAgents/com.ganyan.web.plist
```

## Option 2 — Headless scheduler only (no web UI)

For servers where you don't want the Flask app running.

```bash
cp ops/com.ganyan.daemon.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ganyan.daemon.plist
```

Same `launchctl print` / `launchctl bootout` pattern, different label
(`com.ganyan.daemon`).

## Scheduled jobs

Defined in `src/ganyan/scheduler.py`, times in **Europe/Istanbul**:

| ID | When | What |
|----|------|------|
| `morning_card` | 08:30 daily | Scrape today's program, pre-predict every race |
| `results_poll` | Every 20 min, 13:00–23:59 | Refresh today's finish positions + payouts |
| `pedigree_refresh` | Sun 03:00 | Crawl new horses that gained a tjk_at_id |
| `monthly_retrain` | 1st of month 03:30 | Retrain main + value models on 90-day window |

All schedules are cron-expressible; edit `_add_jobs` in
`scheduler.py` to change them.

## Environment flags

- `GANYAN_SKIP_LAUNCH_REFRESH=1` — skip the 14-day historical refresh
  that runs at Flask startup.
- `GANYAN_SKIP_SCHEDULER=1` — skip the APScheduler embedded in the
  Flask app (useful during dev work).

Set them in the plist's `EnvironmentVariables` dict if you ever need
to disable a feature without editing code.

## Manual ops while the daemon is running

These are safe to run in parallel with the daemon — each command opens
its own DB session and completes quickly:

```bash
uv run ganyan uclu-picks --date today
uv run ganyan exotics-backtest --from 2026-01-01 --model ml
uv run ganyan train              # rolling 90-day window by default
uv run ganyan crawl horses       # incremental pedigree update
```

## Prerequisites

- PostgreSQL 15 running via Homebrew (`brew services start postgresql@15`)
- `uv` at `/opt/homebrew/bin/uv` (adjust plist `ProgramArguments` if
  installed elsewhere)
- Project cloned at the path shown in each plist's `WorkingDirectory`
  key (edit if you move the repo)

## Failure modes

- If `launchctl bootstrap` says "service already bootstrapped" — you
  already installed it.  Run `launchctl bootout` first.
- If the process keeps crashing in a loop, `ThrottleInterval` (30s)
  prevents launchd from hammering restarts.  Check
  `/tmp/ganyan-web.err` or `/tmp/ganyan-daemon.err`.
- To disable just the scheduler without taking down the web app, add
  `GANYAN_SKIP_SCHEDULER=1` to the plist and reload.
