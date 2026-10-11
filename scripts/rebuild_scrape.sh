#!/usr/bin/env bash
# Resumable historical rebuild: month by month, newest first.
# Cards (backfill) then full-field results per month; finished months are
# recorded in $DONE so a restart skips them.
# Usage: scripts/rebuild_scrape.sh 2023-01-01 2026-10-02
#        RESULTS_ONLY=1 scripts/rebuild_scrape.sh ...   # re-parse results only
# Portable across GNU and BSD/macOS: month arithmetic runs in Python, not `date -d`.
set -u
START=${1:?start date}
END=${2:?end date}
STATE_DIR=${REBUILD_STATE_DIR:-logs/rebuild}
DONE="$STATE_DIR/done_months${RESULTS_ONLY:+_results}.txt"
mkdir -p "$STATE_DIR"
touch "$DONE"

ts() { date +%Y-%m-%dT%H:%M:%S%z; }

# One line per month, newest first: "YYYY-MM from to", clipped to START..END.
months=$(uv run python - "$START" "$END" <<'PY'
import datetime as dt
import sys

start = dt.date.fromisoformat(sys.argv[1])
end = dt.date.fromisoformat(sys.argv[2])
year, month = end.year, end.month
while (year, month) >= (start.year, start.month):
    first = dt.date(year, month, 1)
    last = dt.date(year + month // 12, month % 12 + 1, 1) - dt.timedelta(days=1)
    print(f"{year:04d}-{month:02d}", max(first, start), min(last, end))
    year, month = (year - 1, 12) if month == 1 else (year, month - 1)
PY
) || { echo "$(ts) could not compute month windows" >&2; exit 1; }

while read -r month from to; do
  if grep -qx "$month" "$DONE"; then
    echo "skip $month"
  else
    echo "$(ts) start $month ($from -> $to)"
    if { [ -n "${RESULTS_ONLY:-}" ] || uv run ganyan scrape --backfill --from "$from" --to "$to"; } \
       && uv run ganyan scrape --results-range --from "$from" --to "$to" --rescrape; then
      echo "$month" >> "$DONE"
      echo "$(ts) done $month"
    else
      echo "$(ts) FAILED $month (will retry on next run)"
    fi
  fi
done <<< "$months"
echo "$(ts) rebuild finished"
