#!/usr/bin/env bash
# Resumable historical rebuild: month by month, newest first.
# Cards (backfill) then full-field results per month; finished months are
# recorded in $DONE so a restart skips them.
# Usage: scripts/rebuild_scrape.sh 2023-01-01 2026-10-02
#        RESULTS_ONLY=1 scripts/rebuild_scrape.sh ...   # re-parse results only
set -u
START=${1:?start date}
END=${2:?end date}
STATE_DIR=${REBUILD_STATE_DIR:-logs/rebuild}
DONE="$STATE_DIR/done_months${RESULTS_ONLY:+_results}.txt"
mkdir -p "$STATE_DIR"
touch "$DONE"

cur=$(date -d "$(date -d "$END" +%Y-%m-01)" +%Y-%m-%d)
while [[ "$cur" > "$(date -d "$START -1 month" +%Y-%m-%d)" ]]; do
  month=${cur:0:7}
  from=$cur
  [[ "$from" < "$START" ]] && from=$START
  to=$(date -d "$cur +1 month -1 day" +%Y-%m-%d)
  [[ "$to" > "$END" ]] && to=$END
  if grep -qx "$month" "$DONE"; then
    echo "skip $month"
  else
    echo "$(date -Is) start $month ($from -> $to)"
    if { [ -n "${RESULTS_ONLY:-}" ] || uv run ganyan scrape --backfill --from "$from" --to "$to"; } \
       && uv run ganyan scrape --results-range --from "$from" --to "$to" --rescrape; then
      echo "$month" >> "$DONE"
      echo "$(date -Is) done $month"
    else
      echo "$(date -Is) FAILED $month (will retry on next run)"
    fi
  fi
  cur=$(date -d "$cur -1 month" +%Y-%m-%d)
done
echo "$(date -Is) rebuild finished"
