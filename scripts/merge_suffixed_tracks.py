"""Merge race-day-suffixed tracks into their city track.

Card scrapes from 2025-01 to 2026-06 stored some meetings under a track
named with TJK's race-day ordinal ("İzmir  (22. Yarış Günü)"), while the
results pages name the plain city ("İzmir"). Each such race therefore
exists twice: the suffixed copy holds the card-time data (predictions,
picks, AGF snapshots, external signals) and the plain copy holds the
results-page data. ``parser.normalize_track_name`` now strips the suffix;
this script repairs the rows already stored.

Rules, applied in one transaction:
  * The plain race survives. Its null race fields are filled from the
    suffixed copy; its non-null fields win.
  * Entries for the same horse merge into the plain entry the same way
    (plain wins, suffixed fills nulls). Predictions, AGF snapshots and
    external signals are repointed to the merged entry, never deleted.
  * Suffixed entries with no plain twin move into the plain race.
  * Picks and race-level external signals move to the plain race.
  * A suffixed race with no plain twin moves to the plain track.
  * Multi-race pools and picks move to the plain track. A pool identical to
    any pool of that day and type already there is dropped; a different
    pool under the same (date, track, type, index) is a separate pool of
    that day and gets the next free index.
  * Suffixed tracks are deleted once nothing references them.

Totals of predictions, picks, AGF snapshots, external signals and
multi-race picks are asserted unchanged; any failed check rolls back.
Dry run by default (the transaction is rolled back after the checks).

Usage:
  uv run python scripts/merge_suffixed_tracks.py --database-url postgresql+psycopg://.../ganyan_merge_test
  uv run python scripts/merge_suffixed_tracks.py --database-url ... --apply
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import create_engine, text

from ganyan.config import get_settings
from ganyan.scraper.parser import normalize_track_name

# Columns copied into the surviving row only where it is NULL.
RACE_FILL = (
    "post_time", "distance_meters", "surface", "race_type", "horse_type",
    "weight_rule", "pace_l800_leader_s", "pace_l800_runner_up_s",
    "ganyan_payout_tl", "ikili_payout_tl", "sirali_ikili_payout_tl",
    "uclu_payout_tl", "dortlu_payout_tl", "temp_celsius", "humidity_pct",
    "wind_kmh", "precipitation_mm", "track_condition", "kum_condition",
    "cim_sertlik", "track_elevation", "conditions_history",
)
ENTRY_FILL = (
    "gate_number", "start_gate", "jockey", "weight_kg", "hp", "kgs", "s20",
    "eid", "gny", "agf", "last_six", "finish_position", "finish_time",
    "performance_score", "predicted_probability", "equipment",
    "plase_payout_tl", "age_at_race", "trainer_at_race",
)
PRESERVED_TOTALS = {
    "predictions": "SELECT count(*) FROM predictions",
    "picks": "SELECT count(*) FROM picks",
    "agf_snapshots": "SELECT count(*) FROM agf_snapshots",
    "external_signals": "SELECT count(*) FROM external_signals",
    "multi_race_picks": "SELECT count(*) FROM multi_race_picks",
}


def _fill(columns, target, source):
    return ", ".join(f"{c} = coalesce({target}.{c}, {source}.{c})" for c in columns)


def _scalar(conn, sql, **params):
    return conn.execute(text(sql), params).scalar_one()


def merge(conn) -> dict[str, int]:
    stats: dict[str, int] = {}

    tracks = conn.execute(text("SELECT id, name FROM tracks")).all()
    plain_ids = {name: tid for tid, name in tracks if normalize_track_name(name) == name}
    mapping = []
    for tid, name in tracks:
        plain = normalize_track_name(name)
        if plain == name:
            continue
        if plain not in plain_ids:
            plain_ids[plain] = _scalar(
                conn, "INSERT INTO tracks (name) VALUES (:n) RETURNING id", n=plain,
            )
        mapping.append({"s": tid, "p": plain_ids[plain]})
    stats["suffixed_tracks"] = len(mapping)
    if not mapping:
        return stats

    conn.execute(text("CREATE TEMP TABLE track_map (s_track int PRIMARY KEY, p_track int) ON COMMIT DROP"))
    conn.execute(text("INSERT INTO track_map VALUES (:s, :p)"), mapping)

    conn.execute(text("""
        CREATE TEMP TABLE race_map ON COMMIT DROP AS
        SELECT s.id AS s_race, p.id AS p_race
        FROM races s
        JOIN track_map m ON m.s_track = s.track_id
        JOIN races p ON p.track_id = m.p_track AND p.date = s.date
                    AND p.race_number = s.race_number
    """))
    stats["race_pairs"] = _scalar(conn, "SELECT count(*) FROM race_map")

    # Race-level fields, then status (resulted on either copy wins).
    conn.execute(text(f"""
        UPDATE races p SET {_fill(RACE_FILL, 'p', 's')},
               status = CASE WHEN s.status = 'resulted' THEN s.status ELSE p.status END
        FROM race_map m JOIN races s ON s.id = m.s_race
        WHERE p.id = m.p_race
    """))

    conn.execute(text("""
        CREATE TEMP TABLE entry_map ON COMMIT DROP AS
        SELECT se.id AS s_entry, pe.id AS p_entry
        FROM race_map m
        JOIN race_entries se ON se.race_id = m.s_race
        JOIN race_entries pe ON pe.race_id = m.p_race AND pe.horse_id = se.horse_id
    """))
    stats["matched_entries"] = _scalar(conn, "SELECT count(*) FROM entry_map")

    conn.execute(text(f"""
        UPDATE race_entries p SET {_fill(ENTRY_FILL, 'p', 's')}
        FROM entry_map m JOIN race_entries s ON s.id = m.s_entry
        WHERE p.id = m.p_entry
    """))
    for table in ("predictions", "agf_snapshots", "external_signals"):
        stats[f"{table}_repointed_entry"] = conn.execute(text(f"""
            UPDATE {table} t SET race_entry_id = m.p_entry
            FROM entry_map m WHERE t.race_entry_id = m.s_entry
        """)).rowcount
    conn.execute(text("DELETE FROM race_entries e USING entry_map m WHERE e.id = m.s_entry"))

    stats["moved_entries"] = conn.execute(text("""
        UPDATE race_entries e SET race_id = m.p_race
        FROM race_map m WHERE e.race_id = m.s_race
    """)).rowcount

    stats["picks_repointed"] = conn.execute(text("""
        UPDATE picks t SET race_id = m.p_race FROM race_map m WHERE t.race_id = m.s_race
    """)).rowcount
    stats["external_signals_repointed_race"] = conn.execute(text("""
        UPDATE external_signals t SET race_id = m.p_race FROM race_map m WHERE t.race_id = m.s_race
    """)).rowcount
    stats["paired_races_deleted"] = conn.execute(text(
        "DELETE FROM races r USING race_map m WHERE r.id = m.s_race",
    )).rowcount

    stats["unpaired_races_moved"] = conn.execute(text("""
        UPDATE races r SET track_id = m.p_track FROM track_map m WHERE r.track_id = m.s_track
    """)).rowcount

    # Multi-race pools: a pool identical to any plain-track pool of that day
    # (whatever its index; the two pages numbered a day's pools differently)
    # is dropped; a distinct pool clashing on index is renumbered.
    stats["pools_identical_dropped"] = conn.execute(text("""
        DELETE FROM multi_race_pools s
        USING track_map m, multi_race_pools p
        WHERE s.track_id = m.s_track AND p.track_id = m.p_track AND p.date = s.date
          AND p.pool_type = s.pool_type
          AND (p.start_race_no, p.end_race_no, p.winning_combo, p.payout_tl)
              IS NOT DISTINCT FROM (s.start_race_no, s.end_race_no, s.winning_combo, s.payout_tl)
    """)).rowcount
    clashes = conn.execute(text("""
        SELECT s.id, m.p_track, s.date, s.pool_type
        FROM multi_race_pools s
        JOIN track_map m ON m.s_track = s.track_id
        JOIN multi_race_pools p ON p.track_id = m.p_track AND p.date = s.date
             AND p.pool_type = s.pool_type AND p.pool_index = s.pool_index
        ORDER BY s.date, s.pool_type, s.pool_index
    """)).all()
    for pool_id, p_track, day, pool_type in clashes:
        conn.execute(text("""
            UPDATE multi_race_pools SET track_id = :t, pool_index = (
                SELECT coalesce(max(pool_index), 0) + 1 FROM multi_race_pools
                WHERE track_id = :t AND date = :d AND pool_type = :k)
            WHERE id = :id
        """), {"t": p_track, "d": day, "k": pool_type, "id": pool_id})
    stats["pools_renumbered"] = len(clashes)
    stats["pools_repointed"] = conn.execute(text("""
        UPDATE multi_race_pools t SET track_id = m.p_track FROM track_map m WHERE t.track_id = m.s_track
    """)).rowcount
    stats["multi_race_picks_repointed"] = conn.execute(text("""
        UPDATE multi_race_picks t SET track_id = m.p_track FROM track_map m WHERE t.track_id = m.s_track
    """)).rowcount

    stats["tracks_deleted"] = conn.execute(text(
        "DELETE FROM tracks t USING track_map m WHERE t.id = m.s_track",
    )).rowcount
    return stats


def check(conn, before: dict[str, int]) -> list[str]:
    failures = []
    for name, sql in PRESERVED_TOTALS.items():
        after = _scalar(conn, sql)
        if after != before[name]:
            failures.append(f"{name}: {before[name]} -> {after}")
    names = conn.execute(text("SELECT name FROM tracks")).scalars().all()
    leftover = [n for n in names if normalize_track_name(n) != n]
    if leftover:
        failures.append(f"suffixed tracks remain: {leftover[:5]}")
    dup = _scalar(conn, """
        SELECT count(*) FROM (SELECT 1 FROM races
        GROUP BY track_id, date, race_number HAVING count(*) > 1) d
    """)
    if dup:
        failures.append(f"duplicate (track, date, race_number): {dup}")
    orphans = _scalar(conn, """
        SELECT count(*) FROM race_entries e LEFT JOIN races r ON r.id = e.race_id
        WHERE r.id IS NULL
    """)
    if orphans:
        failures.append(f"entries without a race: {orphans}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--database-url", default=None,
                    help="Target database (default: DATABASE_URL from settings).")
    ap.add_argument("--apply", action="store_true",
                    help="Commit the merge. Without it the transaction is rolled back.")
    args = ap.parse_args()

    engine = create_engine(args.database_url or get_settings().database_url)
    print(f"database: {engine.url.render_as_string(hide_password=True)}")
    with engine.connect() as conn:
        trans = conn.begin()
        before = {k: _scalar(conn, sql) for k, sql in PRESERVED_TOTALS.items()}
        before_counts = {
            t: _scalar(conn, f"SELECT count(*) FROM {t}")
            for t in ("tracks", "races", "race_entries", "multi_race_pools")
        }
        stats = merge(conn)
        failures = check(conn, before)
        after_counts = {
            t: _scalar(conn, f"SELECT count(*) FROM {t}") for t in before_counts
        }
        for key, value in stats.items():
            print(f"  {key:34s} {value}")
        for t in before_counts:
            print(f"  {t + ' rows':34s} {before_counts[t]} -> {after_counts[t]}")
        for name in before:
            print(f"  {name + ' (preserved)':34s} {before[name]}")
        if failures:
            trans.rollback()
            print("CHECKS FAILED, rolled back:")
            for f in failures:
                print(f"  - {f}")
            return 1
        if args.apply:
            trans.commit()
            print("checks passed; committed.")
        else:
            trans.rollback()
            print("checks passed; dry run rolled back (use --apply to commit).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
