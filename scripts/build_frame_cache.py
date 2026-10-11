"""Build the per-race feature frame once for every resulted race, in parallel.

``build_race_frame`` issues ~200 queries per race, and every training run
rebuilds frames for its whole window. For historical races the training
cutoff (``race_cutoff``) and the inference cutoff (``prediction_cutoff``)
coincide, so one frame per race serves training and scoring alike.
``scripts/accuracy_experiment.py --frame-cache`` reuses the result.

The cache is only valid for the database state and code it was built
from: rebuild it after scraping, crawling or any feature change.

Usage:
  uv run python scripts/build_frame_cache.py --from 2023-01-01 --to 2026-10-02 \
      --workers 4 --out logs/experiments/frames.pkl
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd


def _build_chunk(job):
    """Build one chunk of race frames and save it; skip if already saved."""
    path, race_ids = job
    path = Path(path)
    if path.exists():
        return path, len(race_ids)
    from ganyan.db.models import Race
    from ganyan.db.session import get_session
    from ganyan.predictor.ml import features as ml_features
    from ganyan.time import race_cutoff

    # The AGF-reliability table is one aggregate over all races before a
    # date; on a static DB it is identical for every race that day, and
    # chunks hold contiguous dates, so memoise it per date.
    original = ml_features.precompute_agf_reliability_table
    memo = {}

    def memoised(session, before_date=None, min_sample=30):
        key = (before_date, min_sample)
        if key not in memo:
            memo[key] = original(session, before_date=before_date, min_sample=min_sample)
        return memo[key]

    ml_features.precompute_agf_reliability_table = memoised
    session = get_session()
    frames = []
    try:
        for race_id in race_ids:
            race = session.get(Race, race_id)
            frame = ml_features.build_race_frame(session, race_id, as_of=race_cutoff(race))
            frame["race_id"] = race_id
            frames.append(frame)
            session.expire_all()
    finally:
        session.close()
    tmp = path.with_suffix(".tmp")
    pd.concat(frames, ignore_index=True).to_pickle(tmp)
    tmp.replace(path)
    return path, len(race_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from ganyan.db.models import Race, RaceStatus
    from ganyan.db.session import get_session
    from ganyan.predictor.ml.artifacts import pipeline_digest

    session = get_session()
    rows = session.query(Race.id, Race.date).filter(
        Race.status == RaceStatus.resulted,
        Race.date >= date.fromisoformat(args.start),
        Race.date <= date.fromisoformat(args.end),
    ).order_by(Race.date, Race.id).all()
    session.close()
    # Contiguous blocks of ~7 days: same-day races share the memoised
    # AGF-reliability table, and saved chunks make the build resumable.
    out = Path(args.out)
    chunk_dir = out.with_suffix(".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    dates = sorted({d for _, d in rows})
    block = {d: i // 7 for i, d in enumerate(dates)}
    grouped = {}
    for race_id, d in rows:
        grouped.setdefault(block[d], []).append(race_id)
    jobs = [(str(chunk_dir / f"chunk_{b:04d}.pkl"), ids) for b, ids in sorted(grouped.items())]
    started = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (_, n) in enumerate(pool.map(_build_chunk, jobs), 1):
            done += n
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} chunks, {done} races, "
                      f"{time.time() - started:.0f}s", flush=True)
    cache = pd.concat([pd.read_pickle(path) for path, _ in jobs], ignore_index=True)
    race_ids = [race_id for race_id, _ in rows]
    cache.attrs["pipeline_sha256"] = pipeline_digest()
    cache.to_pickle(out)
    print(f"wrote {len(race_ids)} races, {len(cache)} rows to {out}")


if __name__ == "__main__":
    main()
