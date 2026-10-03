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


def _build_chunk(race_ids):
    from ganyan.db.models import Race
    from ganyan.db.session import get_session
    from ganyan.predictor.ml.features import build_race_frame
    from ganyan.time import race_cutoff

    session = get_session()
    frames = []
    try:
        for race_id in race_ids:
            race = session.get(Race, race_id)
            frame = build_race_frame(session, race_id, as_of=race_cutoff(race))
            frame["race_id"] = race_id
            frames.append(frame)
            session.expire_all()
    finally:
        session.close()
    return frames


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
    race_ids = [r for (r,) in session.query(Race.id).filter(
        Race.status == RaceStatus.resulted,
        Race.date >= date.fromisoformat(args.start),
        Race.date <= date.fromisoformat(args.end),
    ).order_by(Race.date, Race.id)]
    session.close()
    chunks = [race_ids[i::args.workers * 8] for i in range(args.workers * 8)]
    started = time.time()
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(_build_chunk, chunks), 1):
            frames.extend(result)
            print(f"{i}/{len(chunks)} chunks, {len(frames)} races, "
                  f"{time.time() - started:.0f}s", flush=True)
    cache = pd.concat(frames, ignore_index=True)
    cache.attrs["pipeline_sha256"] = pipeline_digest()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache.to_pickle(out)
    print(f"wrote {len(race_ids)} races, {len(cache)} rows to {out}")


if __name__ == "__main__":
    main()
