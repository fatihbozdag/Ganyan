"""Holdout eval: Bayes posterior vs LightGBM ranker."""
import os
import argparse
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ganyan.db.models import Race
from ganyan.predictor.bayes.calibration import (
    brier_top1, log_likelihood_of_winner, top1_hit_rate,
)
from ganyan.predictor.bayes.predictor import predict_from_posterior
from ganyan.predictor.bayes.trainer import load_posterior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_date", required=True)
    p.add_argument("--to", dest="to_date", required=True)
    p.add_argument("--posterior", required=True)
    args = p.parse_args()

    fd = date.fromisoformat(args.from_date)
    td = date.fromisoformat(args.to_date)
    base = Path(args.posterior)

    print(f"Loading posterior from {base}…")
    idata, frame = load_posterior(base)
    print(
        f"  posterior horses={len(frame.horse_index)} "
        f"jockeys={len(frame.jockey_index)} sires={len(frame.sire_index)} "
        f"track_dist={len(frame.track_dist_index)}"
    )

    eng = create_engine(os.environ.get('DATABASE_URL', "sqlite:///data/ganyan.db"))
    bayes_races = []
    lgbm_races = []
    n_skipped = 0

    with Session(eng) as s:
        from ganyan.predictor.speed_figures import (
            build_horse_speed_history, compute_track_variants, horse_speed_score,
        )
        from ganyan.predictor.workouts import (
            build_horse_workout_history, horse_workout_score,
        )
        from ganyan.predictor.pace import (
            build_horse_pace_history, compute_pace_baseline, horse_pace_score,
        )
        print("Building speed-figure history (variants + per-horse) …")
        variants = compute_track_variants(s, to_date=td)
        speed_history = build_horse_speed_history(s, variants, to_date=td)
        workout_history = build_horse_workout_history(s, to_date=td)
        pace_baseline = compute_pace_baseline(s, to_date=td)
        pace_history = build_horse_pace_history(s, pace_baseline, to_date=td)
        print(f"  variants: {len(variants)} (track,bucket,date) cells; "
              f"horses with speed history: {len(speed_history)}; "
              f"horses with workouts: {len(workout_history)}; "
              f"horses with pace history: {len(pace_history)}")

        races = s.execute(
            select(Race).where(Race.date >= fd, Race.date <= td)
            .order_by(Race.date, Race.race_number)
        ).scalars().all()

        for r in races:
            finishers = [e for e in r.entries if e.finish_position is not None]
            if len(finishers) < 3:
                n_skipped += 1
                continue
            try:
                winner_id = next(
                    e.horse_id for e in finishers if e.finish_position == 1
                )
            except StopIteration:
                n_skipped += 1
                continue

            race_in = {
                "horse_ids": [e.horse_id for e in r.entries],
                "jockeys": [e.jockey or "" for e in r.entries],
                "sires": [
                    (e.horse.sire or "") if e.horse else ""
                    for e in r.entries
                ],
                "track_id": r.track_id,
                "distance_meters": r.distance_meters or 0,
                "agfs": [
                    float(e.agf) if e.agf is not None else 0.0
                    for e in r.entries
                ],
                "kgss": [
                    float(e.kgs) if e.kgs is not None else 0.0
                    for e in r.entries
                ],
                "s20s": [
                    float(e.s20) if e.s20 is not None else 0.0
                    for e in r.entries
                ],
                "last_sixes": [e.last_six or "" for e in r.entries],
                "speeds": [
                    horse_speed_score(speed_history, e.horse_id, r.date) or 0.0
                    for e in r.entries
                ],
                "workouts": [
                    horse_workout_score(workout_history, e.horse_id, r.date) or 0.0
                    for e in r.entries
                ],
                "paces": [
                    horse_pace_score(pace_history, e.horse_id, r.date) or 0.0
                    for e in r.entries
                ],
            }
            preds = predict_from_posterior(idata, frame, race_in)
            bayes_races.append([
                (p.mean_prob, p.horse_id == winner_id) for p in preds
            ])

            lgbm_probs = [
                (
                    float(e.predicted_probability or 0.0) / 100.0,
                    e.horse_id == winner_id,
                )
                for e in r.entries
            ]
            if any(p > 0 for p, _ in lgbm_probs) and abs(
                sum(p for p, _ in lgbm_probs) - 1.0
            ) < 0.05:
                lgbm_races.append(lgbm_probs)

    print(f"\nHoldout {fd} → {td}: bayes={len(bayes_races)} races, "
          f"lgbm={len(lgbm_races)} races, skipped={n_skipped}")
    print(f"\n{'metric':<15} {'BAYES':>10} {'LGBM':>10}")
    print("-" * 40)
    for label, fn in [
        ("top1_rate", top1_hit_rate),
        ("brier_top1", brier_top1),
        ("logL_winner", log_likelihood_of_winner),
    ]:
        b = fn(bayes_races)
        l = fn(lgbm_races)
        print(f"{label:<15} {b:>10.4f} {l:>10.4f}")


if __name__ == "__main__":
    main()
