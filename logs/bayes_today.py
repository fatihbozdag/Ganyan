"""Run the Bayes posterior on today's races and compare to actual results."""
import os
import argparse
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ganyan.db.models import Race, Track
from ganyan.predictor.bayes.predictor import predict_from_posterior
from ganyan.predictor.bayes.trainer import load_posterior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", required=True)
    p.add_argument("--posterior", default="models/bayes_pl_v1")
    args = p.parse_args()

    target_date = date.fromisoformat(args.date)
    base = Path(args.posterior)

    idata, frame = load_posterior(base)

    eng = create_engine(os.environ.get('DATABASE_URL', "sqlite:///data/ganyan.db"))
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
        variants = compute_track_variants(s, to_date=target_date)
        speed_history = build_horse_speed_history(s, variants, to_date=target_date)
        workout_history = build_horse_workout_history(s, to_date=target_date)
        pace_baseline = compute_pace_baseline(s, to_date=target_date)
        pace_history = build_horse_pace_history(s, pace_baseline, to_date=target_date)

        races = s.execute(
            select(Race).join(Track).where(Race.date == target_date)
            .order_by(Race.date, Race.race_number)
        ).scalars().all()

        bayes_top1_hits = 0
        lgbm_top1_hits = 0
        bayes_top3_hits = 0
        lgbm_top3_hits = 0
        graded = 0

        print(
            f"\n{'Track':<10} {'R':>2}  "
            f"{'Winner':<22}  {'LGBM top-1':<22}  {'Bayes top-1':<22}  "
            f"{'L?':>2} {'B?':>2}  {'Bayes mean±90%CI':>22}"
        )
        print("-" * 120)

        for r in races:
            entries = list(r.entries)
            finishers = [e for e in entries if e.finish_position is not None]
            if not finishers or len(entries) < 3:
                continue

            try:
                winner = next(e for e in finishers if e.finish_position == 1)
            except StopIteration:
                continue
            top3_actual = sorted(finishers, key=lambda e: e.finish_position)[:3]
            top3_actual_ids = {e.horse_id for e in top3_actual}

            # Bayes prediction
            race_in = {
                "horse_ids": [e.horse_id for e in entries],
                "horse_names": [e.horse.name if e.horse else "?" for e in entries],
                "jockeys": [e.jockey or "" for e in entries],
                "sires": [(e.horse.sire or "") if e.horse else "" for e in entries],
                "track_id": r.track_id,
                "distance_meters": r.distance_meters or 0,
                "agfs": [
                    float(e.agf) if e.agf is not None else 0.0
                    for e in entries
                ],
                "kgss": [
                    float(e.kgs) if e.kgs is not None else 0.0
                    for e in entries
                ],
                "s20s": [
                    float(e.s20) if e.s20 is not None else 0.0
                    for e in entries
                ],
                "last_sixes": [e.last_six or "" for e in entries],
                "speeds": [
                    horse_speed_score(speed_history, e.horse_id, r.date) or 0.0
                    for e in entries
                ],
                "workouts": [
                    horse_workout_score(workout_history, e.horse_id, r.date) or 0.0
                    for e in entries
                ],
                "paces": [
                    horse_pace_score(pace_history, e.horse_id, r.date) or 0.0
                    for e in entries
                ],
            }
            bpreds = predict_from_posterior(idata, frame, race_in)
            b_top1 = bpreds[0]
            bayes_winner_hit = b_top1.horse_id == winner.horse_id
            bayes_top3 = {p.horse_id for p in bpreds[:3]}
            bayes_set_overlap = bayes_top3 & top3_actual_ids

            # LGBM ranking — sort by predicted_probability desc
            lgbm_sorted = sorted(
                entries,
                key=lambda e: float(e.predicted_probability or 0.0),
                reverse=True,
            )
            l_top1 = lgbm_sorted[0]
            lgbm_winner_hit = l_top1.horse_id == winner.horse_id
            lgbm_top3 = {e.horse_id for e in lgbm_sorted[:3]}
            lgbm_set_overlap = lgbm_top3 & top3_actual_ids

            graded += 1
            bayes_top1_hits += int(bayes_winner_hit)
            lgbm_top1_hits += int(lgbm_winner_hit)
            bayes_top3_hits += int(winner.horse_id in bayes_top3)
            lgbm_top3_hits += int(winner.horse_id in lgbm_top3)

            track = r.track.name if r.track else "?"
            l_name = (l_top1.horse.name if l_top1.horse else "?")[:22]
            b_name = b_top1.horse_name[:22]
            w_name = (winner.horse.name if winner.horse else "?")[:22]
            ci = f"{b_top1.mean_prob:.0%} [{b_top1.lo_5:.0%}-{b_top1.hi_95:.0%}]"
            print(
                f"{track:<10} {r.race_number:>2}  "
                f"{w_name:<22}  {l_name:<22}  {b_name:<22}  "
                f"{'✓' if lgbm_winner_hit else '✗':>2} "
                f"{'✓' if bayes_winner_hit else '✗':>2}  "
                f"{ci:>22}"
            )

        if graded:
            print("-" * 120)
            print(
                f"\nResults on {target_date}: {graded} graded races\n"
                f"  LGBM top-1: {lgbm_top1_hits}/{graded} ({lgbm_top1_hits/graded:.1%})\n"
                f"  Bayes top-1: {bayes_top1_hits}/{graded} ({bayes_top1_hits/graded:.1%})\n"
                f"  LGBM winner-in-top3: {lgbm_top3_hits}/{graded} ({lgbm_top3_hits/graded:.1%})\n"
                f"  Bayes winner-in-top3: {bayes_top3_hits}/{graded} ({bayes_top3_hits/graded:.1%})"
            )


if __name__ == "__main__":
    main()
