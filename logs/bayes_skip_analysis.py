"""Skip-decision analysis: with Bayes credible intervals, can we
identify and skip the bets we shouldn't make? If yes, the kept-bet
hit rate should rise sharply.
"""
import os
import argparse
from datetime import date
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from ganyan.db.models import Race
from ganyan.predictor.bayes.predictor import predict_from_posterior
from ganyan.predictor.bayes.trainer import load_posterior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_date", required=True)
    p.add_argument("--to", dest="to_date", required=True)
    p.add_argument("--posterior", default="models/bayes_pl_v1")
    args = p.parse_args()

    fd = date.fromisoformat(args.from_date)
    td = date.fromisoformat(args.to_date)
    base = Path(args.posterior)

    idata, frame = load_posterior(base)
    eng = create_engine(os.environ.get('DATABASE_URL', "sqlite:///data/ganyan.db"))

    rows = []  # (mean_prob, ci_width, lo_5, hit, lgbm_top1_hit)

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
        variants = compute_track_variants(s, to_date=td)
        speed_history = build_horse_speed_history(s, variants, to_date=td)
        workout_history = build_horse_workout_history(s, to_date=td)
        pace_baseline = compute_pace_baseline(s, to_date=td)
        pace_history = build_horse_pace_history(s, pace_baseline, to_date=td)

        races = s.execute(
            select(Race).where(Race.date >= fd, Race.date <= td)
            .order_by(Race.date, Race.race_number)
        ).scalars().all()

        for r in races:
            entries = list(r.entries)
            finishers = [e for e in entries if e.finish_position is not None]
            if len(finishers) < 3 or len(entries) < 3:
                continue
            try:
                winner_id = next(
                    e.horse_id for e in finishers if e.finish_position == 1
                )
            except StopIteration:
                continue

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
            preds = predict_from_posterior(idata, frame, race_in)
            top = preds[0]
            ci_width = top.hi_95 - top.lo_5
            bayes_hit = top.horse_id == winner_id

            lgbm_sorted = sorted(
                entries,
                key=lambda e: float(e.predicted_probability or 0.0),
                reverse=True,
            )
            lgbm_hit = lgbm_sorted[0].horse_id == winner_id

            rows.append((top.mean_prob, ci_width, top.lo_5, bayes_hit, lgbm_hit))

    arr = np.array(rows, dtype=[
        ("mean", "f8"), ("ci_w", "f8"), ("lo5", "f8"),
        ("bayes_hit", "?"), ("lgbm_hit", "?"),
    ])
    n = len(arr)
    print(f"\nHoldout {fd} → {td}: {n} races\n")
    print(f"Baseline: BAYES top-1 = {arr['bayes_hit'].mean():.1%}, "
          f"LGBM top-1 = {arr['lgbm_hit'].mean():.1%}\n")

    print(f"{'gate':<20} {'kept':>5} {'BAYES top-1':>12} {'LGBM top-1':>12}")
    print("-" * 56)

    gates = [
        ("all races", np.ones(n, dtype=bool)),
        ("mean ≥ 25%",       arr["mean"] >= 0.25),
        ("mean ≥ 30%",       arr["mean"] >= 0.30),
        ("mean ≥ 35%",       arr["mean"] >= 0.35),
        ("mean ≥ 40%",       arr["mean"] >= 0.40),
        ("lo_5 ≥ 15%",       arr["lo5"] >= 0.15),
        ("lo_5 ≥ 20%",       arr["lo5"] >= 0.20),
        ("CI width ≤ 30pp",  arr["ci_w"] <= 0.30),
        ("CI width ≤ 25pp",  arr["ci_w"] <= 0.25),
        ("mean≥30% ∧ CI≤25", (arr["mean"] >= 0.30) & (arr["ci_w"] <= 0.25)),
        ("mean≥35% ∧ lo5≥20",(arr["mean"] >= 0.35) & (arr["lo5"] >= 0.20)),
    ]
    for name, mask in gates:
        kept = mask.sum()
        if kept == 0:
            print(f"{name:<20} {kept:>5} {'—':>12} {'—':>12}")
            continue
        b = arr["bayes_hit"][mask].mean()
        l = arr["lgbm_hit"][mask].mean()
        print(f"{name:<20} {kept:>5} {b:>12.1%} {l:>12.1%}")


if __name__ == "__main__":
    main()
