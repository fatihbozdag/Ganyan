"""Single-model top-1 hit-rate OOS backtest for wx-feature evaluation.

Usage:
    # Baseline (current ranker):
    uv run python logs/oos_top1_backtest.py --model-name lightgbm_ranker \
        --output logs/oos_baseline_<UTCdate>.json

    # wx_v1 candidate:
    uv run python logs/oos_top1_backtest.py --model-name lightgbm_ranker_wx_v1 \
        --output logs/oos_wx_v1_<UTCdate>.json

Enforces the V2-aftermath bar: >=365 days AND >=1500 races.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import text

from ganyan.db.session import get_session
from ganyan.predictor.ml.predictor import MLPredictor, load_latest_model

DEFAULT_FROM = date(2025, 1, 1)
DEFAULT_TO = date(2026, 4, 30)
STAKE_TL = 1.0
MIN_WINDOW_DAYS = 365
MIN_RACE_COUNT = 1500
PROGRESS_EVERY = 200


def _assert_min_window(from_date: date, to_date: date, n_races: int) -> None:
    days = (to_date - from_date).days
    assert days >= MIN_WINDOW_DAYS, (
        f"OOS window too short: {days} days < {MIN_WINDOW_DAYS}. "
        "V2 failure bar (in-sample +3.6pp, OOS +0.07pp) requires full-year window."
    )
    assert n_races >= MIN_RACE_COUNT, (
        f"OOS race count too low: {n_races} < {MIN_RACE_COUNT}."
    )


@dataclass
class _Stats:
    races: int = 0
    hits: int = 0
    stake: float = 0.0
    payout: float = 0.0
    pnl: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.races if self.races else 0.0

    @property
    def roi(self) -> float:
        return (self.payout - self.stake) / self.stake if self.stake else 0.0

    def bootstrap_ci(self, n_boot: int = 2000) -> tuple[float, float]:
        if not self.pnl:
            return (0.0, 0.0)
        rng = random.Random(42)
        n = len(self.pnl)
        rois = sorted(
            sum(self.pnl[rng.randrange(n)] for _ in range(n)) / (n * STAKE_TL)
            for _ in range(n_boot)
        )
        return (rois[int(0.025 * n_boot)], rois[int(0.975 * n_boot)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-model top-1 OOS backtest")
    ap.add_argument("--model-name", default="lightgbm_ranker",
                    help="Model filename stem under models/ (default: lightgbm_ranker)")
    ap.add_argument("--from-date", default=str(DEFAULT_FROM),
                    help=f"Start date inclusive (default: {DEFAULT_FROM})")
    ap.add_argument("--to-date", default=str(DEFAULT_TO),
                    help=f"End date inclusive (default: {DEFAULT_TO})")
    ap.add_argument("--output", default=None,
                    help="JSON output path (default: logs/oos_<model_name>_results.json)")
    args = ap.parse_args()

    from_date = date.fromisoformat(args.from_date)
    to_date = date.fromisoformat(args.to_date)
    output_path = args.output or f"logs/oos_{args.model_name}_results.json"

    started = time.time()
    print(f"=== oos_top1_backtest  model={args.model_name}  {from_date} to {to_date} ===")

    session = get_session()
    model = load_latest_model(model_name=args.model_name)
    print(f"loaded: {model.model_version}  features={len(model.feature_columns)}")

    predictor = MLPredictor(session, model=model)

    races = session.execute(text("""
        SELECT r.id, r.date, r.ganyan_payout_tl
        FROM races r
        JOIN race_entries re ON re.race_id = r.id
        WHERE r.date >= :from_date
          AND r.date <= :to_date
          AND r.status = 'resulted'
          AND r.ganyan_payout_tl IS NOT NULL
        GROUP BY r.id, r.date, r.ganyan_payout_tl
        HAVING COUNT(re.id) >= 4
        ORDER BY r.date, r.id
    """), {"from_date": from_date, "to_date": to_date}).fetchall()

    print(f"OOS races with payout: {len(races)}")
    _assert_min_window(from_date, to_date, len(races))

    stats = _Stats()
    skipped = 0

    for i, (race_id, race_date, payout_tl) in enumerate(races, start=1):
        try:
            preds = predictor.predict(race_id)
        except Exception as e:
            print(f"  race {race_id}: predict failed: {e}")
            skipped += 1
            continue
        if not preds:
            skipped += 1
            continue

        model_top1_hid = preds[0].horse_id

        entries = session.execute(text("""
            SELECT horse_id, finish_position FROM race_entries
            WHERE race_id = :rid AND horse_id IS NOT NULL
        """), {"rid": race_id}).fetchall()

        winners = [e[0] for e in entries if e[1] == 1]
        if not winners:
            skipped += 1
            continue
        actual_winner = winners[0]

        stats.races += 1
        stats.stake += STAKE_TL
        if model_top1_hid == actual_winner:
            stats.hits += 1
            payout = STAKE_TL * float(payout_tl)
            stats.payout += payout
            stats.pnl.append(payout - STAKE_TL)
        else:
            stats.pnl.append(-STAKE_TL)

        if i % PROGRESS_EVERY == 0:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0
            eta = (len(races) - i) / rate / 60 if rate > 0 else 0
            print(f"  [{i:>5}/{len(races)}] {elapsed/60:.1f}min  ETA={eta:.1f}min  "
                  f"top1={100*stats.hit_rate:.2f}%")

    elapsed = time.time() - started
    ci_lo, ci_hi = stats.bootstrap_ci()

    print(f"\n=== done in {elapsed/60:.1f} min ===")
    print(f"skipped: {skipped}")
    print(f"n={stats.races}  hit_rate={100*stats.hit_rate:.2f}% ({stats.hits}/{stats.races})")
    print(f"ROI={100*stats.roi:+.2f}%  CI95=[{100*ci_lo:+.2f}%, {100*ci_hi:+.2f}%]")

    result = {
        "model_name": args.model_name,
        "model_version": model.model_version,
        "n_features": len(model.feature_columns),
        "window": [str(from_date), str(to_date)],
        "n_races": stats.races,
        "n_hits": stats.hits,
        "hit_rate": stats.hit_rate,
        "roi": stats.roi,
        "roi_ci95": [ci_lo, ci_hi],
        "skipped": skipped,
        "elapsed_s": round(elapsed, 1),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"results saved: {output_path}")


if __name__ == "__main__":
    main()
