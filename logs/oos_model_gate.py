"""Per-MODEL OOS gate: paired live-vs-candidate comparison on 2025+ history.

This is the invariant-#7 swap gate.  It exists because
``logs/discordance_oos_backtest.py`` IGNORES ``--model`` — that script
always scores the production EnsemblePredictor, so it can never compare
two single models (discovered 2026-08-13 when a candidate and the live
model returned bit-identical "gate" numbers).

For each resulted race in the fixed OOS window (both models must be
trained strictly AFTER the window ends):
  - MLPredictor(top-1) for LIVE and CANDIDATE on the same race
  - paired hit/miss vs actual winner -> McNemar exact test
  - AGF-favorite baseline for context

Swap rule: candidate must show >= +1.0pp OOS top-1 over live.

Usage:
  uv run python logs/oos_model_gate.py --candidate candidates/lightgbm_ranker_v2_20260813
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date
from pathlib import Path

from sqlalchemy import select, text

from ganyan.db.models import RaceEntry
from ganyan.db.session import get_session
from ganyan.predictor.ml import MLPredictor, load_latest_model

FROM_DATE = date(2025, 1, 1)
TO_DATE = date(2026, 1, 30)  # live model trained from 2026-02-05 -> OOS before that
PROGRESS_EVERY = 400

MIN_WINDOW_DAYS = 365
MIN_RACE_COUNT = 1500


def assert_min_window(from_date: date, to_date: date, n_races: int) -> None:
    days = (to_date - from_date).days + 1
    assert days >= MIN_WINDOW_DAYS, (
        f"OOS window too short: {days} days < {MIN_WINDOW_DAYS}"
    )
    assert n_races >= MIN_RACE_COUNT, (
        f"OOS race count too low: {n_races} < {MIN_RACE_COUNT}"
    )


def load_named(name: str):
    """Resolve 'candidates/foo' -> (models/candidates, foo); 'foo' -> (models, foo)."""
    p = Path(name)
    model_dir = Path("models") / p.parent if str(p.parent) != "." else Path("models")
    return load_latest_model(model_dir=model_dir, model_name=p.name)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value via binomial(b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="model name relative to models/, e.g. candidates/lightgbm_ranker_v2_20260813")
    ap.add_argument("--live", default="lightgbm_ranker")
    args = ap.parse_args()

    print(f"=== oos_model_gate start {time.strftime('%H:%M:%S')} ===", flush=True)
    print(f"window: {FROM_DATE} to {TO_DATE}")
    print(f"live={args.live}  candidate={args.candidate}")

    s = get_session()
    rows = s.execute(text("""
        SELECT r.id
        FROM races r
        JOIN race_entries re ON re.race_id = r.id
        WHERE r.date >= :from_date AND r.date <= :to_date
          AND r.status = 'resulted'
        GROUP BY r.id, r.date
        HAVING COUNT(re.id) >= 4
        ORDER BY r.date, r.id
    """), {"from_date": FROM_DATE, "to_date": TO_DATE}).fetchall()
    races = [r[0] for r in rows]
    print(f"OOS resulted races in window: {len(races)}")
    assert_min_window(FROM_DATE, TO_DATE, len(races))

    live_model = load_named(args.live)
    cand_model = load_named(args.candidate)
    live = MLPredictor(s, model=live_model)
    cand = MLPredictor(s, model=cand_model)
    print(f"live features={len(live_model.feature_columns)}  "
          f"candidate features={len(cand_model.feature_columns)}")

    n = live_hits = cand_hits = agf_hits = agf_n = 0
    b = c = 0  # b: live hit & cand miss; c: live miss & cand hit
    t0 = time.time()
    for i, rid in enumerate(races, 1):
        winner_hids = {row[0] for row in s.execute(
            select(RaceEntry.horse_id).where(
                RaceEntry.race_id == rid,
                RaceEntry.finish_position == 1,
            )
        ).all()}
        if not winner_hids:
            continue
        try:
            live_preds = live.predict(rid)
            cand_preds = cand.predict(rid)
        except Exception as exc:  # noqa: BLE001 — skip unpredictable races, count below
            print(f"  race {rid}: predict failed: {exc}", flush=True)
            continue
        if not live_preds or not cand_preds:
            continue
        n += 1
        lh = live_preds[0].horse_id in winner_hids
        ch = cand_preds[0].horse_id in winner_hids
        live_hits += lh
        cand_hits += ch
        if lh and not ch:
            b += 1
        elif ch and not lh:
            c += 1
        # AGF favorite baseline (deterministic tiebreak on horse_id)
        arow = s.execute(text("""
            SELECT horse_id FROM race_entries
            WHERE race_id = :rid AND agf IS NOT NULL
            ORDER BY agf DESC, horse_id ASC LIMIT 1
        """), {"rid": rid}).fetchone()
        if arow is not None:
            agf_n += 1
            agf_hits += arow[0] in winner_hids
        if i % PROGRESS_EVERY == 0:
            el = (time.time() - t0) / 60
            print(f"  [{i}/{len(races)}] {el:.1f}min  "
                  f"live={live_hits/max(n,1)*100:.2f}%  cand={cand_hits/max(n,1)*100:.2f}%",
                  flush=True)

    live_pct = live_hits / n * 100 if n else 0.0
    cand_pct = cand_hits / n * 100 if n else 0.0
    delta_pp = cand_pct - live_pct
    p = mcnemar_exact(b, c)
    swap = delta_pp >= 1.0

    print(f"\n=== done in {(time.time()-t0)/60:.1f} min ===")
    print(f"paired races scored: {n}")
    print(f"LIVE      top1 = {live_pct:.2f}%  ({live_hits}/{n})")
    print(f"CANDIDATE top1 = {cand_pct:.2f}%  ({cand_hits}/{n})")
    if agf_n:
        print(f"AGF FAV   top1 = {agf_hits/agf_n*100:.2f}%  ({agf_hits}/{agf_n})")
    print(f"delta = {delta_pp:+.2f}pp   discordant pairs: live-only={b}, cand-only={c}")
    print(f"McNemar exact p = {p:.4f}")
    print(f"GATE (>= +1.0pp): {'SWAP' if swap else 'NO SWAP — live model stays'}")

    out = {
        "window": [str(FROM_DATE), str(TO_DATE)], "n": n,
        "live": {"name": args.live, "top1_hits": live_hits, "top1_pct": live_pct},
        "candidate": {"name": args.candidate, "top1_hits": cand_hits, "top1_pct": cand_pct},
        "agf_baseline": {"n": agf_n, "top1_hits": agf_hits},
        "delta_pp": delta_pp, "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p,
        "swap": swap,
    }
    Path("logs/oos_model_gate_results.json").write_text(json.dumps(out, indent=2))
    print("results saved: logs/oos_model_gate_results.json")


if __name__ == "__main__":
    main()
