"""Paired accuracy gate on a declared, untouched evaluation window."""
import argparse
import json
import math
from datetime import date
from pathlib import Path

from ganyan.db.models import Race, RaceStatus
from ganyan.db.session import get_session
from ganyan.predictor.ml.predictor import MLPredictor, load_latest_model
from ganyan.predictor.ml.artifacts import model_root, pipeline_digest

MIN_WINDOW_DAYS = 365
MIN_RACE_COUNT = 1500


def assert_min_window(from_date, to_date, n_races):
    if (to_date - from_date).days + 1 < MIN_WINDOW_DAYS or n_races < MIN_RACE_COUNT:
        raise ValueError("Gate requires at least 365 days and 1500 scored paired races")


def validate_window(metadata, start, end, mode="forward"):
    try:
        first = date.fromisoformat(metadata["data_from_date"])
        last = date.fromisoformat(metadata["data_to_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing exact training/selection/calibration data provenance") from exc
    if first > last or start > end:
        raise ValueError("Invalid date interval")
    if mode == "forward":
        if last >= start:
            raise ValueError("Forward gate overlaps or precedes model development data")
    elif not (end < first or start > last):
        raise ValueError("Retrospective gate overlaps model development data")


def mcnemar_exact(b, c):
    n = b + c
    return min(1.0, 2 * (sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)) if n else 1.0


def load_named(name):
    path = Path(name)
    if path.parent == Path("."):
        return load_latest_model(model_name=path.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Use a model path relative to models/")
    return load_latest_model(model_dir=model_root() / path.parent, model_name=path.name)


def evaluate_gate(session, live_model, candidate_model, start, end, *, mode="forward"):
    if candidate_model.metadata.get("pipeline_sha256") != pipeline_digest():
        raise ValueError("Candidate was trained with a different or unrecorded feature pipeline")
    if candidate_model.metadata.get("feature_schema") != 2:
        raise ValueError("Candidate must use corrected feature schema 2")
    for model in (live_model, candidate_model):
        validate_window(model.metadata, start, end, mode)
    live, candidate = MLPredictor(session, live_model), MLPredictor(session, candidate_model)
    races = (session.query(Race).filter(Race.status == RaceStatus.resulted,
             Race.date >= start, Race.date <= end).order_by(Race.date, Race.id).all())
    scored, excluded = [], []
    b = c = live_hits = candidate_hits = agf_hits = agf_n = 0
    for race in races:
        entries = [e for e in race.entries if not e.scratched]
        winners = {e.horse_id for e in entries if e.finish_position == 1}
        if len(entries) < 4 or not winners or any(e.finish_position is None for e in entries):
            excluded.append({"race_id": race.id, "reason": "incomplete result field"})
            continue
        # Prediction failures fail the gate, never silently select an easier cohort.
        lp, cp = live.predict(race.id), candidate.predict(race.id)
        expected = {e.horse_id for e in entries}
        if {p.horse_id for p in lp} != expected or {p.horse_id for p in cp} != expected:
            raise ValueError(f"Incomplete predictions for race {race.id}")
        if any(not math.isfinite(p.probability) for p in [*lp, *cp]):
            raise ValueError("Non-finite probabilities")
        lh, ch = lp[0].horse_id in winners, cp[0].horse_id in winners
        live_hits += lh
        candidate_hits += ch
        b += lh and not ch
        c += ch and not lh
        market = [e for e in entries if e.agf is not None]
        if market:
            agf_n += 1
            agf_hits += max(market, key=lambda e: (float(e.agf), -e.horse_id)).horse_id in winners
        scored.append((race.id, race.date))
    if not scored:
        raise ValueError("No scored races")
    first, last = min(d for _, d in scored), max(d for _, d in scored)
    assert_min_window(first, last, len(scored))
    if len(excluded) / max(1, len(races)) > 0.05:
        raise ValueError("More than 5% of resulted races excluded")
    delta = 100 * (candidate_hits - live_hits) / len(scored)
    p_value = mcnemar_exact(b, c)
    return {"version": 2, "mode": mode, "n": len(scored),
            "coverage_days": (last - first).days + 1,
            "window": [str(first), str(last)], "requested_window": [str(start), str(end)],
            "race_ids": [rid for rid, _ in scored], "exclusions": excluded,
            "live_top1_pct": 100 * live_hits / len(scored),
            "candidate_top1_pct": 100 * candidate_hits / len(scored),
            "agf_baseline": {"n": agf_n, "hits": agf_hits},
            "delta_pp": delta, "mcnemar_p": p_value,
            "swap": mode == "forward" and delta >= 1.0 and p_value < 0.05,
            "live_artifact": live_model.artifact,
            "candidate_artifact": candidate_model.artifact,
            "pipeline_sha256": pipeline_digest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--live", default="lightgbm_ranker")
    parser.add_argument("--from", dest="start", type=date.fromisoformat, required=True)
    parser.add_argument("--to", dest="end", type=date.fromisoformat, required=True)
    parser.add_argument("--retrospective", action="store_true", help="Diagnostic only; cannot authorize promotion")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with get_session() as session:
        result = evaluate_gate(session, load_named(args.live), load_named(args.candidate),
                               args.start, args.end,
                               mode="retrospective" if args.retrospective else "forward")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"race_ids", "exclusions"}}, indent=2))


if __name__ == "__main__":
    main()
