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


def _check_candidate(metadata):
    if metadata.get("pipeline_sha256") != pipeline_digest():
        raise ValueError("Candidate was trained with a different or unrecorded feature pipeline")
    if metadata.get("feature_schema") != 2:
        raise ValueError("Candidate must use corrected feature schema 2")


def evaluate_gate(session, live_model, candidate_model, start, end, *, mode="forward"):
    _check_candidate(candidate_model.metadata)
    for model in (live_model, candidate_model):
        validate_window(model.metadata, start, end, mode)
    live, candidate = MLPredictor(session, live_model), MLPredictor(session, candidate_model)
    result = paired_top1(session, live.predict, candidate.predict, start, end, mode=mode)
    result.update(live_artifact=live_model.artifact,
                  candidate_artifact=candidate_model.artifact)
    return result


def evaluate_ensemble_gate(session, live_models, candidate_models, start, end, *, mode="forward"):
    """Same paired forward test as :func:`evaluate_gate`, for whole head sets.

    Diagnostic for ensembles: promotion still goes head-by-head through
    ``model-promote``.
    """
    from ganyan.predictor.ml.ensemble import EnsemblePredictor

    for model in candidate_models:
        _check_candidate(model.metadata)
    for model in (*live_models, *candidate_models):
        validate_window(model.metadata, start, end, mode)
    live = EnsemblePredictor(session, models=list(live_models))
    candidate = EnsemblePredictor(session, models=list(candidate_models))
    result = paired_top1(session, live.predict, candidate.predict, start, end, mode=mode)
    result.update(
        live_artifact={m.metadata["approved_head"]: m.artifact for m in live_models},
        candidate_artifact={m.metadata["approved_head"]: m.artifact for m in candidate_models},
        swap=False,
    )
    return result


def write_manifest(root):
    """Write ``active.json`` listing every head trained into ``root``.

    Used to assemble a candidate ensemble for :func:`evaluate_ensemble_gate`;
    refuses the production model directory.
    """
    from ganyan.predictor.ml.artifacts import candidate_directory, digest

    root = candidate_directory(root)
    heads = {}
    for meta in sorted(root.glob("*.meta.json")):
        name = meta.name[: -len(".meta.json")]
        weights = [p for p in (root / f"{name}.txt", root / f"{name}.npz") if p.exists()]
        if len(weights) != 1:
            raise ValueError(f"Expected exactly one weights file for head {name}")
        heads[name] = {"model": weights[0].name, "metadata": meta.name,
                       "model_sha256": digest(weights[0]), "metadata_sha256": digest(meta)}
    if not heads:
        raise ValueError(f"No trained heads in {root}")
    path = root / "active.json"
    path.write_text(json.dumps({"version": 1, "heads": heads}, indent=2) + "\n")
    return path


def paired_top1(session, live_predict, candidate_predict, start, end, *, mode="forward"):
    races = (session.query(Race).filter(Race.status == RaceStatus.resulted,
             Race.date >= start, Race.date <= end).order_by(Race.date, Race.id).all())
    scored, excluded = [], []
    b = c = live_hits = candidate_hits = agf_hits = agf_n = 0
    live_top3 = candidate_top3 = 0
    for race in races:
        entries = [e for e in race.entries if not e.scratched]
        winners = {e.horse_id for e in entries if e.finish_position == 1}
        if len(entries) < 4 or not winners or any(e.finish_position is None for e in entries):
            excluded.append({"race_id": race.id, "reason": "incomplete result field"})
            continue
        # Prediction failures fail the gate, never silently select an easier cohort.
        lp, cp = live_predict(race.id), candidate_predict(race.id)
        expected = {e.horse_id for e in entries}
        if {p.horse_id for p in lp} != expected or {p.horse_id for p in cp} != expected:
            raise ValueError(f"Incomplete predictions for race {race.id}")
        if any(not math.isfinite(_probability(p)) for p in [*lp, *cp]):
            raise ValueError("Non-finite probabilities")
        lh, ch = lp[0].horse_id in winners, cp[0].horse_id in winners
        live_hits += lh
        candidate_hits += ch
        live_top3 += any(p.horse_id in winners for p in lp[:3])
        candidate_top3 += any(p.horse_id in winners for p in cp[:3])
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
            "live_top3_pct": 100 * live_top3 / len(scored),
            "candidate_top3_pct": 100 * candidate_top3 / len(scored),
            "agf_baseline": {"n": agf_n, "hits": agf_hits},
            "delta_pp": delta, "mcnemar_p": p_value,
            "swap": mode == "forward" and delta >= 1.0 and p_value < 0.05,
            "pipeline_sha256": pipeline_digest()}


def _probability(prediction):
    value = getattr(prediction, "probability", None)
    return prediction.mean_probability if value is None else value


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
