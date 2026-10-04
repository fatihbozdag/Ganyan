"""Train named model arms on one cutoff and score them on a later window.

Every arm trains only on races dated <= ``--train-to`` (each head's
metadata records its exact data window), so a scoring window that starts
after the cutoff is a genuine forward test. Arms are scored on the same
races with the same exclusion rule as ``ml/gate.py``; pairwise McNemar
p-values compare each arm with the first (baseline) arm, and the AGF
favourite is reported as the market baseline.

Selection runs score a validation year inside the development data; the
final test year is scored once, with the arms fixed beforehand.

Usage:
  uv run python scripts/accuracy_experiment.py --train-to 2024-09-30 \
      --score-from 2024-10-01 --score-to 2025-09-30 --out logs/experiments/dev \
      --arms ranker_90d ranker_all ranker_all_h05
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

from ganyan.db.models import Race, RaceStatus
from ganyan.db.session import get_session
from ganyan.predictor.ml import ensemble as ensemble_mod
from ganyan.predictor.ml import features as features_mod
from ganyan.predictor.ml import predictor as predictor_mod
from ganyan.predictor.ml.ensemble import EnsemblePredictor, load_all_models
from ganyan.predictor.ml.features import build_race_frame
from ganyan.predictor.ml.gate import mcnemar_exact, write_manifest
from ganyan.predictor.ml.linear_ranker import train_conditional_logit
from ganyan.predictor.ml.predictor import MLPredictor, load_latest_model
from ganyan.predictor.ml.trainer import train_ranker

logger = logging.getLogger("accuracy_experiment")

SPECIALISTS = [
    ("Handikap", "lightgbm_spec_handikap"), ("Maiden", "lightgbm_spec_maiden"),
    ("ŞARTLI", "lightgbm_spec_sartli"), ("KV", "lightgbm_spec_kv"),
    ("SATIŞ", "lightgbm_spec_satis"), ("G", "lightgbm_spec_stakes"),
]
DEFAULT_WINDOW_DAYS = 90  # ganyan train's default window (cli/main.py)


def _ranker(session, root, start, end, *, name="lightgbm_ranker", holdout=0.2, **kw):
    train_ranker(session, from_date=start, to_date=end, holdout_fraction=holdout,
                 model_dir=root, model_name=name, **kw)


def _ensemble(session, root, end, *, short_start, long_start, holdout):
    """The production head set (approved_models.json, 10 heads)."""
    _ranker(session, root, short_start, end, holdout=holdout)
    _ranker(session, root, short_start, end, name="lightgbm_value", holdout=holdout,
            exclude_features=["agf_edge", "agf_raw"])
    _ranker(session, root, short_start, end, name="lightgbm_finish_time", holdout=holdout,
            objective="finish_time")
    for prefix, name in SPECIALISTS:
        try:
            _ranker(session, root, long_start, end, name=name, holdout=holdout,
                    race_type_prefix=prefix)
        except RuntimeError as exc:
            logger.warning("specialist %s skipped: %s", prefix, exc)
    train_conditional_logit(session, from_date=long_start, to_date=end,
                            holdout_fraction=holdout, model_dir=root,
                            model_name="linear_conditional_logit")


def build_arms(data_start, end):
    short = end - timedelta(days=DEFAULT_WINDOW_DAYS)
    return {
        # Single ranker, CLI defaults: last 90 days, 20% diagnostic holdout.
        "ranker_90d": ("single", lambda s, r: _ranker(s, r, short, end)),
        "ranker_all": ("single", lambda s, r: _ranker(s, r, data_start, end)),
        "ranker_all_h05": ("single", lambda s, r: _ranker(s, r, data_start, end, holdout=0.05)),
        # Ensemble with each command's defaults (ranker/value/finish 90 days,
        # specialists + linear all history).
        "ensemble_default": ("ensemble", lambda s, r: _ensemble(
            s, r, end, short_start=short, long_start=data_start, holdout=0.2)),
        "ensemble_all_h05": ("ensemble", lambda s, r: _ensemble(
            s, r, end, short_start=data_start, long_start=data_start, holdout=0.05)),
        # Same heads as ensemble_all_h05, ordered by mean probability
        # instead of the #1-vote count.
        "ensemble_all_h05_meanprob": ("ensemble_mean_prob", "ensemble_all_h05"),
    }


def score_window(session, start, end):
    races = (session.query(Race).filter(Race.status == RaceStatus.resulted,
             Race.date >= start, Race.date <= end).order_by(Race.date, Race.id).all())
    scored, excluded = [], 0
    for race in races:
        entries = [e for e in race.entries if not e.scratched]
        winners = {e.horse_id for e in entries if e.finish_position == 1}
        if len(entries) < 4 or not winners or any(e.finish_position is None for e in entries):
            excluded += 1
            continue
        market = [e for e in entries if e.agf is not None]
        agf_top = max(market, key=lambda e: (float(e.agf), -e.horse_id)).horse_id if market else None
        scored.append((race.id, race.date, winners, agf_top, {e.horse_id for e in entries}))
    return scored, excluded, len(races)


def run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = get_session()
    data_start = date.fromisoformat(args.data_from)
    train_to = date.fromisoformat(args.train_to)
    s_from, s_to = date.fromisoformat(args.score_from), date.fromisoformat(args.score_to)
    if s_from <= train_to:
        raise SystemExit("score window must start after --train-to")
    arms = build_arms(data_start, train_to)

    # Feature frames are a pure function of the (static) DB; cache them so
    # every arm sees identical inputs and the frame is built once per race.
    # For past races the training cutoff and the inference cutoff coincide.
    cache = {}
    if args.frame_cache:
        import pandas as pd
        from ganyan.predictor.ml.artifacts import pipeline_digest
        table = pd.read_pickle(args.frame_cache)
        if table.attrs.get("pipeline_sha256") != pipeline_digest():
            raise SystemExit("frame cache was built from different code; rebuild it")
        for race_id, frame in table.groupby("race_id", sort=False):
            cache[int(race_id)] = frame.drop(columns="race_id").reset_index(drop=True)
        logger.info("loaded %d cached race frames", len(cache))

    def cached_frame(sess, race_id, *, as_of=None):
        if race_id not in cache:
            cache[race_id] = build_race_frame(sess, race_id, as_of=as_of)
        return cache[race_id].copy()

    features_mod.build_race_frame = cached_frame
    ensemble_mod.build_race_frame = cached_frame
    predictor_mod.build_race_frame = cached_frame

    scored, n_excluded, n_races = score_window(session, s_from, s_to)
    if n_excluded / max(1, n_races) > 0.05:
        raise SystemExit(f"{n_excluded}/{n_races} races excluded (>5%)")
    hits = {}
    for name in args.arms:
        kind, train = arms[name]
        if isinstance(train, str):  # reuse another arm's trained heads
            source = train
            train = arms[source][1]
        else:
            source = name
        root = out / source
        if not (root / "active.json").exists():
            logger.info("training %s", source)
            train(session, root)
            write_manifest(root)
        if kind == "single":
            predict = MLPredictor(session, load_latest_model(model_dir=root, model_name="lightgbm_ranker")).predict
        else:
            aggregation = "mean_prob" if kind == "ensemble_mean_prob" else "convergence"
            predict = EnsemblePredictor(session, models=load_all_models(root),
                                        aggregation=aggregation).predict
        top1, top3 = [], []
        for race_id, _, winners, _, expected in scored:
            preds = predict(race_id)
            if {p.horse_id for p in preds} != expected:
                raise SystemExit(f"{name}: incomplete predictions for race {race_id}")
            top1.append(preds[0].horse_id in winners)
            top3.append(any(p.horse_id in winners for p in preds[:3]))
        hits[name] = (top1, top3)
        logger.info("%s top-1 %.2f%%", name, 100 * sum(top1) / len(top1))

    n = len(scored)
    agf = [agf_top is not None and agf_top in winners for _, _, winners, agf_top, _ in scored]
    base = args.arms[0]
    report = {"train_to": str(train_to), "data_from": str(data_start),
              "score_window": [str(s_from), str(s_to)], "n": n,
              "excluded": n_excluded, "coverage_days": (scored[-1][1] - scored[0][1]).days + 1,
              "agf_favourite_top1_pct": 100 * sum(agf) / n, "baseline_arm": base, "arms": {}}
    for name, (top1, top3) in hits.items():
        b = sum(x and not y for x, y in zip(hits[base][0], top1))
        c = sum(y and not x for x, y in zip(hits[base][0], top1))
        b_m = sum(x and not y for x, y in zip(agf, top1))
        c_m = sum(y and not x for x, y in zip(agf, top1))
        report["arms"][name] = {
            "top1_pct": 100 * sum(top1) / n, "top3_pct": 100 * sum(top3) / n,
            "delta_vs_baseline_pp": 100 * (sum(top1) - sum(hits[base][0])) / n,
            "mcnemar_p_vs_baseline": mcnemar_exact(b, c),
            "delta_vs_agf_pp": 100 * (sum(top1) - sum(agf)) / n,
            "mcnemar_p_vs_agf": mcnemar_exact(b_m, c_m),
        }
    path = out / f"report_{s_from}_{s_to}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-from", default="2023-01-01")
    parser.add_argument("--train-to", required=True)
    parser.add_argument("--score-from", required=True)
    parser.add_argument("--score-to", required=True)
    parser.add_argument("--out", required=True,
                        help="Directory under models/candidates or logs/experiments")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--frame-cache", help="Pickle from scripts/build_frame_cache.py")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
