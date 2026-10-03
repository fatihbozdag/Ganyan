"""Train and persist the Bayesian PL posterior."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Tuple

import arviz as az
from sqlalchemy.orm import Session

from ganyan.predictor.bayes.data import TrainingFrame, build_training_frame
from ganyan.predictor.bayes.model import (
    build_full_hierarchical_pl_model, fit_advi,
)


def save_posterior(idata: az.InferenceData, frame: TrainingFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    nc_path = base.with_suffix(".nc")
    idata.to_netcdf(str(nc_path))
    idx = {
        "feature_schema": frame.feature_schema,
        "horse_index": {str(k): v for k, v in frame.horse_index.items()},
        "jockey_index": frame.jockey_index,
        "sire_index": frame.sire_index,
        "track_dist_index": {
            f"{tid}_{db}": v for (tid, db), v in frame.track_dist_index.items()
        },
    }
    base.with_suffix(".indices.json").write_text(json.dumps(idx))


def load_posterior(base: Path) -> Tuple[az.InferenceData, TrainingFrame]:
    idata = az.from_netcdf(str(base.with_suffix(".nc")))
    raw = json.loads(base.with_suffix(".indices.json").read_text())
    frame = TrainingFrame(feature_schema=raw.get("feature_schema", 1))
    frame.horse_index = {int(k): v for k, v in raw["horse_index"].items()}
    frame.jockey_index = raw["jockey_index"]
    frame.sire_index = raw["sire_index"]
    track_dist: dict[tuple[int, int], int] = {}
    for key, v in raw["track_dist_index"].items():
        tid, db = key.split("_")
        track_dist[(int(tid), int(db))] = v
    frame.track_dist_index = track_dist
    return idata, frame


def train_full(
    session: Session,
    from_date: date,
    to_date: date,
    output_base: Path,
    n_iter: int = 60_000,
    seed: int = 0,
) -> None:
    frame = build_training_frame(session, from_date, to_date)
    model = build_full_hierarchical_pl_model(frame)
    idata = fit_advi(model, n_iter=n_iter, seed=seed)
    save_posterior(idata, frame, output_base)
