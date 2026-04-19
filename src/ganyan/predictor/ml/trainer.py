"""LightGBM LambdaRank trainer with temporal holdout."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ganyan.predictor.ml.features import (
    FEATURE_COLUMNS,
    GROUP_COLUMN,
    TrainingFrame,
    build_training_frame,
)


logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3].parent / "models"
DEFAULT_MODEL_BASENAME = "lightgbm_ranker"

_DEFAULT_LGBM_PARAMS: dict = {
    "objective": "lambdarank",
    # Evaluate on a richer range of cut-offs than just top-1.  NDCG@1
    # saturates after a single good tree when one feature (AGF) already
    # sorts the leader correctly most of the time; @5/@10 expose
    # residual ranking improvements the other features can still add.
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3, 5, 10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "verbose": -1,
    "lambdarank_truncation_level": 10,
}

# Default patience: generous so lambdarank has room to keep refining
# when the top-ranker metric plateaus.
_DEFAULT_EARLY_STOPPING_ROUNDS = 100


@dataclass
class TrainingResult:
    """What the trainer produces.

    Attributes
    ----------
    model_path:
        Absolute path to the saved LightGBM model file.
    metadata_path:
        Path to the JSON sidecar with feature list / training metadata.
    train_races, test_races:
        Race counts in each split.
    metrics:
        Holdout metrics — top-1 accuracy, top-3 accuracy, mean winner
        rank, NDCG@{1,3}.
    feature_importance:
        Feature → importance (gain) dict sorted descending.
    """

    model_path: Path
    metadata_path: Path
    train_races: int
    test_races: int
    metrics: dict[str, float]
    feature_importance: dict[str, float] = field(default_factory=dict)


def _temporal_split(
    frame: TrainingFrame, holdout_fraction: float,
) -> tuple[TrainingFrame, TrainingFrame]:
    """Split a training frame at the date quantile."""
    unique_dates = sorted(frame.race_dates.unique())
    if len(unique_dates) < 2:
        # Too little data to hold anything out.
        return frame, TrainingFrame(
            features=frame.features.iloc[:0].copy(),
            target=frame.target.iloc[:0].copy(),
            groups=frame.groups.iloc[:0].copy(),
            race_dates=frame.race_dates.iloc[:0].copy(),
        )

    cutoff_idx = max(1, int(len(unique_dates) * (1.0 - holdout_fraction)))
    cutoff_date = unique_dates[cutoff_idx]

    train_mask = frame.race_dates < cutoff_date
    test_mask = ~train_mask

    def _slice(mask: pd.Series) -> TrainingFrame:
        return TrainingFrame(
            features=frame.features[mask].reset_index(drop=True),
            target=frame.target[mask].reset_index(drop=True),
            groups=frame.groups[mask].reset_index(drop=True),
            race_dates=frame.race_dates[mask].reset_index(drop=True),
        )

    return _slice(train_mask), _slice(test_mask)


def _evaluate_ranker(
    model: lgb.Booster, frame: TrainingFrame,
) -> dict[str, float]:
    """Score a fitted booster against a labelled frame."""
    if frame.features.empty:
        return {
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "avg_winner_rank": 0.0,
            "n_races": 0,
        }

    scores = model.predict(frame.features)

    df = frame.features.copy()
    df["_score"] = scores
    df["_target"] = frame.target.values
    df["_race"] = frame.groups.values

    top1 = 0
    top3 = 0
    winner_ranks: list[int] = []
    for _race_id, race_df in df.groupby("_race", sort=False):
        ranked = race_df.sort_values("_score", ascending=False).reset_index(drop=True)
        # Winner = highest target in the race (ties broken by first appearance).
        winner_target = race_df["_target"].max()
        winners = ranked[ranked["_target"] == winner_target]
        if winners.empty:
            continue
        winner_rank = int(winners.index[0]) + 1  # 1-based
        winner_ranks.append(winner_rank)
        if winner_rank == 1:
            top1 += 1
        if winner_rank <= 3:
            top3 += 1

    n = len(winner_ranks)
    return {
        "top1_accuracy": (top1 / n) * 100 if n else 0.0,
        "top3_accuracy": (top3 / n) * 100 if n else 0.0,
        "avg_winner_rank": float(np.mean(winner_ranks)) if winner_ranks else 0.0,
        "n_races": n,
    }


def train_ranker(
    session: Session,
    *,
    from_date: date_type | None = None,
    to_date: date_type | None = None,
    holdout_fraction: float = 0.2,
    num_boost_round: int = 500,
    early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING_ROUNDS,
    model_dir: Path | None = None,
    model_name: str | None = None,
    params: dict | None = None,
) -> TrainingResult:
    """Fit a LightGBM LambdaRank model on resulted races.

    Parameters
    ----------
    from_date, to_date:
        Optional inclusive bounds on race dates.
    holdout_fraction:
        Fraction of the chronologically latest races reserved for the
        walk-forward test set.  The final model is fit on the train set
        with early stopping on the test set.
    num_boost_round:
        Maximum training rounds.  Early stopping may terminate sooner.
    params:
        LightGBM parameter overrides merged over the defaults.

    The trained booster is saved to ``<model_dir>/<name>.txt`` with a
    JSON sidecar (``<name>.meta.json``) capturing the feature column
    list, training bounds, and evaluation metrics.
    """
    effective_params = {**_DEFAULT_LGBM_PARAMS, **(params or {})}
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name or DEFAULT_MODEL_BASENAME

    frame = build_training_frame(
        session, from_date=from_date, to_date=to_date,
    )
    if frame.features.empty:
        raise RuntimeError(
            "No AGF-bearing resulted races found for training. "
            "Run `ganyan scrape --results-range --from <date>` first.",
        )

    train, test = _temporal_split(frame, holdout_fraction)

    train_dataset = lgb.Dataset(
        train.features,
        label=train.target,
        group=train.group_sizes(),
        free_raw_data=False,
    )
    valid_dataset = None
    callbacks = []
    if not test.features.empty:
        valid_dataset = lgb.Dataset(
            test.features,
            label=test.target,
            group=test.group_sizes(),
            reference=train_dataset,
            free_raw_data=False,
        )
        callbacks.append(
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        )
    callbacks.append(lgb.log_evaluation(period=0))  # silence per-iter chatter

    booster = lgb.train(
        params=effective_params,
        train_set=train_dataset,
        num_boost_round=num_boost_round,
        valid_sets=[train_dataset] + ([valid_dataset] if valid_dataset else []),
        valid_names=["train"] + (["test"] if valid_dataset else []),
        callbacks=callbacks,
    )

    metrics = _evaluate_ranker(booster, test)

    # Save model + metadata.
    model_path = model_dir / f"{model_name}.txt"
    meta_path = model_dir / f"{model_name}.meta.json"
    booster.save_model(str(model_path))

    importance_gain = booster.feature_importance(importance_type="gain")
    feature_importance = dict(
        sorted(
            zip(FEATURE_COLUMNS, importance_gain.tolist()),
            key=lambda kv: kv[1], reverse=True,
        )
    )

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "params": effective_params,
        "train_races": int(train.groups.nunique()),
        "test_races": int(test.groups.nunique()),
        "train_rows": int(len(train.features)),
        "test_rows": int(len(test.features)),
        "holdout_fraction": holdout_fraction,
        "from_date": from_date.isoformat() if from_date else None,
        "to_date": to_date.isoformat() if to_date else None,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "num_boost_round": num_boost_round,
        "best_iteration": booster.best_iteration or num_boost_round,
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    logger.info(
        "Trained LightGBM ranker — train_races=%d test_races=%d top1=%.2f%% top3=%.2f%%",
        metadata["train_races"],
        metadata["test_races"],
        metrics["top1_accuracy"],
        metrics["top3_accuracy"],
    )

    return TrainingResult(
        model_path=model_path,
        metadata_path=meta_path,
        train_races=metadata["train_races"],
        test_races=metadata["test_races"],
        metrics=metrics,
        feature_importance=feature_importance,
    )
