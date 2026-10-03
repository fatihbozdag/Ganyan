"""LightGBM LambdaRank trainer with temporal holdout."""

from __future__ import annotations

import json
from ganyan.predictor.ml.artifacts import pipeline_digest
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ganyan.predictor.ml.features import (
    FEATURE_COLUMNS,
    TrainingFrame,
    build_training_frame,
)


logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3].parent / "models"
DEFAULT_MODEL_BASENAME = "lightgbm_ranker"


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        return out.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None

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


def temporal_kfold(
    frame: TrainingFrame, n_folds: int,
) -> list[tuple[TrainingFrame, TrainingFrame]]:
    """Generate ``n_folds`` expanding-window temporal splits.

    Returns a list of ``(train, test)`` ``TrainingFrame`` pairs where
    fold ``i`` uses the first ``i+1`` date chunks for training and the
    next chunk for test.  Strictly expanding — each fold's train set is
    a superset of the previous fold's.  Matches walk-forward evaluation
    practice (unlike random k-fold which leaks future info backward in
    time-series data).
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    unique_dates = sorted(frame.race_dates.unique())
    if len(unique_dates) < n_folds + 1:
        raise ValueError(
            f"Need at least {n_folds + 1} unique race dates for "
            f"{n_folds}-fold temporal CV (have {len(unique_dates)})",
        )

    # Split dates into n_folds+1 contiguous chunks.  First chunk is
    # always training (we need some history to fit anything).
    chunk_size = len(unique_dates) // (n_folds + 1)
    chunks: list[list] = []
    for i in range(n_folds + 1):
        start = i * chunk_size
        stop = (i + 1) * chunk_size if i < n_folds else len(unique_dates)
        chunks.append(unique_dates[start:stop])

    folds: list[tuple[TrainingFrame, TrainingFrame]] = []
    for fold_idx in range(1, n_folds + 1):
        train_dates = set().union(*[set(c) for c in chunks[:fold_idx]])
        test_dates = set(chunks[fold_idx])
        train_mask = frame.race_dates.isin(train_dates)
        test_mask = frame.race_dates.isin(test_dates)

        def _slice(mask):
            return TrainingFrame(
                features=frame.features[mask].reset_index(drop=True),
                target=frame.target[mask].reset_index(drop=True),
                ev_target=frame.ev_target[mask].reset_index(drop=True),
                finish_time_target=frame.finish_time_target[mask].reset_index(drop=True),
                groups=frame.groups[mask].reset_index(drop=True),
                race_dates=frame.race_dates[mask].reset_index(drop=True),
            )

        folds.append((_slice(train_mask), _slice(test_mask)))
    return folds


def cross_validate_ranker(
    session: Session,
    *,
    n_folds: int = 5,
    from_date: date_type | None = None,
    to_date: date_type | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = _DEFAULT_EARLY_STOPPING_ROUNDS,
    params: dict | None = None,
    race_type_prefix: str | None = None,
) -> dict:
    """Walk-forward K-fold CV to put a confidence interval on top-1.

    Fits a fresh LambdaRank per fold, evaluates on that fold's held-out
    chunk.  Returns per-fold metrics and aggregate mean / std / 95% CI
    across folds.  A single point estimate (like our 43.08% headline)
    hides its own uncertainty; ``(mean ± 1.96·std/√n)`` exposes it.
    """
    effective_params = {**_DEFAULT_LGBM_PARAMS, **(params or {})}
    frame = build_training_frame(
        session, from_date=from_date, to_date=to_date,
        race_type_prefix=race_type_prefix,
    )
    if frame.features.empty:
        raise RuntimeError("No training data for CV.")
    folds = temporal_kfold(frame, n_folds)
    per_fold: list[dict] = []
    for i, (train, test) in enumerate(folds):
        train, selection = _temporal_split(train, 0.2)
        if selection.features.empty:
            raise ValueError("Insufficient dates for inner temporal validation")
        train_ds = lgb.Dataset(
            train.features, label=train.target, group=train.group_sizes(),
            free_raw_data=False,
        )
        valid_ds = lgb.Dataset(
            selection.features, label=selection.target, group=selection.group_sizes(),
            reference=train_ds, free_raw_data=False,
        )
        booster = lgb.train(
            params=effective_params,
            train_set=train_ds,
            num_boost_round=num_boost_round,
            valid_sets=[train_ds, valid_ds],
            valid_names=["train", "test"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=False,
                    first_metric_only=False,
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        metrics = _evaluate_ranker(booster, test)
        metrics["fold"] = i + 1
        metrics["train_races"] = int(train.groups.nunique())
        metrics["test_races"] = int(test.groups.nunique())
        per_fold.append(metrics)

    top1s = [m["top1_accuracy"] for m in per_fold]
    top3s = [m["top3_accuracy"] for m in per_fold]
    return {
        "n_folds": n_folds,
        "per_fold": per_fold,
        "top1_mean": float(np.mean(top1s)),
        "top1_std": float(np.std(top1s, ddof=1)) if len(top1s) > 1 else 0.0,
        "top1_95_ci_halfwidth": (
            1.96 * float(np.std(top1s, ddof=1)) / (len(top1s) ** 0.5)
            if len(top1s) > 1 else 0.0
        ),
        "top3_mean": float(np.mean(top3s)),
        "top3_std": float(np.std(top3s, ddof=1)) if len(top3s) > 1 else 0.0,
    }


def _temporal_split(
    frame: TrainingFrame, holdout_fraction: float,
) -> tuple[TrainingFrame, TrainingFrame]:
    """Split a training frame at the date quantile."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    unique_dates = sorted(frame.race_dates.unique())
    if len(unique_dates) < 2:
        # Too little data to hold anything out.
        return frame, TrainingFrame(
            features=frame.features.iloc[:0].copy(),
            target=frame.target.iloc[:0].copy(),
            ev_target=frame.ev_target.iloc[:0].copy(),
            finish_time_target=frame.finish_time_target.iloc[:0].copy(),
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
            ev_target=frame.ev_target[mask].reset_index(drop=True),
            finish_time_target=frame.finish_time_target[mask].reset_index(drop=True),
            groups=frame.groups[mask].reset_index(drop=True),
            race_dates=frame.race_dates[mask].reset_index(drop=True),
        )

    return _slice(train_mask), _slice(test_mask)


def _fit_temperature(
    model: lgb.Booster, frame: TrainingFrame,
) -> float:
    """Fit a single scalar ``T`` minimising NLL of the within-race softmax.

    Objective is proper log-loss on the winner per race.  LambdaRank
    margins have an arbitrary scale; fitting one scalar temperature on
    the holdout (Guo et al. 2017, multi-class Platt) re-calibrates the
    aggregate confidence without touching the rank ordering.

    Search is a grid-seeded golden-section over ``log T ∈ [-5, 3]``
    (``T ∈ [0.007, 20]``) — wider than the previous [-3, 3] so the
    optimum isn't forced onto the floor when the booster emits small
    raw margins.  We also print NLL + Brier + ECE across a human-
    readable grid so operators can sanity-check whether the optimum is
    genuinely interior or is hugging a boundary.
    """
    if frame.features.empty:
        return 1.0

    raw = model.predict(frame.features)
    df = frame.features.copy()
    df["_score"] = raw
    df["_target"] = frame.target.values
    df["_race"] = frame.groups.values

    race_blocks: list[tuple[np.ndarray, int]] = []
    for _race_id, race_df in df.groupby("_race", sort=False):
        scores = race_df["_score"].to_numpy(dtype=float)
        targets = race_df["_target"].to_numpy()
        winner_target = targets.max()
        winner_rows = np.where(targets == winner_target)[0]
        if winner_rows.size == 0:
            continue
        race_blocks.append((scores, int(winner_rows[0])))
    if not race_blocks:
        return 1.0

    def probs_for(t: float, scores: np.ndarray) -> np.ndarray:
        shifted = scores / max(t, 1e-6)
        shifted -= shifted.max()
        exps = np.exp(shifted)
        return exps / exps.sum()

    def nll(t: float) -> float:
        total = 0.0
        for scores, widx in race_blocks:
            p = probs_for(t, scores)
            total -= float(np.log(max(p[widx], 1e-12)))
        return total

    def brier(t: float) -> float:
        total = 0.0
        for scores, widx in race_blocks:
            p = probs_for(t, scores)
            y = np.zeros_like(p)
            y[widx] = 1.0
            total += float(np.sum((p - y) ** 2))
        return total / len(race_blocks)


    # Emit a human-readable sweep so operators can see the curve shape
    # and whether the fitted optimum is at a boundary.
    grid = [0.01, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    logger.info("Temperature sweep (NLL / Brier):")
    for t in grid:
        logger.info(
            "  T=%.3f  NLL/race=%.4f  Brier/race=%.4f",
            t, nll(t) / len(race_blocks), brier(t),
        )

    # Golden-section on log-T in [log 0.007, log 20] = [-5, 3].
    a, b = -5.0, 3.0
    phi = (1 + 5**0.5) / 2
    resphi = 2 - phi
    tol = 1e-4
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = nll(float(np.exp(x1)))
    f2 = nll(float(np.exp(x2)))
    while abs(b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + resphi * (b - a)
            f1 = nll(float(np.exp(x1)))
        else:
            a, x1, f1 = x1, x2, f2
            x2 = b - resphi * (b - a)
            f2 = nll(float(np.exp(x2)))
    t_star = float(np.exp((a + b) / 2))
    t_final = max(0.01, min(t_star, 20.0))
    # Boundary-hugging is a sign the grid shape is pathological (often
    # because LambdaRank margins are tiny and NLL rewards extreme
    # peaking).  Log a visible warning so the operator can eyeball the
    # sweep table and decide whether to override T manually.
    if t_final <= 0.02 or t_final >= 19.0:
        logger.warning(
            "Temperature optimum at boundary (T=%.4f).  Check the sweep "
            "table above — the fit may have been dominated by tail races "
            "where the winner was severely misranked.  Consider manual "
            "T override or retraining after more data is available.",
            t_final,
        )
    logger.info(
        "Temperature search complete — fitted T=%.4f  NLL/race=%.4f  "
        "Brier/race=%.4f",
        t_final, nll(t_final) / len(race_blocks), brier(t_final),
    )
    return t_final


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

    return _ranking_metrics(df, "_score")


def _ranking_metrics(df, score_column, *, ascending=False):
    """Expected accuracy under a uniform tie break; independent of row/outcome order."""
    from math import comb
    top1 = top3 = 0.0
    ranks = []
    for _, group in df.groupby("_race", sort=False):
        scores = group[score_column]
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite prediction scores")
        winners = group["_target"] == group["_target"].max()
        best = scores[winners].min() if ascending else scores[winners].max()
        ahead = int((scores < best).sum() if ascending else (scores > best).sum())
        tied = int((scores == best).sum())
        winning_ties = int(((scores == best) & winners).sum())
        ranks.append(ahead + (tied + 1) / (winning_ties + 1))
        def hit_at(k):
            slots = min(tied, max(0, k - ahead))
            return 1.0 - comb(tied - winning_ties, slots) / comb(tied, slots) if slots else 0.0
        top1 += hit_at(1)
        top3 += hit_at(3)
    n = len(ranks)
    return {"top1_accuracy": 100 * top1 / n if n else 0.0,
            "top3_accuracy": 100 * top3 / n if n else 0.0,
            "avg_winner_rank": float(np.mean(ranks)) if n else 0.0,
            "n_races": n, "tie_policy": "uniform_expected"}


_EV_LGBM_PARAMS: dict = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "verbose": -1,
}


_FINISH_TIME_LGBM_PARAMS: dict = {
    # Huber loss instead of plain L2 — finish_time has heavy tails (DNFs
    # marked with absurd times, scratched-but-still-recorded entries)
    # that would dominate L2 gradients.
    "objective": "huber",
    "alpha": 0.9,  # Huber tuning param
    "metric": ["mae", "rmse"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "verbose": -1,
}


def _evaluate_finish_time_model(
    model: lgb.Booster, frame: TrainingFrame,
) -> dict[str, float]:
    """Evaluate finish-time regressor on holdout.

    Reports MAE/RMSE on the seconds target plus a winner-rank metric:
    if you sort horses ascending by predicted finish time, where does
    the actual race winner land?  That's the bridge from continuous
    time prediction to "who wins".
    """
    if frame.features.empty:
        return {"n_races": 0}
    y = frame.finish_time_target.to_numpy(dtype=float)
    mask = ~np.isnan(y)
    if not mask.any():
        return {"n_races": 0}

    preds = model.predict(frame.features)
    df = frame.features.copy()
    df["_pred"] = preds
    df["_true"] = y
    df["_target"] = frame.target.values
    df["_race"] = frame.groups.values

    err = preds[mask] - y[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # Rank-of-winner: per race, sort ascending by predicted seconds and
    # find where the actual winner (max rank_score) lands.  Same metric
    # as the rank objective so the heads are comparable head-to-head.
    return {**_ranking_metrics(df, "_pred", ascending=True),
            "mae_seconds": mae, "rmse_seconds": rmse}


def _evaluate_ev_model(
    model: lgb.Booster, frame: TrainingFrame,
) -> dict[str, float]:
    """Evaluate a regression-EV model: realised ROI of betting top-1 picks
    plus simple +EV-threshold strategies.

    The booster predicts per-horse expected return; we measure what
    actually happened if we'd flat-bet 1 unit on the model's pick per
    race.  Compare to flat-betting the AGF favourite as a baseline.
    """
    if frame.features.empty:
        return {"n_races": 0}
    scores = model.predict(frame.features)
    df = frame.features.copy()
    df["_pred_ev"] = scores
    df["_realised_ev"] = frame.ev_target.values
    df["_target"] = frame.target.values
    df["_race"] = frame.groups.values

    # Top-1 by predicted EV per race
    top1_realised = []
    positive_ev_realised = []
    for _, race_df in df.groupby("_race", sort=False):
        if race_df["_realised_ev"].isna().any():
            continue  # Missing pool excludes wins and losses together.
        tied = race_df[race_df["_pred_ev"] == race_df["_pred_ev"].max()]
        top1_realised.append(float(tied["_realised_ev"].mean()))
        # Bet on EVERY horse the model predicts +EV for
        positive = race_df[race_df["_pred_ev"] > 0.0]
        if not positive.empty:
            positive_ev_realised.extend(positive["_realised_ev"].tolist())

    n = len(top1_realised)
    return {
        "n_races": n,
        "top1_ev_avg": float(np.mean(top1_realised)) if n else 0.0,
        "top1_ev_total": float(np.sum(top1_realised)) if n else 0.0,
        "positive_ev_picks": len(positive_ev_realised),
        "positive_ev_realised_avg": (
            float(np.mean(positive_ev_realised)) if positive_ev_realised else 0.0
        ),
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
    exclude_features: list[str] | None = None,
    objective: str = "rank",
    race_type_prefix: str | None = None,
) -> TrainingResult:
    """Fit a LightGBM model on resulted races.

    ``objective="rank"`` (default) uses LambdaRank → predict ranking of
    horses within each race.  ``objective="ev"`` switches to a regression
    head whose target is the realised return per 1-TL bet — for the
    winner ``ganyan_payout - 1`` (AGF-implied if pool unknown), for
    losers ``-1``.  The EV model directly learns to find horses where
    win probability times payout exceeds the takeout, instead of just
    ranking horses by win likelihood.  At inference, sort by predicted
    EV and bet the +EV picks.
    """
    if objective not in {"rank", "ev", "finish_time"}:
        raise ValueError(
            f"objective must be 'rank', 'ev', or 'finish_time', got {objective!r}",
        )

    if objective == "ev":
        effective_params = {**_EV_LGBM_PARAMS, **(params or {})}
    elif objective == "finish_time":
        effective_params = {**_FINISH_TIME_LGBM_PARAMS, **(params or {})}
    else:
        effective_params = {**_DEFAULT_LGBM_PARAMS, **(params or {})}
    from ganyan.predictor.ml.artifacts import candidate_directory
    model_dir = candidate_directory(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = model_name or DEFAULT_MODEL_BASENAME
    if Path(model_name).name != model_name:
        raise ValueError("model_name must be a filename stem; use model_dir for directories")

    frame = build_training_frame(
        session, from_date=from_date, to_date=to_date,
        race_type_prefix=race_type_prefix,
    )
    if frame.features.empty:
        raise RuntimeError(
            "No AGF-bearing resulted races found for training. "
            "Run `ganyan scrape --results-range --from <date>` first.",
        )

    # Optionally strip specified columns before training.  Used by the
    # value-betting variant that must NOT see AGF during training so the
    # engineered features are forced to carry real weight.  The excluded
    # list is persisted so inference rebuilds the matrix the same way.
    excluded = list(exclude_features or [])
    if set(excluded) - set(FEATURE_COLUMNS):
        raise ValueError(f"Unknown excluded features: {sorted(set(excluded) - set(FEATURE_COLUMNS))}")
    if set(excluded) == set(FEATURE_COLUMNS):
        raise ValueError("At least one feature is required")
    if excluded:
        keep_cols = [c for c in FEATURE_COLUMNS if c not in excluded]
        frame = TrainingFrame(
            features=frame.features[keep_cols].reset_index(drop=True),
            target=frame.target,
            ev_target=frame.ev_target,
            finish_time_target=frame.finish_time_target,
            groups=frame.groups,
            race_dates=frame.race_dates,
        )
        logger.info(
            "Excluded %d feature(s) from training: %s",
            len(excluded), excluded,
        )

    train, test = _temporal_split(frame, holdout_fraction)
    train, selection = _temporal_split(train, 0.2)

    if objective in {"ev", "finish_time"}:
        # Regression: per-row label is either realised EV or actual
        # finish time.  No groups parameter (LightGBM regression
        # objective is independent of within-race grouping).
        if objective == "ev":
            train_label_full = train.ev_target
            valid_label_full = selection.ev_target
        else:
            train_label_full = train.finish_time_target
            valid_label_full = selection.finish_time_target
        # Drop any rows where the target couldn't be computed.
        train_mask = train_label_full.notna()
        train_features = train.features[train_mask].reset_index(drop=True)
        train_label = train_label_full[train_mask].reset_index(drop=True)
        train_dataset = lgb.Dataset(
            train_features, label=train_label, free_raw_data=False,
        )
        valid_dataset = None
        callbacks = []
        if not selection.features.empty:
            valid_mask = valid_label_full.notna()
            valid_features = selection.features[valid_mask].reset_index(drop=True)
            valid_label = valid_label_full[valid_mask].reset_index(drop=True)
            if not valid_features.empty:
                valid_dataset = lgb.Dataset(
                    valid_features, label=valid_label,
                    reference=train_dataset, free_raw_data=False,
                )
                callbacks.append(
                    lgb.early_stopping(
                        stopping_rounds=early_stopping_rounds,
                        verbose=False,
                        first_metric_only=False,
                    ),
                )
    else:
        train_dataset = lgb.Dataset(
            train.features,
            label=train.target,
            group=train.group_sizes(),
            free_raw_data=False,
        )
        valid_dataset = None
        callbacks = []
        if not selection.features.empty:
            valid_dataset = lgb.Dataset(
                selection.features,
                label=selection.target,
                group=selection.group_sizes(),
                reference=train_dataset,
                free_raw_data=False,
            )
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=False,
                    first_metric_only=False,
                ),
            )
    callbacks.append(lgb.log_evaluation(period=0))  # silence per-iter chatter

    booster = lgb.train(
        params=effective_params,
        train_set=train_dataset,
        num_boost_round=num_boost_round,
        valid_sets=[train_dataset] + ([valid_dataset] if valid_dataset else []),
        valid_names=["train"] + (["selection"] if valid_dataset else []),
        callbacks=callbacks,
    )

    if objective == "ev":
        metrics = _evaluate_ev_model(booster, test)
        temperature = 1.0
    elif objective == "finish_time":
        metrics = _evaluate_finish_time_model(booster, test)
        # The ensemble converts predicted seconds → within-race
        # probabilities via softmax(-z_score(seconds) / T).  Default
        # T=1; tunable later if convergence votes need re-balancing.
        temperature = 1.0
    else:
        metrics = _evaluate_ranker(booster, test)
        # Fit the within-race softmax temperature on holdout.  Persisted
        # alongside the booster; MLPredictor reads it back and applies
        # ``raw / T`` before the softmax so downstream Kelly sizing gets
        # calibrated (not arbitrary-scaled) probabilities.
        temperature = _fit_temperature(booster, selection)
    metrics["softmax_temperature"] = temperature
    metrics["objective"] = objective

    # Save model + metadata.
    model_path = model_dir / f"{model_name}.txt"
    meta_path = model_dir / f"{model_name}.meta.json"
    booster.save_model(str(model_path))

    trained_feature_cols = [
        c for c in FEATURE_COLUMNS if c not in excluded
    ]
    importance_gain = booster.feature_importance(importance_type="gain")
    feature_importance = dict(
        sorted(
            zip(trained_feature_cols, importance_gain.tolist()),
            key=lambda kv: kv[1], reverse=True,
        )
    )

    metadata = {
        "feature_columns": trained_feature_cols,
        "excluded_features": excluded,
        "objective": objective,
        "race_type_prefix": race_type_prefix,
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
        "softmax_temperature": temperature,
        "feature_schema": 2,
        "pipeline_sha256": pipeline_digest(),
        "data_from_date": str(frame.race_dates.min()),
        "data_to_date": str(frame.race_dates.max()),
        "training_from_date": str(train.race_dates.min()),
        "training_to_date": str(train.race_dates.max()),
        "selection_from_date": str(selection.race_dates.min()) if len(selection.race_dates) else None,
        "selection_to_date": str(selection.race_dates.max()) if len(selection.race_dates) else None,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    if objective == "ev":
        logger.info(
            "Trained EV regressor — train_races=%d test_races=%d "
            "top1_avg_ev=%.4f positive_ev_picks=%d positive_ev_avg=%.4f",
            metadata["train_races"],
            metadata["test_races"],
            metrics.get("top1_ev_avg", 0.0),
            metrics.get("positive_ev_picks", 0),
            metrics.get("positive_ev_realised_avg", 0.0),
        )
    elif objective == "finish_time":
        logger.info(
            "Trained finish-time regressor — train_races=%d test_races=%d "
            "MAE=%.3fs RMSE=%.3fs top1=%.2f%% top3=%.2f%%",
            metadata["train_races"],
            metadata["test_races"],
            metrics.get("mae_seconds", 0.0),
            metrics.get("rmse_seconds", 0.0),
            metrics.get("top1_accuracy", 0.0),
            metrics.get("top3_accuracy", 0.0),
        )
    else:
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
