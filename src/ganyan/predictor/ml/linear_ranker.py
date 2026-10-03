"""Linear rankers: Conditional Logit (McFadden) and Plackett-Luce.

Both are textbook racing models.  A single linear score
``s_i = β · x_i`` is computed per horse; winner / ordering
probabilities follow by normalisation.  They give the ensemble
*methodologically diverse* heads that don't share LightGBM's tree-
specific biases — useful because convergence across different model
families is a stronger agreement signal than convergence across the
same family with different hyperparams.

- **Conditional Logit** (Bolton & Chapman 1986, Benter 1994):
  ``P(horse i wins race) = exp(β·x_i) / Σ_j exp(β·x_j)``.
  MLE maximises ``Σ_race log(P(actual_winner wins))``.  Fits a single
  β vector jointly across all training races.

- **Plackett-Luce** generalises the same logit to ordered finishes:
  ``P(ordering) = Π_k exp(β·x_k) / Σ_{j∈remaining_k} exp(β·x_j)``.
  Uses the top-3 of each race for richer training signal.

Implementation: numpy gradient descent (Adam).  Small state, fits in
seconds even on 13k races.  No scipy / torch dependency.
"""

from __future__ import annotations

import json
from ganyan.predictor.ml.artifacts import pipeline_digest
import logging
from dataclasses import dataclass
from datetime import date as date_type
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from ganyan.predictor.ml.features import (
    TrainingFrame,
    build_training_frame,
)
from ganyan.predictor.ml.trainer import DEFAULT_MODEL_DIR, _temporal_split


logger = logging.getLogger(__name__)


@dataclass
class LinearRankerResult:
    model_path: Path
    metadata_path: Path
    train_races: int
    test_races: int
    metrics: dict
    coefficients: dict[str, float]


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise standardisation, returning ``(X_std, mean, std)`` so
    inference can apply the same transform.

    NaN-safe in three failure modes that real production data hits:
    1. **All-NaN columns** (e.g., ``late_agf_drift`` before snapshots
       have accumulated): mean → 0, std → 1, column contributes zero
       to the score regardless of input.
    2. **Constant columns** (zero variance): std → 1 to avoid divide-by-
       zero; the column contributes a constant offset that conditional
       logit's softmax-normalisation ignores.
    3. **Sparse columns** with some NaN: NaN values fill with the
       column mean (trained on non-NaN rows), preserving the others.

    After all that, defensively replace any remaining NaN/Inf with 0
    so downstream linear algebra stays finite.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std) | (std == 0), 1.0, std)
    filled = np.where(np.isnan(X), mean, X)
    out = (filled - mean) / std
    out = np.where(np.isfinite(out), out, 0.0)
    return out, mean, std


def _adam_step(g: np.ndarray, m: np.ndarray, v: np.ndarray, t: int,
               lr: float = 0.05, b1: float = 0.9, b2: float = 0.999,
               eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    return lr * m_hat / (np.sqrt(v_hat) + eps), m, v


def _iter_race_blocks(frame: TrainingFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slice the frame into ``[(X_race, rank_score_race), ...]`` blocks
    so conditional logit can loop per-race.  ``rank_score`` lets us
    recover the winner (``argmax``) and top-k ordering."""
    X = frame.features.to_numpy(dtype=float)
    y = frame.target.to_numpy()
    groups = frame.groups.to_numpy()
    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    # groups are contiguous after sort_values in build_training_frame.
    i = 0
    n = len(groups)
    while i < n:
        j = i
        gid = groups[i]
        while j < n and groups[j] == gid:
            j += 1
        blocks.append((X[i:j], y[i:j]))
        i = j
    return blocks


def _conditional_logit_nll(
    beta: np.ndarray, blocks: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, np.ndarray]:
    """Return (NLL, gradient) for winner-only conditional logit."""
    total = 0.0
    grad = np.zeros_like(beta)
    for Xr, yr in blocks:
        scores = Xr @ beta
        scores -= scores.max()
        exps = np.exp(scores)
        probs = exps / exps.sum()
        # Winner = max rank_score; break ties by first index.
        widx = int(np.argmax(yr))
        total -= float(np.log(max(probs[widx], 1e-12)))
        # Gradient: x_winner − Σ_i p_i x_i
        grad -= Xr[widx]
        grad += probs @ Xr
    return total, grad


def _plackett_luce_nll(
    beta: np.ndarray, blocks: list[tuple[np.ndarray, np.ndarray]],
    top_k: int = 3,
) -> tuple[float, np.ndarray]:
    """NLL under Plackett-Luce using the top-``top_k`` actual finishers."""
    total = 0.0
    grad = np.zeros_like(beta)
    for Xr, yr in blocks:
        n = len(yr)
        if n < 2:
            continue
        k = min(top_k, n - 1)
        # Order by actual finish (rank_score descending = position ascending).
        order = np.argsort(-yr)
        scores = Xr @ beta
        # Numerically stable: shift max.
        scores = scores - scores.max()
        remaining_mask = np.ones(n, dtype=bool)
        for step in range(k):
            idx = order[step]
            # P(finisher at step is winner among remaining)
            exps = np.exp(scores)
            exps[~remaining_mask] = 0.0
            denom = exps.sum()
            if denom <= 0:
                break
            p = exps[idx] / denom
            total -= float(np.log(max(p, 1e-12)))
            # Gradient contribution for this step
            probs = exps / denom
            grad -= Xr[idx]
            grad += probs @ Xr
            remaining_mask[idx] = False
    return total, grad


def train_conditional_logit(
    session: Session,
    *,
    from_date: date_type | None = None,
    to_date: date_type | None = None,
    holdout_fraction: float = 0.2,
    epochs: int = 200,
    lr: float = 0.05,
    model_dir: Path | None = None,
    model_name: str = "linear_conditional_logit",
    plackett_luce: bool = False,
    top_k: int = 3,
    race_type_prefix: str | None = None,
) -> LinearRankerResult:
    """Fit a Conditional-Logit (or Plackett-Luce) ranker via Adam.

    Saves weights + standardisation params as a JSON sidecar so the
    ensemble can load and apply inference without LightGBM boilerplate.
    """
    frame = build_training_frame(
        session, from_date=from_date, to_date=to_date,
        race_type_prefix=race_type_prefix,
    )
    if frame.features.empty:
        raise RuntimeError("No training data for linear ranker.")

    train, test = _temporal_split(frame, holdout_fraction)

    # Standardise on train; apply same transform to test.
    X_train = train.features.to_numpy(dtype=float)
    X_train_std, mean, std = _standardise(X_train)
    # Rebuild frames with standardised features so block iterator
    # yields z-scored matrices directly.
    import pandas as pd
    feature_cols = list(train.features.columns)
    train_frame = TrainingFrame(
        features=pd.DataFrame(X_train_std, columns=feature_cols),
        target=train.target,
        ev_target=train.ev_target,
        finish_time_target=train.finish_time_target,
        groups=train.groups,
        race_dates=train.race_dates,
    )
    X_test_raw = test.features.to_numpy(dtype=float)
    X_test_std = np.where(np.isnan(X_test_raw), mean, X_test_raw)
    X_test_std = (X_test_std - mean) / std
    test_frame = TrainingFrame(
        features=pd.DataFrame(X_test_std, columns=feature_cols),
        target=test.target,
        ev_target=test.ev_target,
        finish_time_target=test.finish_time_target,
        groups=test.groups,
        race_dates=test.race_dates,
    )

    train_blocks = _iter_race_blocks(train_frame)
    test_blocks = _iter_race_blocks(test_frame)

    beta = np.zeros(len(feature_cols))
    m_state = np.zeros_like(beta)
    v_state = np.zeros_like(beta)
    loss_fn = _plackett_luce_nll if plackett_luce else _conditional_logit_nll
    best_beta = beta.copy()
    best_loss = float("inf")
    patience = 20
    stale = 0
    for epoch in range(1, epochs + 1):
        if plackett_luce:
            nll, grad = loss_fn(beta, train_blocks, top_k=top_k)  # type: ignore[arg-type]
        else:
            nll, grad = loss_fn(beta, train_blocks)
        step, m_state, v_state = _adam_step(grad, m_state, v_state, epoch, lr=lr)
        beta = beta - step
        if nll < best_loss - 1e-6:
            best_loss = nll
            best_beta = beta.copy()
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    # Expected tie performance is deterministic and independent of row order.
    from ganyan.predictor.ml.trainer import _ranking_metrics
    evaluation = pd.DataFrame({"_race": test.groups.to_numpy(),
                               "_target": test.target.to_numpy(),
                               "_pred": X_test_std @ best_beta})
    metrics = _ranking_metrics(evaluation, "_pred")
    n_test = metrics["n_races"]
    metrics["final_nll"] = best_loss
    if Path(model_name).name != model_name:
        raise ValueError("Model name must be a basename")

    from ganyan.predictor.ml.artifacts import candidate_directory
    model_dir = candidate_directory(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    meta_path = model_dir / f"{model_name}.meta.json"
    npz_path = model_dir / f"{model_name}.npz"

    np.savez(
        npz_path, beta=best_beta, mean=mean, std=std,
        feature_columns=np.array(feature_cols),
    )
    from ganyan.predictor.ml.trainer import _git_sha
    from datetime import datetime, timezone

    metadata = {
        "model_family": "plackett_luce" if plackett_luce else "conditional_logit",
        "objective": "rank",  # compatible with ensemble rank-head contract
        "feature_columns": feature_cols,
        "train_races": int(train.groups.nunique()),
        "test_races": int(test.groups.nunique()),
        "metrics": metrics,
        "npz_path": npz_path.name,
        "standardisation": {"mean": mean.tolist(), "std": std.tolist()},
        "race_type_prefix": race_type_prefix,
        "feature_schema": 2,
        "pipeline_sha256": pipeline_digest(),
        "data_from_date": str(frame.race_dates.min()),
        "data_to_date": str(frame.race_dates.max()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    coefficients = dict(
        sorted(
            zip(feature_cols, best_beta.tolist()),
            key=lambda kv: abs(kv[1]), reverse=True,
        )
    )
    metadata["coefficients"] = coefficients
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    logger.info(
        "Trained %s: top1=%.2f%% top3=%.2f%% on %d test races",
        metadata["model_family"], metrics["top1_accuracy"],
        metrics["top3_accuracy"], n_test,
    )

    return LinearRankerResult(
        model_path=npz_path,
        metadata_path=meta_path,
        train_races=int(train.groups.nunique()),
        test_races=int(test.groups.nunique()),
        metrics=metrics,
        coefficients=coefficients,
    )
