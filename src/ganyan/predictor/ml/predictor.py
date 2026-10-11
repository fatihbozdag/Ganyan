"""Inference-time predictor that mirrors :class:`BayesianPredictor`'s API.

Loads a LightGBM booster from disk, builds the per-race feature matrix
with :func:`ml.features.build_race_frame`, and converts raw LightGBM
scores into well-behaved win probabilities via a within-race softmax.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sqlalchemy.orm import Session

from ganyan.db.models import Race
from ganyan.predictor.bayesian import Prediction
from ganyan.predictor.ml.features import FEATURE_COLUMNS, build_race_frame
from ganyan.predictor.ml.trainer import (
    DEFAULT_MODEL_BASENAME,
)


@dataclass
class LoadedModel:
    """A deserialised booster plus its training-time feature list."""

    booster: lgb.Booster
    feature_columns: list[str]
    model_version: str
    # Within-race softmax temperature fitted on the training holdout.
    # Defaults to ``1.0`` for legacy models trained before calibration
    # was added — those models are uncalibrated but still rank-usable.
    softmax_temperature: float = 1.0
    metadata: dict = field(default_factory=dict)
    artifact: dict = field(default_factory=dict)


def load_latest_model(
    model_dir: Path | None = None,
    model_name: str | None = None,
) -> LoadedModel:
    """Load the most recently saved booster + metadata.

    Raises :class:`FileNotFoundError` when no model has been trained yet.
    """
    from ganyan.predictor.ml.artifacts import approved_paths, artifact_identity, model_root
    model_name = model_name or DEFAULT_MODEL_BASENAME
    if model_dir is None:
        paths = approved_paths(model_name, model_root())
        model_path, meta_path = paths["model"], paths["metadata"]
    else:
        model_dir = Path(model_dir)
        model_path = model_dir / f"{model_name}.txt"
        meta_path = model_dir / f"{model_name}.meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run `ganyan train` first.",
        )
    metadata: dict = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
    booster = lgb.Booster(model_file=str(model_path))
    feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)
    best_iter = metadata.get("best_iteration")
    objective = metadata.get("objective") or "rank"
    # Distinct version string per objective so ensemble logs can tell
    # ``lightgbm-rank-it123`` apart from ``lightgbm-finish_time-it250``
    # at a glance, and the audit Predictions table shows which head
    # wrote each row.
    model_version = f"lightgbm-{objective}"
    if best_iter is not None:
        model_version = f"{model_version}-it{best_iter}"
    # Distinguish multiple finish-time/value models on disk by appending
    # the model's filename stem when it's not the canonical default.
    if model_name not in {None, DEFAULT_MODEL_BASENAME}:
        model_version = f"{model_version}-{model_name}"
    temperature = float(metadata.get("softmax_temperature", 1.0) or 1.0)
    return LoadedModel(
        booster=booster,
        feature_columns=feature_columns,
        model_version=model_version,
        softmax_temperature=temperature,
        metadata=metadata,
        artifact=artifact_identity(model_path, meta_path),
    )


class MLPredictor:
    """LightGBM-based predictor with the same public API as BayesianPredictor.

    Use via::

        predictor = MLPredictor(session)
        preds = predictor.predict(race_id)
        preds = predictor.predict_and_save(race_id)

    The loaded model is memoised on the instance; pass ``model=`` to
    override (useful for unit tests).
    """

    def __init__(
        self,
        session: Session,
        model: LoadedModel | None = None,
    ) -> None:
        self.session = session
        self._model = model

    @property
    def model(self) -> LoadedModel:
        if self._model is None:
            self._model = load_latest_model()
        return self._model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, race_id: int) -> list[Prediction]:
        """Return a list of :class:`Prediction` sorted by probability desc."""
        race = self.session.get(Race, race_id)
        if race is None or not race.entries:
            return []

        frame = build_race_frame(self.session, race_id)
        if frame.empty:
            return []

        # Drop scratched horses so softmax + sort run only over runners.
        scratched_hids = {
            int(e.horse_id) for e in race.entries
            if getattr(e, "scratched", False)
        }
        if scratched_hids:
            frame = frame[~frame["horse_id"].isin(scratched_hids)].reset_index(drop=True)
            if frame.empty:
                return []

        X = validated_features(frame, self.model.feature_columns)
        raw_scores = np.asarray(self.model.booster.predict(X), dtype=float)
        if not np.isfinite(raw_scores).all():
            raise ValueError("Model produced non-finite scores")
        if self.model.metadata.get("objective") == "finish_time":
            raw_scores = -(raw_scores - raw_scores.mean()) / (float(raw_scores.std()) or 1.0)

        # Within-race softmax with the training-time fitted temperature.
        # LambdaRank margins have an arbitrary scale; ``T`` was chosen
        # on the holdout to minimise NLL so the normalised probabilities
        # are aggregate-calibrated (not just rank-correct).
        probs = _softmax(raw_scores, temperature=self.model.softmax_temperature)

        # Map back to entries so we can get horse.name.
        entries_by_id = {e.horse_id: e for e in race.entries}
        agf_by_hid = {
            e.horse_id: float(e.agf) if e.agf is not None else -1.0
            for e in race.entries
        }
        predictions: list[Prediction] = []
        for i, row in frame.iterrows():
            horse_id = int(row["horse_id"])
            entry = entries_by_id.get(horse_id)
            if entry is None:
                continue
            factors = {
                col: float(row[col])
                for col in self.model.feature_columns
                if col in row and row[col] is not None and not _isnan(row[col])
            }
            predictions.append(
                Prediction(
                    horse_id=horse_id,
                    horse_name=entry.horse.name if entry.horse else "?",
                    probability=float(probs[i] * 100.0),
                    confidence=_confidence(probs, i),
                    contributing_factors=factors,
                )
            )

        # AGF as tiebreaker: LambdaRank emits identical raw scores for
        # horses the trees can't differentiate (common in sparse
        # handikap cards) — after softmax those stay tied, and the
        # single "top pick" becomes arbitrary.  On 2026-04-24 this cost
        # 4 ticket hits the model had already ranked correctly.  Break
        # ties by market-implied strength; horse_id provides a final
        # deterministic fallback when AGF is also missing/equal.
        predictions.sort(
            key=lambda p: (
                p.probability,
                agf_by_hid.get(p.horse_id, -1.0),
                -p.horse_id,
            ),
            reverse=True,
        )
        return predictions

    def predict_and_save(self, race_id: int) -> list[Prediction]:
        """Run :meth:`predict` and persist to both RaceEntry and Prediction."""
        from ganyan.predictor.records import save_predictions
        preds = self.predict(race_id)
        return save_predictions(self.session, race_id, preds,
                                self.model.model_version, self.model.artifact)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically-stable softmax with optional temperature scaling.

    ``temperature > 1`` flattens the distribution (less confident);
    ``< 1`` sharpens it.  ``T = 1`` is the identity.
    """
    if len(x) == 0:
        return x
    t = max(float(temperature), 1e-6)
    scaled = np.asarray(x, dtype=float) / t
    shifted = scaled - scaled.max()
    exps = np.exp(shifted)
    return exps / exps.sum()


def _confidence(probs: np.ndarray, idx: int) -> float:
    """Heuristic confidence — how far above uniform is this pick?

    Confidence = min(1, (p - uniform) / (p_max - uniform)) clamped to
    [0, 1].  A horse at the softmax maximum scores 1.0; a horse right
    at uniform scores 0.0.
    """
    if len(probs) == 0:
        return 0.0
    uniform = 1.0 / len(probs)
    p_max = float(probs.max())
    if p_max <= uniform:
        return 0.0
    score = (float(probs[idx]) - uniform) / (p_max - uniform)
    return max(0.0, min(1.0, score))


def _isnan(x) -> bool:
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False


def validated_features(frame, columns):
    """Only the explicitly retired, always-empty apprentice feature is compatible."""
    missing = set(columns) - set(frame.columns)
    unexpected = missing - {"apprentice_jockey"}
    if unexpected:
        raise ValueError(f"Model requires unavailable features: {sorted(unexpected)}")
    return frame.reindex(columns=columns).astype("float64")
