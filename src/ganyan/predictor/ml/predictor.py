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

from ganyan.db.models import Prediction as PredictionRow, Race, RaceEntry
from ganyan.predictor.bayesian import Prediction
from ganyan.predictor.ml.features import FEATURE_COLUMNS, build_race_frame
from ganyan.predictor.ml.trainer import (
    DEFAULT_MODEL_BASENAME, DEFAULT_MODEL_DIR,
)


ML_MODEL_VERSION_PREFIX = "lightgbm-lambdarank"


@dataclass
class LoadedModel:
    """A deserialised booster plus its training-time feature list."""

    booster: lgb.Booster
    feature_columns: list[str]
    model_version: str
    metadata: dict = field(default_factory=dict)


def load_latest_model(
    model_dir: Path | None = None,
    model_name: str | None = None,
) -> LoadedModel:
    """Load the most recently saved booster + metadata.

    Raises :class:`FileNotFoundError` when no model has been trained yet.
    """
    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_name = model_name or DEFAULT_MODEL_BASENAME
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
    model_version = (
        f"{ML_MODEL_VERSION_PREFIX}-it{best_iter}"
        if best_iter is not None else ML_MODEL_VERSION_PREFIX
    )
    return LoadedModel(
        booster=booster,
        feature_columns=feature_columns,
        model_version=model_version,
        metadata=metadata,
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

        feature_cols = self.model.feature_columns
        X = frame[feature_cols].astype("float64")
        raw_scores = self.model.booster.predict(X)

        # Within-race softmax.  LightGBM's rank scores are
        # unnormalised log-preferences; exponentiating and normalising
        # is the standard way to turn them into a probability simplex.
        probs = _softmax(raw_scores)

        # Map back to entries so we can get horse.name.
        entries_by_id = {e.horse_id: e for e in race.entries}
        predictions: list[Prediction] = []
        for i, row in frame.iterrows():
            horse_id = int(row["horse_id"])
            entry = entries_by_id.get(horse_id)
            if entry is None:
                continue
            factors = {
                col: float(row[col])
                for col in feature_cols
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

        predictions.sort(key=lambda p: p.probability, reverse=True)
        return predictions

    def predict_and_save(self, race_id: int) -> list[Prediction]:
        """Run :meth:`predict` and persist to both RaceEntry and Prediction."""
        preds = self.predict(race_id)
        entries = {
            (e.race_id, e.horse_id): e
            for e in self.session.query(RaceEntry)
            .filter(RaceEntry.race_id == race_id)
            .all()
        }
        version = self.model.model_version
        for p in preds:
            entry = entries.get((race_id, p.horse_id))
            if entry is None:
                continue
            entry.predicted_probability = p.probability
            self.session.add(
                PredictionRow(
                    race_entry_id=entry.id,
                    model_version=version,
                    probability=p.probability,
                    confidence=p.confidence,
                    factors=p.contributing_factors,
                )
            )
        return preds


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax (subtracts max before exp)."""
    if len(x) == 0:
        return x
    shifted = x - np.max(x)
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
