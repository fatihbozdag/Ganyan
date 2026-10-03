"""Multi-model ensemble predictor.

Loads explicitly approved, hash-verified model heads and runs them against the
same race.  Convergence — multiple independent models agreeing on a
horse — is treated as a stronger signal than any single model's pick.

Heterogeneous heads supported:
- LightGBM rank-objective models (default ``.txt`` boosters)
- LightGBM finish-time regressors (sort ascending = rank)
- Linear rankers (numpy ``.npz`` + standardisation, conditional logit
  or Plackett-Luce)
- Per-race-type specialists — auto-skipped on races whose race_type
  doesn't match the specialist's training prefix.

Each model carries its own ``feature_columns`` list and a per-objective
prediction routine; ``build_race_frame`` produces a superset feature
matrix that every head slices what it needs from.

Output per horse: probability from each applicable model, convergence
score (how many models rank this horse at #1), mean probability across
heads, disagreement (std), sorted by ``(convergence, mean_prob, agf)``
descending.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from ganyan.db.models import Prediction as PredictionRow, Race, RaceEntry
from ganyan.predictor.bayesian import Prediction
from ganyan.predictor.ml.features import build_race_frame
from ganyan.predictor.ml.predictor import (
    LoadedModel, _softmax, load_latest_model, validated_features,
)
from ganyan.predictor.ml.trainer import DEFAULT_MODEL_DIR


logger = logging.getLogger(__name__)


@dataclass
class LoadedLinearModel:
    """Adapter for numpy linear-ranker (conditional logit / Plackett-Luce).

    Mimics the relevant surface of ``LoadedModel`` so EnsemblePredictor
    can iterate over heterogeneous heads.  Inference: standardise the
    feature row (mean/std from training), dot with β, return raw scores
    that get softmax'd within-race like any other rank head.
    """

    name: str
    beta: np.ndarray
    feature_columns: list[str]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    model_family: str  # "conditional_logit" or "plackett_luce"
    metadata: dict = field(default_factory=dict)
    softmax_temperature: float = 1.0  # MLE-fitted softmax already; T=1
    artifact: dict = field(default_factory=dict)

    @property
    def model_version(self) -> str:
        return f"linear-{self.model_family}-{self.name}"

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Apply the same standardisation as training, return β·x scores."""
        filled = np.where(np.isnan(X), self.feat_mean, X)
        std_X = (filled - self.feat_mean) / self.feat_std
        std_X = np.where(np.isfinite(std_X), std_X, 0.0)
        return std_X @ self.beta


def _load_linear_model(
    name: str, meta: dict, model_dir: Path,
) -> LoadedLinearModel | None:
    npz_path = model_dir / meta.get("npz_path", f"{name}.npz")
    if not npz_path.exists():
        return None
    npz = np.load(npz_path, allow_pickle=False)
    feat_cols = list(meta.get("feature_columns") or npz["feature_columns"].tolist())
    return LoadedLinearModel(
        name=name,
        beta=np.asarray(npz["beta"], dtype=float),
        feature_columns=feat_cols,
        feat_mean=np.asarray(npz["mean"], dtype=float),
        feat_std=np.asarray(npz["std"], dtype=float),
        model_family=meta.get("model_family", "linear"),
        metadata=meta,
    )


@dataclass
class EnsemblePrediction:
    """Per-horse summary across all loaded models."""

    horse_id: int
    horse_name: str
    # Mean probability across models (weighted equally for now).
    mean_probability: float
    # Number of models that ranked this horse strictly at position 1.
    convergence_top1: int
    # Number of models that placed this horse in their top-3.
    convergence_top3: int
    # Standard deviation of per-model probabilities — higher = models
    # disagree on this horse, lower = models agree on its share.
    disagreement: float
    # Average rank (1-based) across models.
    mean_rank: float
    # Per-model details: model_name -> {probability, rank}
    by_model: dict[str, dict] = field(default_factory=dict)


def _list_model_names(model_dir):
    from ganyan.predictor.ml.artifacts import approved_manifest
    return sorted(approved_manifest(model_dir)["heads"])


def load_all_models(model_dir=None):
    from ganyan.predictor.ml.artifacts import approved_paths, model_root
    model_dir = Path(model_dir or model_root())
    out = []
    for name in _list_model_names(model_dir):
        paths = approved_paths(name, model_dir)
        metadata = json.loads(paths["metadata"].read_text())
        if metadata.get("objective") == "ev":
            continue
        if "model_family" in metadata:
            metadata["npz_path"] = str(paths["model"])
            model = _load_linear_model(name, metadata, model_dir)
        else:
            # Read the manifest-resolved immutable pair even after promotion.
            model = load_latest_model(model_dir=paths["model"].parent,
                                      model_name=paths["model"].stem)
        if model is None:
            raise FileNotFoundError(f"Approved head missing: {name}")
        # Keep head names unique even for content-addressed releases named model.txt.
        from ganyan.predictor.ml.artifacts import artifact_identity
        model.artifact = artifact_identity(paths["model"], paths["metadata"])
        model.metadata["approved_head"] = name
        if isinstance(model, LoadedModel):
            model.model_version = name
        out.append(model)
    return out


def _model_applies_to_race(
    model: LoadedModel | LoadedLinearModel, race_type: str | None,
) -> bool:
    """Race-type specialist gate: only run a head whose training data
    matches the race's race_type prefix.

    Generic heads (``race_type_prefix`` missing or null) apply to every
    race.  Specialists with a prefix only apply when ``race_type``
    starts with it.
    """
    prefix = (model.metadata or {}).get("race_type_prefix")
    if not prefix:
        return True
    if not race_type:
        return False
    return race_type.startswith(prefix)


class EnsemblePredictor:
    """Run every available rank-objective model on the same race and
    aggregate.

    Usage::

        predictor = EnsemblePredictor(session)
        preds = predictor.predict(race_id)
        for p in preds[:5]:
            print(p.horse_name, p.convergence_top1, p.mean_probability)
    """

    def __init__(
        self,
        session: Session,
        models: list[LoadedModel] | None = None,
    ) -> None:
        self.session = session
        self._models = models

    @property
    def models(self) -> list[LoadedModel]:
        if self._models is None:
            self._models = load_all_models()
            if not self._models:
                raise FileNotFoundError(
                    "No trained models found.  Run `ganyan train` first.",
                )
        return self._models

    def predict(self, race_id: int) -> list[EnsemblePrediction]:
        race = self.session.get(Race, race_id)
        if race is None or not race.entries:
            return []
        frame = build_race_frame(self.session, race_id)
        if frame.empty:
            return []

        # Drop scratched horses pre-prediction so the softmax + sort
        # operate over only the actual runners. The probability mass
        # of scratched horses redistributes naturally.
        scratched_hids = {
            int(e.horse_id) for e in race.entries
            if getattr(e, "scratched", False)
        }
        if scratched_hids:
            frame = frame[~frame["horse_id"].isin(scratched_hids)].reset_index(drop=True)
            if frame.empty:
                return []

        entries_by_id = {e.horse_id: e for e in race.entries if e.horse_id not in scratched_hids}
        agf_by_hid = {
            e.horse_id: float(e.agf) if e.agf is not None else -1.0
            for e in race.entries if e.horse_id not in scratched_hids
        }
        horse_ids = [int(h) for h in frame["horse_id"]]

        # Per-model probabilities {model_name -> {horse_id -> prob}}
        per_model: dict[str, dict[int, float]] = {}
        per_model_rank: dict[str, dict[int, int]] = {}

        for model in self.models:
            # Specialist gating: skip models whose training prefix
            # doesn't match this race's race_type.
            if not _model_applies_to_race(model, race.race_type):
                continue
            X_df = validated_features(frame, model.feature_columns)
            if isinstance(model, LoadedLinearModel):
                raw = model.predict_raw(X_df.to_numpy())
            else:
                raw = np.asarray(model.booster.predict(X_df), dtype=float)
            if not np.isfinite(raw).all():
                raise ValueError(f"Non-finite scores from {model.model_version}")
            obj = (model.metadata or {}).get("objective", "rank")
            if obj == "finish_time":
                # Predicted finish times in seconds.  Smaller = better.
                # Convert to within-race probabilities by negating and
                # softmax'ing the z-score; this puts the head on the
                # same probability simplex as the rank-objective heads
                # without distorting the sort order.
                std = float(raw.std()) or 1.0
                z = -(raw - raw.mean()) / std
                probs = _softmax(z, temperature=model.softmax_temperature)
            else:
                probs = _softmax(raw, temperature=model.softmax_temperature)
            # AGF-tiebreak when probs equal.
            order_key = [
                (float(probs[i]), agf_by_hid.get(horse_ids[i], -1.0), -horse_ids[i])
                for i in range(len(horse_ids))
            ]
            ranking = sorted(
                range(len(horse_ids)), key=lambda i: order_key[i], reverse=True,
            )
            ranks = {horse_ids[ranking[r]]: r + 1 for r in range(len(ranking))}
            per_model[model.model_version] = {
                horse_ids[i]: float(probs[i]) for i in range(len(horse_ids))
            }
            per_model_rank[model.model_version] = ranks

        if not per_model:
            raise ValueError("No approved model applies to this race")

        # Aggregate per horse.
        out: list[EnsemblePrediction] = []
        for hid in horse_ids:
            entry = entries_by_id.get(hid)
            if entry is None:
                continue
            probs_list = [per_model[m].get(hid, 0.0) for m in per_model]
            ranks_list = [
                per_model_rank[m].get(hid, len(horse_ids)) for m in per_model_rank
            ]
            mean_p = float(np.mean(probs_list)) if probs_list else 0.0
            std_p = float(np.std(probs_list)) if len(probs_list) > 1 else 0.0
            top1 = sum(1 for r in ranks_list if r == 1)
            top3 = sum(1 for r in ranks_list if r <= 3)
            mean_rank = float(np.mean(ranks_list)) if ranks_list else 0.0

            by_model = {
                m: {
                    "probability": per_model[m].get(hid, 0.0) * 100.0,
                    "rank": per_model_rank[m].get(hid, None),
                }
                for m in per_model
            }
            out.append(
                EnsemblePrediction(
                    horse_id=hid,
                    horse_name=entry.horse.name if entry.horse else "?",
                    mean_probability=mean_p * 100.0,
                    convergence_top1=top1,
                    convergence_top3=top3,
                    disagreement=std_p * 100.0,
                    mean_rank=mean_rank,
                    by_model=by_model,
                )
            )

        # Sort: more models agreeing at #1 wins; tiebreak by mean prob,
        # then by AGF (market) as final fallback to break perfect ties
        # — same convention as the single-model tiebreaker.
        out.sort(
            key=lambda p: (
                p.convergence_top1,
                p.mean_probability,
                agf_by_hid.get(p.horse_id, -1.0),
            ),
            reverse=True,
        )
        return out

    def predict_as_predictions(self, race_id: int) -> list[Prediction]:
        """Adapter to the ``Prediction`` shape used by CLI/web layers.

        The ensemble's ``mean_probability`` becomes ``Prediction.probability``;
        ``convergence_top1`` is exposed via ``confidence`` (normalised by
        the number of loaded models so values stay in 0..1) and the
        per-model breakdown is shoved into ``contributing_factors``.
        """
        rows = self.predict(race_id)
        n_models = max(1, len(rows[0].by_model)) if rows else 1
        out: list[Prediction] = []
        for r in rows:
            factors = {
                f"{name}_prob": float(d["probability"])
                for name, d in r.by_model.items()
            }
            factors["convergence_top1"] = float(r.convergence_top1)
            factors["convergence_top3"] = float(r.convergence_top3)
            factors["disagreement"] = float(r.disagreement)
            factors["mean_rank"] = float(r.mean_rank)
            out.append(
                Prediction(
                    horse_id=r.horse_id,
                    horse_name=r.horse_name,
                    probability=r.mean_probability,
                    confidence=r.convergence_top1 / n_models,
                    contributing_factors=factors,
                )
            )
        return out

    def predict_and_save(self, race_id: int) -> list[Prediction]:
        """Run ``predict_as_predictions`` and persist to RaceEntry +
        Prediction rows — same contract as ``MLPredictor.predict_and_save``
        so callers (notably the scheduler's morning_card job) can swap
        predictors without further changes.
        """
        from ganyan.predictor.records import save_predictions
        preds = self.predict_as_predictions(race_id)
        identity = {"heads": {m.model_version: m.artifact for m in self.models}}
        return save_predictions(self.session, race_id, preds,
                                f"ensemble-{len(self.models)}-heads", identity)
