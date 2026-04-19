"""Bayesian prediction engine for horse racing."""

import math
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ganyan.db.models import Prediction as PredictionRow, Race, RaceEntry
from ganyan.predictor.features import extract_features, HorseFeatures
from ganyan.scraper.parser import parse_eid_to_seconds, parse_last_six


# Bump this when the feature set, weights, or formula change so the
# predictions audit table can distinguish results across model variants.
MODEL_VERSION = "bayesian-v2"


# Feature weights for likelihood computation.
FEATURE_WEIGHTS: dict[str, float] = {
    "speed": 0.22,
    "form": 0.20,
    "weight": 0.10,
    "rest": 0.10,
    "class": 0.13,
    "jockey": 0.12,
    "trainer": 0.05,
    "gate": 0.03,
    "surface_affinity": 0.05,
}


@dataclass
class Prediction:
    horse_id: int
    horse_name: str
    probability: float  # 0-100
    confidence: float  # 0-1
    contributing_factors: dict = field(default_factory=dict)  # feature_name -> impact


class BayesianPredictor:
    """Naive-Bayesian predictor combining multiple horse-racing features."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def predict_and_save(self, race_id: int) -> list[Prediction]:
        """Predict and persist to both the ``race_entries`` slot (for quick
        lookup) and the ``predictions`` audit table (keeps every run).
        """
        predictions = self.predict(race_id)
        entries = {
            (e.race_id, e.horse_id): e
            for e in self.session.query(RaceEntry)
            .filter(RaceEntry.race_id == race_id)
            .all()
        }
        for p in predictions:
            entry = entries.get((race_id, p.horse_id))
            if entry is None:
                continue
            entry.predicted_probability = p.probability
            # Append audit row (never overwrites prior predictions).
            self.session.add(
                PredictionRow(
                    race_entry_id=entry.id,
                    model_version=MODEL_VERSION,
                    probability=p.probability,
                    confidence=p.confidence,
                    factors=p.contributing_factors,
                )
            )
        return predictions

    def predict(self, race_id: int) -> list[Prediction]:
        """Predict win probabilities for all entries in a race.

        Returns a list of Prediction objects sorted by probability descending.
        """
        race = self.session.get(Race, race_id)
        if race is None:
            return []

        entries: list[RaceEntry] = race.entries
        if not entries:
            return []

        # Compute field averages for relative features.
        weights = [float(e.weight_kg) for e in entries if e.weight_kg is not None]
        hps = [float(e.hp) for e in entries if e.hp is not None]
        field_avg_weight = sum(weights) / len(weights) if weights else None
        field_avg_hp = sum(hps) / len(hps) if hps else None

        distance = race.distance_meters

        # Extract features for each entry.  All history-based lookups use
        # ``before_date=race.date`` so training/evaluation stays leak-free.
        entry_features: list[tuple[RaceEntry, HorseFeatures]] = []
        for entry in entries:
            eid_seconds = parse_eid_to_seconds(entry.eid)
            last_six_parsed = parse_last_six(entry.last_six)
            trainer_name = entry.horse.trainer if entry.horse else None
            features = extract_features(
                eid_seconds=eid_seconds,
                distance_meters=distance,
                last_six_parsed=last_six_parsed,
                weight_kg=float(entry.weight_kg) if entry.weight_kg is not None else None,
                field_avg_weight=field_avg_weight,
                kgs=int(entry.kgs) if entry.kgs is not None else None,
                hp=float(entry.hp) if entry.hp is not None else None,
                field_avg_hp=field_avg_hp,
                session=self.session,
                jockey=entry.jockey,
                trainer=trainer_name,
                horse_id=entry.horse_id,
                gate_number=entry.gate_number,
                surface=race.surface,
                race_date=race.date,
            )
            entry_features.append((entry, features))

        # Compute likelihoods and contributing factors.
        n = len(entry_features)
        prior = 1.0 / n

        likelihoods: list[float] = []
        all_factors: list[dict[str, float]] = []

        for _entry, features in entry_features:
            likelihood, factors = self._compute_likelihood(features)
            likelihoods.append(likelihood)
            all_factors.append(factors)

        # Posterior = prior * likelihood (unnormalized).
        posteriors = [prior * lk for lk in likelihoods]

        # Normalize to sum to 100%.
        total_posterior = sum(posteriors)
        if total_posterior <= 0:
            # Fallback: uniform distribution.
            probabilities = [100.0 / n] * n
        else:
            probabilities = [(p / total_posterior) * 100.0 for p in posteriors]

        # Compute confidence per prediction.
        confidences = self._compute_confidences(entry_features, probabilities, n)

        # Build predictions.
        predictions: list[Prediction] = []
        for i, (entry, _features) in enumerate(entry_features):
            predictions.append(
                Prediction(
                    horse_id=entry.horse_id,
                    horse_name=entry.horse.name,
                    probability=probabilities[i],
                    confidence=confidences[i],
                    contributing_factors=all_factors[i],
                )
            )

        # Sort by probability descending.
        predictions.sort(key=lambda p: p.probability, reverse=True)
        return predictions

    @staticmethod
    def _compute_likelihood(features: HorseFeatures) -> tuple[float, dict[str, float]]:
        """Compute weighted likelihood from features.

        Returns (likelihood, contributing_factors) where likelihood is a
        positive float and contributing_factors maps feature names to their
        signed impact values.
        """
        factors: dict[str, float] = {}
        weighted_sum = 0.0

        # Speed figure: higher is better. Normalize to 0-1 range using a
        # reference speed of ~15 m/s (typical for 1200-2000m races).
        if features.speed_figure is not None:
            impact = (features.speed_figure - 14.0) / 4.0  # center around 14 m/s
            factors["speed"] = impact
            weighted_sum += FEATURE_WEIGHTS["speed"] * impact
        else:
            factors["speed"] = 0.0

        # Form cycle: already 0-1 where 1 is best.
        if features.form_cycle is not None:
            impact = features.form_cycle - 0.5  # center around 0.5
            factors["form"] = impact
            weighted_sum += FEATURE_WEIGHTS["form"] * impact
        else:
            factors["form"] = 0.0

        # Weight delta: positive = lighter than field average (advantage).
        if features.weight_delta is not None:
            impact = features.weight_delta * 5.0  # amplify small differences
            factors["weight"] = impact
            weighted_sum += FEATURE_WEIGHTS["weight"] * impact
        else:
            factors["weight"] = 0.0

        # Rest fitness: already 0-1 where 1 is optimal rest.
        if features.rest_fitness is not None:
            impact = features.rest_fitness - 0.5
            factors["rest"] = impact
            weighted_sum += FEATURE_WEIGHTS["rest"] * impact
        else:
            factors["rest"] = 0.0

        # Class indicator: positive = above average HP.
        if features.class_indicator is not None:
            impact = features.class_indicator * 3.0  # amplify
            factors["class"] = impact
            weighted_sum += FEATURE_WEIGHTS["class"] * impact
        else:
            factors["class"] = 0.0

        # Jockey win rate: deviation from 10% baseline.
        if features.jockey_win_rate is not None:
            impact = (features.jockey_win_rate - 0.10) * 5.0
            factors["jockey"] = impact
            weighted_sum += FEATURE_WEIGHTS["jockey"] * impact
        else:
            factors["jockey"] = 0.0

        # Trainer win rate: same structure as jockey.
        if features.trainer_win_rate is not None:
            impact = (features.trainer_win_rate - 0.10) * 5.0
            factors["trainer"] = impact
            weighted_sum += FEATURE_WEIGHTS["trainer"] * impact
        else:
            factors["trainer"] = 0.0

        # Gate bias: already centered and scaled.
        if features.gate_bias is not None:
            factors["gate"] = features.gate_bias
            weighted_sum += FEATURE_WEIGHTS["gate"] * features.gate_bias
        else:
            factors["gate"] = 0.0

        # Surface affinity: deviation from 10% baseline, like win rates.
        if features.surface_affinity is not None:
            impact = (features.surface_affinity - 0.10) * 5.0
            factors["surface_affinity"] = impact
            weighted_sum += FEATURE_WEIGHTS["surface_affinity"] * impact
        else:
            factors["surface_affinity"] = 0.0

        # Convert to positive likelihood using softmax-style exp.
        likelihood = math.exp(weighted_sum)

        return likelihood, factors

    @staticmethod
    def _compute_confidences(
        entry_features: list[tuple[RaceEntry, HorseFeatures]],
        probabilities: list[float],
        n: int,
    ) -> list[float]:
        """Compute confidence score (0-1) for each prediction.

        Confidence is based on:
        - Data completeness (how many features are available)
        - Separation from uniform distribution
        """
        confidences: list[float] = []
        uniform_prob = 100.0 / n if n > 0 else 0.0

        for i, (_entry, features) in enumerate(entry_features):
            # Data completeness: fraction of non-None features.
            feature_values = [
                features.speed_figure,
                features.form_cycle,
                features.weight_delta,
                features.rest_fitness,
                features.class_indicator,
                features.jockey_win_rate,
                features.trainer_win_rate,
                features.gate_bias,
                features.surface_affinity,
            ]
            available = sum(1 for v in feature_values if v is not None)
            completeness = available / len(feature_values)

            # Separation: how far this prediction is from uniform.
            if uniform_prob > 0:
                separation = min(abs(probabilities[i] - uniform_prob) / uniform_prob, 1.0)
            else:
                separation = 0.0

            # Confidence = weighted combination of completeness and separation.
            confidence = 0.7 * completeness + 0.3 * separation
            confidence = max(0.0, min(1.0, confidence))
            confidences.append(confidence)

        return confidences
