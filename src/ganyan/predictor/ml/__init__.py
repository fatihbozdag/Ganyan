"""Machine-learning predictor (LightGBM LambdaRank).

An alternative to :class:`ganyan.predictor.bayesian.BayesianPredictor`
that *learns* feature weights and interactions from historical race
data rather than using hand-picked constants.

Exports:
- :class:`MLPredictor` — inference-time, same public API as BayesianPredictor
- :func:`train_ranker` — fit a new model from the current DB
- :func:`build_training_frame` — produce the feature matrix
"""

from ganyan.predictor.ml.features import FEATURE_COLUMNS, build_training_frame
from ganyan.predictor.ml.predictor import MLPredictor, load_latest_model
from ganyan.predictor.ml.trainer import TrainingResult, train_ranker

__all__ = [
    "FEATURE_COLUMNS",
    "MLPredictor",
    "TrainingResult",
    "build_training_frame",
    "load_latest_model",
    "train_ranker",
]
