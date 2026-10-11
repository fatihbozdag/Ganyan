"""Inference: load posterior + predict win probabilities for a race."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import arviz as az

from ganyan.predictor.bayes.data import (
    TrainingFrame, distance_bucket_for, summarize_last_six,
)


@dataclass
class BayesPrediction:
    horse_id: int
    horse_name: str
    mean_prob: float
    lo_5: float
    hi_95: float
    mean_score: float


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std > 1e-9:
        return (x - x.mean()) / std
    return np.zeros_like(x)


def predict_from_posterior(
    idata: az.InferenceData,
    frame: TrainingFrame,
    race: dict,
) -> List[BayesPrediction]:
    """Posterior over win probability for one race.

    `race` keys (required):
      horse_ids, jockeys, sires, track_id, distance_meters, agfs
    `race` keys (optional, used only if posterior has the matching coef):
      kgss, s20s, last_sixes (str list), horse_names
    """
    from ganyan.predictor.bayes.standardize import zscore_missing
    standardize = zscore_missing if frame.feature_schema >= 2 else lambda x: _zscore(np.nan_to_num(x))
    missing = float("nan") if frame.feature_schema >= 2 else 0.0
    post = idata.posterior
    theta = post["theta"].stack(sample=("chain", "draw")).values
    alpha = post["alpha_jockey"].stack(sample=("chain", "draw")).values
    beta = post["beta_sire"].stack(sample=("chain", "draw")).values
    gamma = post["gamma_track_dist"].stack(sample=("chain", "draw")).values
    delta = post["delta_agf"].stack(sample=("chain", "draw")).values
    sigma_theta = post["sigma_theta"].stack(sample=("chain", "draw")).values

    def _opt_coef(name: str) -> np.ndarray | None:
        if name in post.data_vars:
            return post[name].stack(sample=("chain", "draw")).values
        return None

    delta_kgs = _opt_coef("delta_kgs")
    delta_s20 = _opt_coef("delta_s20")
    delta_last6 = _opt_coef("delta_last6")
    delta_speed = _opt_coef("delta_speed")
    delta_workouts = _opt_coef("delta_workouts")
    delta_pace = _opt_coef("delta_pace")

    S = delta.shape[0]
    n = len(race["horse_ids"])
    rng = np.random.default_rng(0)

    horse_idx: List[int] = []
    cold_sires: List[int] = []
    for hid, sire in zip(race["horse_ids"], race["sires"]):
        if hid in frame.horse_index:
            horse_idx.append(frame.horse_index[hid])
            cold_sires.append(-1)
        else:
            sire_idx = frame.sire_index.get(sire or "", 0)
            horse_idx.append(-1)
            cold_sires.append(sire_idx)

    jockey_idx = [frame.jockey_index.get(j or "", -1) for j in race["jockeys"]]
    td_key = (race["track_id"], distance_bucket_for(race["distance_meters"]))
    td_idx = frame.track_dist_index.get(td_key, -1)

    score = np.zeros((n, S))
    for k in range(n):
        if horse_idx[k] >= 0:
            score[k] += theta[horse_idx[k], :]
        else:
            sire_mu = beta[cold_sires[k], :]
            score[k] += sire_mu + rng.normal(0, sigma_theta, size=S)
        if jockey_idx[k] >= 0:
            score[k] += alpha[jockey_idx[k], :]
        if td_idx >= 0:
            score[k] += gamma[td_idx, :]

    agfs = np.asarray(race["agfs"], dtype=float)
    score += np.outer(standardize(agfs), delta)

    if delta_kgs is not None and "kgss" in race:
        kgs_arr = np.asarray(
            [k if k is not None else missing for k in race["kgss"]], dtype=float,
        )
        score += np.outer(standardize(kgs_arr), delta_kgs)
    if delta_s20 is not None and "s20s" in race:
        s20_arr = np.asarray(
            [s if s is not None else missing for s in race["s20s"]], dtype=float,
        )
        score += np.outer(standardize(s20_arr), delta_s20)
    if delta_last6 is not None and "last_sixes" in race:
        last6_arr = np.asarray(
            [summarize_last_six(s) for s in race["last_sixes"]], dtype=float,
        )
        score += np.outer(standardize(last6_arr), delta_last6)
    if delta_speed is not None and "speeds" in race:
        speed_arr = np.asarray(
            [s if s is not None else missing for s in race["speeds"]], dtype=float,
        )
        score += np.outer(standardize(speed_arr), delta_speed)
    if delta_workouts is not None and "workouts" in race:
        workout_arr = np.asarray(
            [w if w is not None else missing for w in race["workouts"]], dtype=float,
        )
        score += np.outer(standardize(workout_arr), delta_workouts)
    if delta_pace is not None and "paces" in race:
        pace_arr = np.asarray(
            [p if p is not None else missing for p in race["paces"]], dtype=float,
        )
        score += np.outer(standardize(pace_arr), delta_pace)

    score -= score.max(axis=0, keepdims=True)
    exps = np.exp(score)
    probs = exps / exps.sum(axis=0, keepdims=True)

    mean_prob = probs.mean(axis=1)
    lo_5 = np.quantile(probs, 0.05, axis=1)
    hi_95 = np.quantile(probs, 0.95, axis=1)
    mean_score = score.mean(axis=1)

    names: Sequence[str] = race.get("horse_names", [str(hid) for hid in race["horse_ids"]])
    out = [
        BayesPrediction(
            horse_id=int(hid),
            horse_name=name,
            mean_prob=float(mp),
            lo_5=float(l5),
            hi_95=float(h95),
            mean_score=float(ms),
        )
        for hid, name, mp, l5, h95, ms in zip(
            race["horse_ids"], names, mean_prob, lo_5, hi_95, mean_score,
        )
    ]
    out.sort(key=lambda p: -p.mean_prob)
    return out
