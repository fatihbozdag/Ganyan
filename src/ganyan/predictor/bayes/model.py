"""PyMC Plackett-Luce models — simple and hierarchical (vectorized)."""
from __future__ import annotations

import pymc as pm
import pytensor.tensor as pt

from ganyan.predictor.bayes.data import TrainingFrame, matrices_for_pymc


def _vectorized_pl_loglik(score_matrix, valid_mask):
    """Plackett-Luce loglik on a padded (R, K_max) score matrix.

    For valid ordering positions k = 0..K-1 in race r:
        loglik_r = Σ_k (score[r,k] − logsumexp_{j≥k, j valid} score[r,j])

    Strategy:
      - For numerical stability, subtract row-wise max of *valid* scores.
      - Build exp_shifted with 0 at pad positions (so they don't contribute
        to the tail sum).
      - Reverse-cumsum gives tail sum exp; clip to a tiny positive floor
        before log to keep gradients finite at pad positions (those
        positions are then masked out of the final sum).
    """
    valid_f = pt.cast(valid_mask, "float64")
    # Row-wise max over valid entries only — safe to use a very negative
    # placeholder for pad because those positions are then zeroed below.
    masked_for_max = pt.where(valid_mask, score_matrix, -1e9)
    score_max = masked_for_max.max(axis=1, keepdims=True)
    shifted = score_matrix - score_max
    # Zero out pad positions in exp space so tail sum ignores them.
    exp_shifted = pt.exp(shifted) * valid_f
    rev = exp_shifted[:, ::-1]
    cum_rev = pt.cumsum(rev, axis=1)
    cum = cum_rev[:, ::-1]
    # Floor the cumsum so log is finite even at trailing pad positions
    # (those positions are masked out of the final sum anyway).
    cum_safe = pt.maximum(cum, 1e-30)
    log_tail_sum = pt.log(cum_safe) + score_max
    pos_loglik = score_matrix - log_tail_sum
    return (pos_loglik * valid_f).sum()


def build_simple_pl_model(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    mats = matrices_for_pymc(frame)
    horse_idx_mat = mats["horse_idx_mat"]
    valid_mask = mats["valid_mask"]
    coords = {"horse": list(range(n_horses))}
    with pm.Model(coords=coords) as model:
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, dims="horse")
        score = theta[horse_idx_mat]
        pm.Potential("plackett_luce", _vectorized_pl_loglik(score, valid_mask))
    return model


def build_hierarchical_pl_model(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    n_jockeys = len(frame.jockey_index)
    n_sires = len(frame.sire_index)
    n_track_dist = len(frame.track_dist_index)
    mats = matrices_for_pymc(frame)

    coords = {
        "horse": list(range(n_horses)),
        "jockey": list(range(n_jockeys)),
        "sire": list(range(n_sires)),
        "track_dist": list(range(n_track_dist)),
    }
    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfNormal("sigma_theta", 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", 0.5)

        beta_sire = pm.Normal("beta_sire", 0.0, sigma_beta, dims="sire")
        mu_horse = beta_sire[mats["horse_to_sire"]]
        theta = pm.Normal("theta", mu=mu_horse, sigma=sigma_theta, dims="horse")
        alpha_jockey = pm.Normal("alpha_jockey", 0.0, sigma_alpha, dims="jockey")
        # A shared race intercept cancels from PL; keep a zero deterministic
        # for posterior-file compatibility, not an unidentified parameter.
        gamma_track_dist = pm.Deterministic(
            "gamma_track_dist", pt.zeros(n_track_dist), dims="track_dist",
        )

        score = (
            theta[mats["horse_idx_mat"]]
            + alpha_jockey[mats["jockey_idx_mat"]]
            + gamma_track_dist[mats["track_dist_arr"]][:, None]
        )
        pm.Potential(
            "plackett_luce_hier",
            _vectorized_pl_loglik(score, mats["valid_mask"]),
        )
    return model


def build_hierarchical_pl_model_with_agf(frame: TrainingFrame) -> pm.Model:
    n_horses = len(frame.horse_index)
    n_jockeys = len(frame.jockey_index)
    n_sires = len(frame.sire_index)
    n_track_dist = len(frame.track_dist_index)
    mats = matrices_for_pymc(frame)

    coords = {
        "horse": list(range(n_horses)),
        "jockey": list(range(n_jockeys)),
        "sire": list(range(n_sires)),
        "track_dist": list(range(n_track_dist)),
    }
    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfNormal("sigma_theta", 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", 0.5)

        beta_sire = pm.Normal("beta_sire", 0.0, sigma_beta, dims="sire")
        mu_horse = beta_sire[mats["horse_to_sire"]]
        theta = pm.Normal("theta", mu=mu_horse, sigma=sigma_theta, dims="horse")
        alpha_jockey = pm.Normal("alpha_jockey", 0.0, sigma_alpha, dims="jockey")
        # A shared race intercept cancels from PL; keep a zero deterministic
        # for posterior-file compatibility, not an unidentified parameter.
        gamma_track_dist = pm.Deterministic(
            "gamma_track_dist", pt.zeros(n_track_dist), dims="track_dist",
        )
        delta_agf = pm.Normal("delta_agf", 0.0, 1.0)

        score = (
            theta[mats["horse_idx_mat"]]
            + alpha_jockey[mats["jockey_idx_mat"]]
            + gamma_track_dist[mats["track_dist_arr"]][:, None]
            + delta_agf * mats["agf_z_mat"]
        )
        pm.Potential(
            "plackett_luce_agf",
            _vectorized_pl_loglik(score, mats["valid_mask"]),
        )
    return model


def build_full_hierarchical_pl_model(frame: TrainingFrame) -> pm.Model:
    """Hierarchical PL with AGF prior + private signals.

    Adds Normal(0, 1) coefficients on within-race z-scores of:
      - kgs (days since last race)
      - s20 (last-20 perf score)
      - last6 (mean recent finish position; lower=better)
      - speed (track-variant-adjusted sec/m, recency-mean; lower=faster,
        so expect a negative coefficient).

    Cold-start safe: if a horse has no value, it enters as 0 (within-race
    mean), so the term contributes 0 to that horse's PL score.
    """
    n_horses = len(frame.horse_index)
    n_jockeys = len(frame.jockey_index)
    n_sires = len(frame.sire_index)
    n_track_dist = len(frame.track_dist_index)
    mats = matrices_for_pymc(frame)

    coords = {
        "horse": list(range(n_horses)),
        "jockey": list(range(n_jockeys)),
        "sire": list(range(n_sires)),
        "track_dist": list(range(n_track_dist)),
    }
    with pm.Model(coords=coords) as model:
        sigma_theta = pm.HalfNormal("sigma_theta", 1.0)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 0.5)
        sigma_beta = pm.HalfNormal("sigma_beta", 0.5)

        beta_sire = pm.Normal("beta_sire", 0.0, sigma_beta, dims="sire")
        mu_horse = beta_sire[mats["horse_to_sire"]]
        theta = pm.Normal("theta", mu=mu_horse, sigma=sigma_theta, dims="horse")
        alpha_jockey = pm.Normal("alpha_jockey", 0.0, sigma_alpha, dims="jockey")
        # A shared race intercept cancels from PL; keep a zero deterministic
        # for posterior-file compatibility, not an unidentified parameter.
        gamma_track_dist = pm.Deterministic(
            "gamma_track_dist", pt.zeros(n_track_dist), dims="track_dist",
        )
        delta_agf = pm.Normal("delta_agf", 0.0, 1.0)
        delta_kgs = pm.Normal("delta_kgs", 0.0, 1.0)
        delta_s20 = pm.Normal("delta_s20", 0.0, 1.0)
        delta_last6 = pm.Normal("delta_last6", 0.0, 1.0)
        delta_speed = pm.Normal("delta_speed", 0.0, 1.0)
        delta_workouts = pm.Normal("delta_workouts", 0.0, 1.0)
        delta_pace = pm.Normal("delta_pace", 0.0, 1.0)

        score = (
            theta[mats["horse_idx_mat"]]
            + alpha_jockey[mats["jockey_idx_mat"]]
            + gamma_track_dist[mats["track_dist_arr"]][:, None]
            + delta_agf * mats["agf_z_mat"]
            + delta_kgs * mats["kgs_z_mat"]
            + delta_s20 * mats["s20_z_mat"]
            + delta_last6 * mats["last6_z_mat"]
            + delta_speed * mats["speed_z_mat"]
            + delta_workouts * mats["workout_z_mat"]
            + delta_pace * mats["pace_z_mat"]
        )
        pm.Potential(
            "plackett_luce_full",
            _vectorized_pl_loglik(score, mats["valid_mask"]),
        )
    return model


def fit_advi(model: pm.Model, n_iter: int = 30_000, seed: int = 0):
    with model:
        approx = pm.fit(n_iter, method="advi", random_seed=seed, progressbar=False)
        idata = approx.sample(draws=2_000, random_seed=seed)
    return idata
