"""Tests for the Bayesian trainer's persistence layer."""
from __future__ import annotations

from pathlib import Path

import pymc as pm

from ganyan.predictor.bayes.data import TrainingFrame
from ganyan.predictor.bayes.trainer import save_posterior, load_posterior


def test_save_and_load_roundtrip(tmp_path: Path):
    with pm.Model():
        pm.Normal("x", 0, 1)
        idata = pm.sample(
            draws=50, tune=50, chains=1, random_seed=0,
            progressbar=False,
        )
    frame = TrainingFrame()
    frame.horse_index = {1: 0, 2: 1}
    frame.jockey_index = {"jA": 0}
    frame.sire_index = {"": 0}
    frame.track_dist_index = {(3, 0): 0}

    base = tmp_path / "bayes_pl"
    save_posterior(idata, frame, base)

    assert base.with_suffix(".nc").exists()
    assert base.with_suffix(".indices.json").exists()

    loaded_idata, loaded_frame = load_posterior(base)
    assert "x" in loaded_idata.posterior
    assert loaded_frame.horse_index == {1: 0, 2: 1}
    assert loaded_frame.jockey_index == {"jA": 0}
    assert (3, 0) in loaded_frame.track_dist_index
