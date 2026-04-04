import pytest
from ganyan.predictor.features import (
    compute_speed_figure,
    compute_form_cycle,
    compute_weight_delta,
    compute_rest_fitness,
    compute_class_indicator,
    extract_features,
    HorseFeatures,
)


def test_compute_speed_figure():
    speed = compute_speed_figure(eid_seconds=90.45, distance_meters=1400)
    assert 15.0 < speed < 16.0


def test_compute_speed_figure_none():
    assert compute_speed_figure(eid_seconds=None, distance_meters=1400) is None
    assert compute_speed_figure(eid_seconds=90.0, distance_meters=None) is None


def test_compute_form_cycle_improving():
    positions = [6, 5, 4, 3, 2, 1]
    score = compute_form_cycle(positions)
    assert score > 0.7


def test_compute_form_cycle_declining():
    positions = [1, 2, 3, 4, 5, 6]
    score = compute_form_cycle(positions)
    assert score < 0.4


def test_compute_form_cycle_empty():
    assert compute_form_cycle([]) is None
    assert compute_form_cycle(None) is None


def test_compute_form_cycle_with_nones():
    positions = [2, None, 3, None, 1, 4]
    score = compute_form_cycle(positions)
    assert score is not None


def test_compute_weight_delta():
    delta = compute_weight_delta(horse_weight=55.0, field_avg_weight=58.0)
    assert delta > 0


def test_compute_weight_delta_heavy():
    delta = compute_weight_delta(horse_weight=62.0, field_avg_weight=58.0)
    assert delta < 0


def test_compute_rest_fitness_optimal():
    fitness = compute_rest_fitness(kgs=21)
    assert fitness > 0.8


def test_compute_rest_fitness_too_long():
    fitness = compute_rest_fitness(kgs=120)
    assert fitness < 0.5


def test_compute_rest_fitness_too_short():
    fitness = compute_rest_fitness(kgs=3)
    assert fitness < 0.6


def test_compute_rest_fitness_none():
    assert compute_rest_fitness(None) is None


def test_compute_class_indicator():
    indicator = compute_class_indicator(hp=85.0, field_avg_hp=80.0)
    assert indicator > 0


def test_extract_features():
    features = extract_features(
        eid_seconds=90.45,
        distance_meters=1400,
        last_six_parsed=[2, 4, 1, 3, 2, 1],
        weight_kg=57.0,
        field_avg_weight=58.0,
        kgs=21,
        hp=85.0,
        field_avg_hp=80.0,
    )
    assert isinstance(features, HorseFeatures)
    assert features.speed_figure is not None
    assert features.form_cycle is not None
    assert features.weight_delta is not None
    assert features.rest_fitness is not None
    assert features.class_indicator is not None
