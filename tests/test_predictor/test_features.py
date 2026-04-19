from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.db.models import (
    Base, Horse, Race, RaceEntry, RaceStatus, Track,
)
from ganyan.predictor.features import (
    compute_speed_figure,
    compute_form_cycle,
    compute_weight_delta,
    compute_rest_fitness,
    compute_class_indicator,
    compute_jockey_win_rate,
    compute_trainer_win_rate,
    compute_gate_bias,
    compute_surface_affinity,
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


# --- Gate bias ----------------------------------------------------------


def test_compute_gate_bias_inside_favored_on_short_sand():
    # Gate 1 of 14 on 1200 m Kum → inside bias positive.
    inside = compute_gate_bias(gate_number=1, distance_meters=1200, surface="Kum")
    outside = compute_gate_bias(gate_number=14, distance_meters=1200, surface="Kum")
    assert inside is not None and outside is not None
    assert inside > outside


def test_compute_gate_bias_none_without_gate():
    assert compute_gate_bias(gate_number=None, distance_meters=1400, surface="Çim") is None


# --- DB-backed features --------------------------------------------------


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _seed_resulted_race(session, track_name, race_date, horses, surface="Kum",
                        distance=1400, race_number=1):
    track = session.query(Track).filter_by(name=track_name).first() or Track(name=track_name)
    session.add(track)
    session.flush()
    race = Race(
        track_id=track.id, date=race_date, race_number=race_number,
        distance_meters=distance, surface=surface, status=RaceStatus.resulted,
    )
    session.add(race)
    session.flush()
    for (name, jockey, trainer, finish_pos) in horses:
        horse = session.query(Horse).filter_by(name=name).first() or Horse(
            name=name, trainer=trainer,
        )
        if horse.trainer is None and trainer is not None:
            horse.trainer = trainer
        session.add(horse)
        session.flush()
        session.add(RaceEntry(
            race_id=race.id, horse_id=horse.id,
            jockey=jockey, finish_position=finish_pos,
        ))
    session.flush()


def test_compute_jockey_win_rate_uses_history(db_session):
    # Jockey A wins 3 of 4 rides; jockey B wins 1 of 4.
    _seed_resulted_race(db_session, "İstanbul", date(2026, 1, 1),
                        [("H1", "A", "T1", 1), ("H2", "B", "T2", 2)])
    _seed_resulted_race(db_session, "İstanbul", date(2026, 1, 2),
                        [("H3", "A", "T1", 1), ("H4", "B", "T2", 2)], race_number=2)
    _seed_resulted_race(db_session, "İstanbul", date(2026, 1, 3),
                        [("H5", "A", "T1", 1), ("H6", "B", "T2", 2)], race_number=3)
    _seed_resulted_race(db_session, "İstanbul", date(2026, 1, 4),
                        [("H7", "A", "T1", 2), ("H8", "B", "T2", 1)], race_number=4)
    db_session.commit()

    rate_a = compute_jockey_win_rate(db_session, "A")
    rate_b = compute_jockey_win_rate(db_session, "B")
    assert rate_a is not None and rate_b is not None
    assert rate_a > rate_b  # A outperforms B


def test_compute_jockey_win_rate_before_date_excludes_future(db_session):
    _seed_resulted_race(db_session, "İstanbul", date(2026, 1, 1),
                        [("H1", "A", "T1", 1)])
    _seed_resulted_race(db_session, "İstanbul", date(2026, 2, 1),
                        [("H2", "A", "T1", 1)], race_number=2)
    db_session.commit()

    # Before 2026-01-15: only the first win counts.
    early = compute_jockey_win_rate(db_session, "A", before_date=date(2026, 1, 15))
    late = compute_jockey_win_rate(db_session, "A", before_date=date(2026, 3, 1))
    # Both non-None; rate should improve as more wins accumulate.
    assert early is not None and late is not None
    assert late > early or late == pytest.approx(early, abs=1e-6)


def test_compute_jockey_win_rate_returns_none_for_unknown(db_session):
    assert compute_jockey_win_rate(db_session, None) is None
    assert compute_jockey_win_rate(db_session, "Nobody") is None


def test_compute_trainer_win_rate(db_session):
    _seed_resulted_race(db_session, "İzmir", date(2026, 1, 1),
                        [("H1", "Jx", "Good", 1), ("H2", "Jy", "Bad", 2)])
    _seed_resulted_race(db_session, "İzmir", date(2026, 1, 2),
                        [("H3", "Jx", "Good", 1), ("H4", "Jy", "Bad", 2)], race_number=2)
    db_session.commit()

    good = compute_trainer_win_rate(db_session, "Good")
    bad = compute_trainer_win_rate(db_session, "Bad")
    assert good is not None and bad is not None
    assert good > bad


def test_compute_surface_affinity_matches_surface(db_session):
    # Horse H wins on Kum, loses on Çim.
    _seed_resulted_race(db_session, "Bursa", date(2026, 1, 1),
                        [("H", "J", "T", 1)], surface="Kum")
    _seed_resulted_race(db_session, "Bursa", date(2026, 1, 2),
                        [("H", "J", "T", 8)], surface="Çim", race_number=2)
    db_session.commit()

    horse = db_session.query(Horse).filter_by(name="H").one()
    kum_aff = compute_surface_affinity(
        db_session, horse.id, surface="Kum", distance_meters=1400,
    )
    cim_aff = compute_surface_affinity(
        db_session, horse.id, surface="Çim", distance_meters=1400,
    )
    assert kum_aff is not None and cim_aff is not None
    assert kum_aff > cim_aff
