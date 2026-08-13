from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import numpy as np
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from ganyan.db.models import Race, RaceEntry, RaceStatus


# Bayesian-smoothing prior for win-rate style features — keeps jockeys
# with a handful of races from dominating purely by low sample size.
_WINRATE_PRIOR_MEAN = 0.10  # typical baseline ≈ 10% across all jockeys
_WINRATE_PRIOR_WEIGHT = 20  # equivalent to "20 pseudo-races at 10%"


@dataclass
class HorseFeatures:
    speed_figure: float | None = None
    form_cycle: float | None = None
    weight_delta: float | None = None
    rest_fitness: float | None = None
    class_indicator: float | None = None
    jockey_win_rate: float | None = None
    trainer_win_rate: float | None = None
    gate_bias: float | None = None
    surface_affinity: float | None = None
    agf_edge: float | None = None  # market (AGF) deviation from uniform
    sire_win_rate: float | None = None  # sire's offspring overall win rate
    sire_surface_rate: float | None = None  # sire's offspring rate on this surface
    dam_win_rate: float | None = None  # dam's offspring overall win rate
    dam_surface_rate: float | None = None  # dam's offspring rate on this surface
    jockey_track_win_rate: float | None = None  # jockey's win rate at this track
    # Domain-derived signals that Turkish handicappers ("sürpriz at")
    # look for — decorrelated from AGF because the public often ignores
    # them.  All are 0/1 indicators except distance_delta_m.
    surface_switch: float | None = None  # 1 if surface differs from last race
    distance_delta_m: float | None = None  # current distance - last race distance
    equipment_changed: float | None = None  # 1 if equipment differs from last
    apprentice_jockey: float | None = None  # 1 if jockey name looks apprentice
    field_pace_density: float | None = None  # fraction of field that's front-running type
    track_affinity: float | None = None  # retained for compatibility
    s20_edge: float | None = None  # last-20-races score, relative to field average
    # Historical AGF-top1 hit rate for races of this type × field-size
    # bucket × surface, measured on resulted races strictly before the
    # current race date.  Values range ~0.15 (Handikap 16/H3) to ~0.61
    # (G3).  The point of this feature is to let the tree learn that
    # AGF carries different weight in different race regimes — in races
    # with low historical AGF reliability, the model should lean on
    # other features.
    agf_reliability: float | None = None
    # Late-money drift on this specific entry: AGF latest snapshot
    # minus AGF earliest snapshot.  Positive = late money came in
    # (insider/trainer signal that public didn't price at card open).
    # Requires the agf_snapshots table to have ≥2 rows per entry,
    # which only accumulates if scrape runs multiple times before
    # post-time.  None on cold-start days.
    late_agf_drift: float | None = None
    # Late program changes detected by comparing earliest vs latest
    # snapshot.  All three are 0/1 floats (None when <2 snapshots).
    late_jockey_change: float | None = None
    late_equipment_change: float | None = None
    late_gate_change: float | None = None
    # Number of distinct tipster tickets that picked this horse on
    # public aggregator sites (currently yarisrehberi.com).  ``None``
    # when no signals captured.  Higher = more "expert consensus".
    tipster_consensus: float | None = None
    # 1 when the listed jockey is on TJK's reported (medical) or
    # penalized list during the race date; 0 when scraped clean;
    # None pre-scrape.  Substitute rider expected — different feature
    # from late_jockey_change (which detects the substitution after
    # the snapshot diff catches it; this one detects BEFORE).
    jockey_discipline_flag: float | None = None
    # Workout-derived fitness signals from TJK İdman İstatistikleri.
    # ``None`` when no workouts captured for this horse yet.
    days_since_workout: float | None = None  # days between latest workout and today
    workout_speed_ms: float | None = None  # m/s on latest split closest to race distance
    n_workouts_recent: float | None = None  # distinct workout dates in lookback window
    # 1 when a steward report exists for this race's track-day.
    steward_report_flag: float | None = None


def compute_agf_edge(
    agf: float | None, field_size: int | None,
) -> float | None:
    """Turn raw AGF% into a relative edge vs uniform.

    ``agf`` is the horse's AGF percentage (0-100).  ``field_size`` is the
    number of runners.  Returns ``(agf - uniform) / uniform`` so a horse
    at the uniform share (no market preference) maps to 0, a favourite
    maps to > 0, and an outsider maps to < 0.  Clamped to +/- 6 to keep
    likelihoods from blowing up on lopsided markets.
    """
    if agf is None or field_size is None or field_size <= 0:
        return None
    uniform = 100.0 / field_size
    if uniform <= 0:
        return None
    edge = (agf - uniform) / uniform
    # Soft clamp — empirically AGF rarely exceeds ~6x uniform.
    if edge > 6.0:
        return 6.0
    if edge < -1.0:
        return -1.0
    return edge


def compute_speed_figure(
    eid_seconds: float | None, distance_meters: int | None
) -> float | None:
    """Normalize EID into speed figure (meters per second)."""
    if eid_seconds is None or distance_meters is None or eid_seconds <= 0:
        return None
    return distance_meters / eid_seconds


def compute_form_cycle(
    positions: list[int | None] | None,
) -> float | None:
    """Compute form score from recent finishing positions.

    Exponential decay weighting (most recent = highest weight).
    Returns 0-1 where 1 = best form.
    """
    if not positions:
        return None
    valid = [(i, p) for i, p in enumerate(positions) if p is not None]
    if not valid:
        return None
    n = len(positions)
    scores = []
    weights = []
    for i, pos in valid:
        weight = np.exp((i - n + 1) * 0.7)
        score = max(0.0, 1.0 - (pos - 1) * 0.15)
        scores.append(score)
        weights.append(weight)
    weights = np.array(weights)
    scores = np.array(scores)
    return float(np.average(scores, weights=weights))


def compute_weight_delta(
    horse_weight: float | None, field_avg_weight: float | None
) -> float | None:
    """Positive = lighter than average (advantage)."""
    if horse_weight is None or field_avg_weight is None:
        return None
    return (field_avg_weight - horse_weight) / field_avg_weight


def compute_s20_edge(
    s20: float | None, field_avg_s20: float | None,
) -> float | None:
    """Relative last-20-races score vs field.

    ``s20`` is the TJK-published "Son 20" score (roughly 0–30).  Returns
    ``(s20 - field_avg) / field_avg`` — positive for above-field horses,
    negative for below-field.  Same shape as ``compute_class_indicator``
    so the downstream likelihood block can amplify it consistently.
    """
    if s20 is None or field_avg_s20 is None or field_avg_s20 <= 0:
        return None
    return (s20 - field_avg_s20) / field_avg_s20


def compute_rest_fitness(kgs: int | None) -> float | None:
    """Gaussian curve centered on 21 days optimal rest."""
    if kgs is None:
        return None
    optimal = 21.0
    sigma = 15.0
    return float(np.exp(-((kgs - optimal) ** 2) / (2 * sigma**2)))


def compute_class_indicator(
    hp: float | None, field_avg_hp: float | None
) -> float | None:
    """Positive = horse has higher HP than field average."""
    if hp is None or field_avg_hp is None or field_avg_hp == 0:
        return None
    return (hp - field_avg_hp) / field_avg_hp


def compute_jockey_win_rate(
    session: Session,
    jockey: str | None,
    before_date: date_type | None = None,
) -> float | None:
    """Smoothed jockey win rate over historical resulted races.

    Returns a value in ``[0, 1]`` where 0.1 is the population baseline.
    ``before_date`` enforces temporal integrity — never look at races
    that happened on or after the race being predicted.
    """
    if not jockey:
        return None
    return _smoothed_person_win_rate(
        session, RaceEntry.jockey, jockey, before_date,
    )


def compute_sire_win_rate(
    session: Session,
    sire: str | None,
    before_date: date_type | None = None,
) -> float | None:
    """Smoothed win rate for all offspring of ``sire`` over historical runs.

    Uses the same Bayesian smoothing (10% prior, weight 20) as the other
    agent-rate features, so sires with only a handful of runners get
    pulled toward the population mean instead of swinging wildly.
    """
    if not sire:
        return None
    from ganyan.db.models import Horse

    base = (
        session.query(RaceEntry)
        .join(Horse, Horse.id == RaceEntry.horse_id)
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            Horse.sire == sire,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date is not None:
        base = base.filter(Race.date < before_date)

    runs = base.with_entities(func.count(RaceEntry.id)).scalar() or 0
    if runs == 0:
        return None
    wins = (
        base.filter(RaceEntry.finish_position == 1)
        .with_entities(func.count(RaceEntry.id))
        .scalar()
    ) or 0
    return _bayesian_smoothed_rate(wins, runs)


def compute_sire_surface_rate(
    session: Session,
    sire: str | None,
    surface: str | None,
    before_date: date_type | None = None,
) -> float | None:
    """Win rate for ``sire``'s offspring on the given ``surface`` (kum / çim).

    Lets the model notice breeding-specific track preferences — a sire
    whose offspring win 18% on turf but 7% on sand is a common pattern.
    """
    if not sire or not surface:
        return None
    from ganyan.db.models import Horse

    base = (
        session.query(RaceEntry)
        .join(Horse, Horse.id == RaceEntry.horse_id)
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            Horse.sire == sire,
            Race.surface == surface,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date is not None:
        base = base.filter(Race.date < before_date)

    runs = base.with_entities(func.count(RaceEntry.id)).scalar() or 0
    if runs == 0:
        return None
    wins = (
        base.filter(RaceEntry.finish_position == 1)
        .with_entities(func.count(RaceEntry.id))
        .scalar()
    ) or 0
    return _bayesian_smoothed_rate(wins, runs)


def compute_dam_win_rate(
    session: Session,
    dam: str | None,
    before_date: date_type | None = None,
) -> float | None:
    """Smoothed win rate for all offspring of ``dam``. Mirrors compute_sire_win_rate.

    Maternal genetic contribution is independent from paternal (sire) — about
    half of the heritable variance comes from each parent. Carrying both as
    features lets the tree weigh them separately.
    """
    if not dam:
        return None
    from ganyan.db.models import Horse

    base = (
        session.query(RaceEntry)
        .join(Horse, Horse.id == RaceEntry.horse_id)
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            Horse.dam == dam,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date is not None:
        base = base.filter(Race.date < before_date)

    runs = base.with_entities(func.count(RaceEntry.id)).scalar() or 0
    if runs == 0:
        return None
    wins = (
        base.filter(RaceEntry.finish_position == 1)
        .with_entities(func.count(RaceEntry.id))
        .scalar()
    ) or 0
    return _bayesian_smoothed_rate(wins, runs)


def compute_dam_surface_rate(
    session: Session,
    dam: str | None,
    surface: str | None,
    before_date: date_type | None = None,
) -> float | None:
    """Win rate for ``dam``'s offspring on the given surface."""
    if not dam or not surface:
        return None
    from ganyan.db.models import Horse

    base = (
        session.query(RaceEntry)
        .join(Horse, Horse.id == RaceEntry.horse_id)
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            Horse.dam == dam,
            Race.surface == surface,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date is not None:
        base = base.filter(Race.date < before_date)

    runs = base.with_entities(func.count(RaceEntry.id)).scalar() or 0
    if runs == 0:
        return None
    wins = (
        base.filter(RaceEntry.finish_position == 1)
        .with_entities(func.count(RaceEntry.id))
        .scalar()
    ) or 0
    return _bayesian_smoothed_rate(wins, runs)


def compute_jockey_track_win_rate(
    session: Session,
    jockey: str | None,
    track_id: int | None,
    before_date: date_type | None = None,
) -> float | None:
    """Smoothed win rate for ``jockey`` at the given ``track_id``.

    Track-specific jockey form: some riders consistently outperform their
    overall rate at familiar tracks (course shape, surface, gate draws).
    Bayesian-smoothed against the population mean so cold starts don't
    swing wildly.
    """
    if not jockey or track_id is None:
        return None

    base = (
        session.query(RaceEntry)
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            RaceEntry.jockey == jockey,
            Race.track_id == track_id,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date is not None:
        base = base.filter(Race.date < before_date)

    runs = base.with_entities(func.count(RaceEntry.id)).scalar() or 0
    if runs == 0:
        return None
    wins = (
        base.filter(RaceEntry.finish_position == 1)
        .with_entities(func.count(RaceEntry.id))
        .scalar()
    ) or 0
    return _bayesian_smoothed_rate(wins, runs)


def compute_trainer_win_rate(
    session: Session,
    trainer: str | None,
    before_date: date_type | None = None,
) -> float | None:
    """Smoothed trainer win rate over historical resulted races.

    Trainer is stored on :class:`Horse` (a snapshot of the *current*
    trainer).  This currently reflects that: a horse is credited to its
    present trainer across all historical runs.  Good enough for a
    first-pass feature; accuracy improves when we add a trainer-history
    table.
    """
    if not trainer:
        return None
    from ganyan.db.models import Horse  # local import to avoid cycle

    base = (
        session.query(RaceEntry)
        .join(Horse, Horse.id == RaceEntry.horse_id)
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            Horse.trainer == trainer,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date is not None:
        base = base.filter(Race.date < before_date)

    runs = base.with_entities(func.count(RaceEntry.id)).scalar() or 0
    if runs == 0:
        return None
    wins = (
        base.filter(RaceEntry.finish_position == 1)
        .with_entities(func.count(RaceEntry.id))
        .scalar()
    ) or 0
    return _bayesian_smoothed_rate(wins, runs)


def _smoothed_person_win_rate(
    session: Session,
    column,
    value: str,
    before_date: date_type | None,
) -> float | None:
    """Internal: compute win rate where a direct RaceEntry column equals value."""
    q = (
        session.query(
            func.count(RaceEntry.id).label("runs"),
        )
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            column == value,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position.isnot(None),
        )
    )
    if before_date is not None:
        q = q.filter(Race.date < before_date)
    runs_row = q.one()
    runs = runs_row.runs or 0
    if runs == 0:
        return None

    wins_q = (
        session.query(func.count(RaceEntry.id))
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            column == value,
            Race.status == RaceStatus.resulted,
            RaceEntry.finish_position == 1,
        )
    )
    if before_date is not None:
        wins_q = wins_q.filter(Race.date < before_date)
    wins = wins_q.scalar() or 0
    return _bayesian_smoothed_rate(wins, runs)


def compute_surface_switch(
    session: Session,
    horse_id: int | None,
    current_surface: str | None,
    before_date: date_type | None,
) -> float | None:
    """Return 1 if the horse's last resulted race was on a different
    surface than the current race, 0 if same surface, ``None`` if we
    have no prior race to compare to.
    """
    if horse_id is None or current_surface is None or before_date is None:
        return None
    prev = (
        session.query(Race.surface)
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.horse_id == horse_id,
            Race.status == RaceStatus.resulted,
            Race.date < before_date,
            Race.surface.isnot(None),
        )
        .order_by(Race.date.desc())
        .limit(1)
        .scalar()
    )
    if prev is None:
        return None
    return 1.0 if prev != current_surface else 0.0


def compute_distance_delta(
    session: Session,
    horse_id: int | None,
    current_distance: int | None,
    before_date: date_type | None,
) -> float | None:
    """Meters-change from this horse's last-raced distance.  Positive =
    stepping up, negative = stepping down.  ``None`` when no history.
    """
    if horse_id is None or current_distance is None or before_date is None:
        return None
    prev = (
        session.query(Race.distance_meters)
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .filter(
            RaceEntry.horse_id == horse_id,
            Race.status == RaceStatus.resulted,
            Race.date < before_date,
            Race.distance_meters.isnot(None),
        )
        .order_by(Race.date.desc())
        .limit(1)
        .scalar()
    )
    if prev is None:
        return None
    return float(current_distance - prev)


def compute_equipment_changed(
    session: Session,
    horse_id: int | None,
    current_equipment: str | None,
    before_date: date_type | None,
) -> float | None:
    """1 if equipment differs from this horse's last race, 0 if same."""
    if horse_id is None or before_date is None:
        return None
    prev = (
        session.query(RaceEntry.equipment)
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(
            RaceEntry.horse_id == horse_id,
            Race.status == RaceStatus.resulted,
            Race.date < before_date,
        )
        .order_by(Race.date.desc())
        .limit(1)
        .scalar()
    )
    if prev is None:
        return None
    norm_prev = (prev or "").strip() or None
    norm_curr = (current_equipment or "").strip() or None
    return 1.0 if norm_prev != norm_curr else 0.0


def compute_apprentice_jockey(jockey: str | None) -> float | None:
    """Return ``None`` until the scraper preserves apprentice markers.

    The TJK page marks apprentices with a ``<sup>`` superscript inside
    the jockey cell, but the parser currently strips that tag before
    storing the name.  The previous heuristic regex (``\\bA\\.``) matched
    the initial of any jockey named "A. X" — a false-positive magnet
    that taught the LightGBM ranker a spurious signal keyed on first
    initials.  Until the scrape preserves a real ``is_apprentice`` bit,
    this feature yields no information, so we return ``None`` (which
    LightGBM's missing-value handler treats as an explicit NA rather
    than a noisy 0/1 label).
    """
    return None


def compute_field_pace_density(
    last_six_by_horse: list[list[int | None] | None],
) -> float | None:
    """Estimate "how speed-laden is this race?".

    For each horse with known last-six positions, we count how many
    front-running finishes (1, 2, 3) they had in their last 6.  A horse
    with >=3 top-3 finishes is treated as a likely front-runner.  The
    returned value is ``front_runners / field_size`` — higher means a
    speed-duel scenario, which classically favours closers.

    Denominator is the *full field*, not just horses with history.  Two
    first-timers alongside one speed horse shouldn't read as a 1.0 pace
    duel; it's 1/3.  Returns ``None`` only when the entire field has no
    history (can't judge pace at all).
    """
    if not last_six_by_horse:
        return None
    field_size = len(last_six_by_horse)
    valid = [ls for ls in last_six_by_horse if ls]
    if not valid:
        return None
    front_runners = 0
    for ls in valid:
        top3 = sum(1 for p in ls if p is not None and p <= 3)
        if top3 >= 3:
            front_runners += 1
    return front_runners / field_size


def _field_size_bucket(field_size: int | None) -> str | None:
    """Bucket field size into the brackets the historical analysis shows
    matter for AGF reliability (1-6 / 7-10 / 11-14 / 15+)."""
    if field_size is None or field_size <= 0:
        return None
    if field_size <= 6:
        return "1-6"
    if field_size <= 10:
        return "7-10"
    if field_size <= 14:
        return "11-14"
    return "15+"


def compute_agf_reliability(
    session: Session,
    race_type: str | None,
    field_size: int | None,
    surface: str | None,
    before_date: date_type | None = None,
    min_sample: int = 30,
) -> float | None:
    """Empirical AGF-top1 hit rate for the regime (race_type × field-size
    bucket × surface), measured on races strictly before ``before_date``.

    This captures the pattern the 2026-04-24 race day + the historical
    data exposed: AGF is ~61% accurate in G3 races but only ~15% in
    "Handikap 16/H3" and ~25% in 15+ horse fields.  Giving the tree this
    feature lets it learn to down-weight AGF in races where AGF is
    historically unreliable and lean on other features instead.

    Returns ``None`` when the historical sample for the regime is below
    ``min_sample`` — the regime hit rate would be too noisy to trust.
    """
    if field_size is None or field_size <= 0:
        return None
    bucket = _field_size_bucket(field_size)

    filters = [Race.status == RaceStatus.resulted]
    if race_type is not None:
        filters.append(Race.race_type == race_type)
    if surface is not None:
        filters.append(Race.surface == surface)
    if before_date is not None:
        filters.append(Race.date < before_date)

    # Bucket by field size in SQL via a subquery with entry counts.
    subq = (
        session.query(
            Race.id.label("race_id"),
            func.count(RaceEntry.id).label("n_entries"),
        )
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .filter(*filters)
        .group_by(Race.id)
        .subquery()
    )

    if bucket == "1-6":
        size_filter = subq.c.n_entries <= 6
    elif bucket == "7-10":
        size_filter = and_(subq.c.n_entries >= 7, subq.c.n_entries <= 10)
    elif bucket == "11-14":
        size_filter = and_(subq.c.n_entries >= 11, subq.c.n_entries <= 14)
    else:
        size_filter = subq.c.n_entries >= 15

    matching_race_ids = (
        session.query(subq.c.race_id).filter(size_filter).subquery()
    )

    # Count races in regime, and races where the AGF-top-1 horse won.
    total = (
        session.query(func.count())
        .select_from(matching_race_ids)
        .scalar()
    ) or 0
    if total < min_sample:
        return None

    # AGF-top1-wins: horse with highest AGF in that race finished 1st.
    # Done via a window function in raw SQL for speed over 10K+ races.
    from sqlalchemy import text
    where_parts = ["r.status = 'resulted'"]
    params: dict[str, object] = {}
    if race_type is not None:
        where_parts.append("r.race_type = :race_type")
        params["race_type"] = race_type
    if surface is not None:
        where_parts.append("r.surface = :surface")
        params["surface"] = surface
    if before_date is not None:
        where_parts.append("r.date < :before_date")
        params["before_date"] = before_date

    if bucket == "1-6":
        size_clause = "field_size <= 6"
    elif bucket == "7-10":
        size_clause = "field_size BETWEEN 7 AND 10"
    elif bucket == "11-14":
        size_clause = "field_size BETWEEN 11 AND 14"
    else:
        size_clause = "field_size >= 15"

    sql = text(f"""
        WITH agf_rank AS (
            SELECT e.race_id, e.finish_position, e.agf,
                   RANK() OVER (PARTITION BY e.race_id ORDER BY e.agf DESC NULLS LAST) AS agf_rk,
                   (SELECT COUNT(*) FROM race_entries ee WHERE ee.race_id=e.race_id) AS field_size
            FROM race_entries e
            JOIN races r ON r.id = e.race_id
            WHERE {' AND '.join(where_parts)}
              AND e.agf IS NOT NULL
        )
        SELECT
            COUNT(DISTINCT race_id) AS n,
            SUM(CASE WHEN finish_position=1 AND agf_rk=1 THEN 1 ELSE 0 END) AS agf_hits
        FROM agf_rank
        WHERE {size_clause}
    """)
    row = session.execute(sql, params).fetchone()
    if row is None or row[0] is None or row[0] < min_sample:
        return None
    n, hits = int(row[0]), int(row[1] or 0)
    return hits / n


def precompute_agf_reliability_table(
    session: Session,
    before_date: date_type | None = None,
    min_sample: int = 30,
) -> dict[tuple[str | None, str, str | None], float]:
    """Build a lookup of historical AGF-top1 hit rate per race regime.

    Key: ``(race_type, field_size_bucket, surface)``.  Value: empirical
    AGF-top1 win rate on resulted races strictly before ``before_date``.
    Regimes with fewer than ``min_sample`` historical races are
    suppressed so the tree doesn't learn off a 5-race fluke.

    One aggregate SQL instead of one query per race during the training
    loop — 50-100× faster on the 6k-race training frame.
    """
    from sqlalchemy import text

    where = ["r.status = 'resulted'"]
    params: dict[str, object] = {}
    if before_date is not None:
        where.append("r.date < :before_date")
        params["before_date"] = before_date
    params["min_sample"] = min_sample

    sql = text(f"""
        WITH race_stats AS (
            SELECT r.id,
                   r.race_type,
                   r.surface,
                   (SELECT COUNT(*) FROM race_entries e
                    WHERE e.race_id = r.id) AS n_entries,
                   (SELECT e.finish_position FROM race_entries e
                    WHERE e.race_id = r.id AND e.agf IS NOT NULL
                    ORDER BY e.agf DESC NULLS LAST, e.id
                    LIMIT 1) AS agf_top1_finish
            FROM races r
            WHERE {' AND '.join(where)}
        )
        SELECT race_type, surface,
               CASE WHEN n_entries <= 6 THEN '1-6'
                    WHEN n_entries <= 10 THEN '7-10'
                    WHEN n_entries <= 14 THEN '11-14'
                    ELSE '15+' END AS bucket,
               COUNT(*) AS n,
               SUM(CASE WHEN agf_top1_finish = 1 THEN 1 ELSE 0 END) AS hits
        FROM race_stats
        WHERE agf_top1_finish IS NOT NULL
        GROUP BY race_type, surface, bucket
        HAVING COUNT(*) >= :min_sample
    """)
    table: dict[tuple[str | None, str, str | None], float] = {}
    for row in session.execute(sql, params):
        race_type, surface, bucket, n, hits = row
        if n == 0:
            continue
        table[(race_type, bucket, surface)] = float(hits) / float(n)
    return table


def lookup_agf_reliability(
    table: dict[tuple[str | None, str, str | None], float],
    race_type: str | None,
    field_size: int | None,
    surface: str | None,
) -> float | None:
    """Look up a precomputed reliability value; ``None`` if regime missing."""
    bucket = _field_size_bucket(field_size)
    if bucket is None:
        return None
    return table.get((race_type, bucket, surface))


def compute_steward_report_flag(
    session: Session, race_id: int | None,
) -> float | None:
    """1 when a steward report exists for this race's date+track, else 0.

    Coarse signal: the existence of a report often correlates with
    incident-rich race days (stewards' inquiries, disqualifications,
    weather disruptions).  Useful as a yes/no without text mining.
    """
    from ganyan.db.models import ExternalSignal

    if race_id is None:
        return None
    found = (
        session.query(ExternalSignal.id)
        .filter(
            ExternalSignal.race_id == race_id,
            ExternalSignal.signal_type == "steward_report",
        )
        .first()
    )
    if found is not None:
        return 1.0
    # Distinguish "no report" from "haven't scraped yet".  The existence
    # check must be scoped to THIS race's date: an unscoped "any row
    # ever" check flips every pre-plugin historical entry from None to
    # 0.0 the moment the plugin first runs — a calendar-position leak
    # sitting on both sides of the temporal holdout.
    race_date = (
        session.query(Race.date).filter(Race.id == race_id).scalar()
    )
    if race_date is None:
        return None
    any_same_day = (
        session.query(ExternalSignal.id)
        .join(Race, ExternalSignal.race_id == Race.id)
        .filter(
            ExternalSignal.source_name == "tjk_steward_reports",
            Race.date == race_date,
        )
        .first()
    )
    return 0.0 if any_same_day is not None else None


def compute_workout_signals(
    session: Session, race_entry_id: int | None,
    race_distance_m: int | None = None,
    race_date: date_type | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Three workout-derived features for the entry's horse:

    1. ``days_since_workout`` — days between the most recent workout
       and the race date.  Lower = recently trained.  ``None`` until a
       workout is captured, or until ``race_date`` is supplied (the
       feature requires a temporal anchor — falls back to today when
       race_date is None for back-compat with live predict callers).
    2. ``workout_speed_ms`` — meters-per-second on the most recent
       timed split.  Higher = sharper recent work.  Picks the split
       closest to ``race_distance_m`` when provided, else the longest
       available split.
    3. ``n_workouts_recent`` — count of distinct workout dates
       captured in the lookback window.  More workouts = more recent
       activity.  Capped at signal cardinality.

    Bound from ``external_signals`` rows where ``signal_type ==
    'workout_split'`` and the row resolves to this entry.
    """
    from ganyan.db.models import ExternalSignal

    if race_entry_id is None:
        return None, None, None
    rows = (
        session.query(ExternalSignal.value, ExternalSignal.payload)
        .filter(
            ExternalSignal.race_entry_id == race_entry_id,
            ExternalSignal.signal_type == "workout_split",
        )
        .all()
    )
    if not rows:
        return None, None, None

    most_recent_date: date_type | None = None
    workout_dates: set = set()
    best_split: tuple[int, float] | None = None  # (distance, seconds)
    target_distance = race_distance_m or 1000

    for value, payload in rows:
        if payload is None or value is None:
            continue
        d = payload.get("workout_date")
        try:
            wd = date_type.fromisoformat(d) if d else None
        except ValueError:
            wd = None
        if wd is None:
            continue
        workout_dates.add(wd)
        if most_recent_date is None or wd > most_recent_date:
            most_recent_date = wd
        # Track the split closest to target distance from the most
        # recent workout — pick by smallest distance gap, breaking
        # ties on a more recent date.
        dist = int(payload.get("distance_m") or 0)
        if dist <= 0:
            continue
        gap = abs(dist - target_distance)
        if best_split is None:
            best_split = (dist, float(value))
            best_date = wd
        else:
            # Prefer smaller distance gap; on tie prefer fresher date.
            curr_dist, _ = best_split
            curr_gap = abs(curr_dist - target_distance)
            if gap < curr_gap or (gap == curr_gap and wd > best_date):
                best_split = (dist, float(value))
                best_date = wd

    days_since = None
    if most_recent_date is not None:
        from datetime import date as _date
        anchor = race_date if race_date is not None else _date.today()
        days_since = float((anchor - most_recent_date).days)

    workout_speed_ms = None
    if best_split is not None and best_split[1] > 0:
        dist, sec = best_split
        workout_speed_ms = dist / sec

    n_workouts = float(len(workout_dates)) if workout_dates else None

    return days_since, workout_speed_ms, n_workouts


def compute_jockey_discipline_flag(
    session: Session, race_entry_id: int | None,
) -> float | None:
    """1 when this entry's listed jockey is on TJK's reported or penalized
    list with an active discipline window covering the race date; else 0.

    Strong sürpriz-at indicator: a disciplined jockey will be replaced
    by a substitute (often an apprentice).  The substitute's identity
    isn't in the morning program, so the model would otherwise be
    blind to the change.

    Returns ``None`` when the discipline scrape hasn't run yet — keeps
    "I don't know" distinguishable from "I know there's no flag".
    """
    from ganyan.db.models import ExternalSignal

    if race_entry_id is None:
        return None
    # Existence query.  Any matching signal — regardless of reported
    # vs penalized — flips the flag to 1.0.  Both kinds of discipline
    # produce the same operational outcome (substitute rider).
    q = (
        session.query(ExternalSignal.id)
        .filter(
            ExternalSignal.race_entry_id == race_entry_id,
            ExternalSignal.source_name == "tjk_discipline",
        )
        .limit(1)
    )
    if session.query(q.exists()).scalar():
        return 1.0
    # Distinguish "no signal at all" (None — pre-scrape) from "scraped
    # but clean" (0).  Scoped to THIS entry's race date: "any
    # tjk_discipline row ever" flips every pre-plugin historical entry
    # from None to 0.0 — a calendar-position leak.  A genuinely clean
    # day (no jockeys flagged league-wide) now reads None rather than
    # 0.0; LightGBM branches on missingness, so the honesty is cheap.
    entry_race_date = (
        session.query(Race.date)
        .join(RaceEntry, RaceEntry.race_id == Race.id)
        .filter(RaceEntry.id == race_entry_id)
        .scalar()
    )
    if entry_race_date is None:
        return None
    any_same_day = (
        session.query(ExternalSignal.id)
        .join(RaceEntry, ExternalSignal.race_entry_id == RaceEntry.id)
        .join(Race, RaceEntry.race_id == Race.id)
        .filter(
            ExternalSignal.source_name == "tjk_discipline",
            Race.date == entry_race_date,
        )
        .limit(1)
    )
    if session.query(any_same_day.exists()).scalar():
        return 0.0
    return None


def compute_tipster_consensus(
    session: Session, race_entry_id: int | None,
) -> float | None:
    """Count distinct tipster tickets that picked this entry.

    Pulled from ``external_signals`` where ``signal_type =
    'tipster_pick'``.  We deduplicate by ``payload->ticket_timestamp``
    so a single ticket containing the same horse multiple times (rare
    but possible in box-style picks) counts once.

    Returns ``None`` when no signals exist for the entry — keeps the
    feature distinguishable from "0 tipsters picked it" via NaN
    handling at the model level.
    """
    from ganyan.db.models import ExternalSignal

    if race_entry_id is None:
        return None
    rows = (
        session.query(ExternalSignal.payload)
        .filter(
            ExternalSignal.race_entry_id == race_entry_id,
            ExternalSignal.signal_type == "tipster_pick",
        )
        .all()
    )
    if not rows:
        return None
    seen: set[str] = set()
    for (payload,) in rows:
        if not payload:
            continue
        ts = payload.get("ticket_timestamp")
        title = payload.get("ticket_title") or ""
        # (timestamp, title) is the de-facto ticket key.
        seen.add(f"{ts}|{title}")
    return float(len(seen))


def compute_late_program_changes(
    session: Session, race_entry_id: int | None,
) -> tuple[float | None, float | None, float | None]:
    """Detect late jockey / equipment / gate changes for an entry.

    Compares the *earliest* snapshot to the *latest* snapshot in
    ``agf_snapshots``.  Returns three flags (any may be ``None`` when
    fewer than 2 snapshots have accumulated):

    1. ``late_jockey_change`` — 1 if the jockey name in the latest
       snapshot differs from the earliest, else 0.  Strong signal: a
       regular rider getting reported (medical) or punished forces an
       apprentice substitution within hours of post.
    2. ``late_equipment_change`` — 1 if the equipment string changed.
       Indicates a trainer-led tactical shift (first-time blinkers,
       tongue tie added) the morning program didn't reflect.
    3. ``late_gate_change`` — 1 if gate_number changed.  Rare but
       happens when scratchings shift the field; useful for surface-
       sensitive races.

    Becomes informative once the daemon has accumulated 2+ snapshots
    per entry; until then all three return ``None``.
    """
    from ganyan.db.models import AgfSnapshot

    if race_entry_id is None:
        return None, None, None
    rows = (
        session.query(
            AgfSnapshot.jockey, AgfSnapshot.equipment,
            AgfSnapshot.gate_number, AgfSnapshot.taken_at,
        )
        .filter(AgfSnapshot.race_entry_id == race_entry_id)
        .order_by(AgfSnapshot.taken_at.asc())
        .all()
    )
    if len(rows) < 2:
        return None, None, None

    first_jockey, first_equip, first_gate, _ = rows[0]
    last_jockey, last_equip, last_gate, _ = rows[-1]

    # Use 0/1 floats for downstream tree splits; str/str comparison
    # treats whitespace and case strictly so spurious normalisation
    # doesn't trigger a false positive.
    def _changed(a, b) -> float | None:
        if a is None and b is None:
            return None
        if a is None or b is None:
            return 1.0  # appearing/disappearing IS a change
        return 1.0 if str(a).strip() != str(b).strip() else 0.0

    return (
        _changed(first_jockey, last_jockey),
        _changed(first_equip, last_equip),
        _changed(first_gate, last_gate),
    )


def compute_late_agf_drift(
    session: Session, race_entry_id: int | None,
) -> float | None:
    """AGF drift between the earliest and latest snapshot for an entry.

    Positive = AGF rose since open (late money came in for this horse).
    Negative = AGF dropped (money flowed away).  Returns ``None`` when
    there are fewer than 2 snapshots for the entry — typical for entries
    fetched in a single morning scrape.  Becomes meaningful only after a
    schedule (or repeated manual scrape) collects multiple readings per
    race over the course of a day.
    """
    from ganyan.db.models import AgfSnapshot

    if race_entry_id is None:
        return None
    rows = (
        session.query(AgfSnapshot.agf, AgfSnapshot.taken_at)
        .filter(AgfSnapshot.race_entry_id == race_entry_id)
        .order_by(AgfSnapshot.taken_at.asc())
        .all()
    )
    if len(rows) < 2:
        return None
    earliest = float(rows[0][0])
    latest = float(rows[-1][0])
    return latest - earliest


def _bayesian_smoothed_rate(wins: int, runs: int) -> float:
    """Apply a Beta(prior_mean·weight, (1-prior_mean)·weight) smoothing."""
    alpha = _WINRATE_PRIOR_MEAN * _WINRATE_PRIOR_WEIGHT
    beta = (1.0 - _WINRATE_PRIOR_MEAN) * _WINRATE_PRIOR_WEIGHT
    return float((wins + alpha) / (runs + alpha + beta))


def compute_gate_bias(
    gate_number: int | None,
    distance_meters: int | None,
    surface: str | None,
) -> float | None:
    """Heuristic gate-bias score in ``[-1, 1]``.

    Short sand (kum) races advantage inside gates; long turf (çim) races
    are relatively neutral.  This is a coarse prior — feed the empirical
    gate-specific win rate when enough data is accumulated.
    """
    if gate_number is None:
        return None
    if distance_meters is None:
        return 0.0

    # Normalise gate to 0..1 assuming a typical 14-horse field.
    normalised = (gate_number - 1) / 13.0

    if surface and surface.lower().startswith("kum") and distance_meters <= 1400:
        # Inside gates slightly favoured on short sand.
        return float(1.0 - 2 * normalised) * 0.5
    if surface and surface.lower().startswith("çim") and distance_meters >= 1800:
        # Mid gates slightly favoured on long turf (draw matters less).
        return float(1.0 - abs(normalised - 0.5) * 2.0) * 0.3

    # Default: inside-tilt is small.
    return float(0.5 - normalised) * 0.2


def compute_surface_affinity(
    session: Session,
    horse_id: int,
    surface: str | None,
    distance_meters: int | None,
    before_date: date_type | None = None,
    distance_band: int = 200,
) -> float | None:
    """Horse's win rate on similar surface/distance combination.

    Returns ``None`` if the horse has no historical runs matching the
    profile.  Otherwise returns a smoothed win rate in ``[0, 1]``.
    """
    if horse_id is None:
        return None

    filters = [
        RaceEntry.horse_id == horse_id,
        RaceEntry.finish_position.isnot(None),
        Race.status == RaceStatus.resulted,
    ]
    if surface is not None:
        filters.append(Race.surface == surface)
    if distance_meters is not None:
        filters.append(
            and_(
                Race.distance_meters >= distance_meters - distance_band,
                Race.distance_meters <= distance_meters + distance_band,
            )
        )
    if before_date is not None:
        filters.append(Race.date < before_date)

    runs = (
        session.query(func.count(RaceEntry.id))
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(*filters)
        .scalar()
    ) or 0
    if runs == 0:
        return None

    wins = (
        session.query(func.count(RaceEntry.id))
        .join(Race, Race.id == RaceEntry.race_id)
        .filter(*filters, RaceEntry.finish_position == 1)
        .scalar()
    ) or 0
    return _bayesian_smoothed_rate(wins, runs)


def extract_features(
    eid_seconds: float | None = None,
    distance_meters: int | None = None,
    last_six_parsed: list[int | None] | None = None,
    weight_kg: float | None = None,
    field_avg_weight: float | None = None,
    kgs: int | None = None,
    hp: float | None = None,
    field_avg_hp: float | None = None,
    s20: float | None = None,
    field_avg_s20: float | None = None,
    *,
    session: Session | None = None,
    jockey: str | None = None,
    trainer: str | None = None,
    horse_id: int | None = None,
    gate_number: int | None = None,
    surface: str | None = None,
    race_date: date_type | None = None,
    agf: float | None = None,
    field_size: int | None = None,
    sire: str | None = None,
    dam: str | None = None,
    track_id: int | None = None,
    equipment: str | None = None,
    field_pace_density: float | None = None,
    agf_reliability: float | None = None,
    race_entry_id: int | None = None,
    race_id_for_signals: int | None = None,
) -> HorseFeatures:
    features = HorseFeatures(
        speed_figure=compute_speed_figure(eid_seconds, distance_meters),
        form_cycle=compute_form_cycle(last_six_parsed),
        weight_delta=compute_weight_delta(weight_kg, field_avg_weight),
        rest_fitness=compute_rest_fitness(kgs),
        class_indicator=compute_class_indicator(hp, field_avg_hp),
        gate_bias=compute_gate_bias(gate_number, distance_meters, surface),
        agf_edge=compute_agf_edge(agf, field_size),
        apprentice_jockey=compute_apprentice_jockey(jockey),
        field_pace_density=field_pace_density,
        s20_edge=compute_s20_edge(s20, field_avg_s20),
        agf_reliability=agf_reliability,
    )
    if session is not None:
        features.late_agf_drift = compute_late_agf_drift(session, race_entry_id)
        (
            features.late_jockey_change,
            features.late_equipment_change,
            features.late_gate_change,
        ) = compute_late_program_changes(session, race_entry_id)
        features.tipster_consensus = compute_tipster_consensus(
            session, race_entry_id,
        )
        features.jockey_discipline_flag = compute_jockey_discipline_flag(
            session, race_entry_id,
        )
        (
            features.days_since_workout,
            features.workout_speed_ms,
            features.n_workouts_recent,
        ) = compute_workout_signals(
            session, race_entry_id, distance_meters, race_date=race_date,
        )
        features.steward_report_flag = compute_steward_report_flag(
            session, race_id_for_signals,
        )
        features.jockey_win_rate = compute_jockey_win_rate(
            session, jockey, before_date=race_date,
        )
        features.trainer_win_rate = compute_trainer_win_rate(
            session, trainer, before_date=race_date,
        )
        features.sire_win_rate = compute_sire_win_rate(
            session, sire, before_date=race_date,
        )
        features.sire_surface_rate = compute_sire_surface_rate(
            session, sire, surface, before_date=race_date,
        )
        features.dam_win_rate = compute_dam_win_rate(
            session, dam, before_date=race_date,
        )
        features.dam_surface_rate = compute_dam_surface_rate(
            session, dam, surface, before_date=race_date,
        )
        features.jockey_track_win_rate = compute_jockey_track_win_rate(
            session, jockey, track_id, before_date=race_date,
        )
        if horse_id is not None:
            features.surface_affinity = compute_surface_affinity(
                session, horse_id, surface, distance_meters,
                before_date=race_date,
            )
            features.track_affinity = features.surface_affinity
            features.surface_switch = compute_surface_switch(
                session, horse_id, surface, race_date,
            )
            features.distance_delta_m = compute_distance_delta(
                session, horse_id, distance_meters, race_date,
            )
            features.equipment_changed = compute_equipment_changed(
                session, horse_id, equipment, race_date,
            )
    return features
