"""Feature-matrix builder for the LightGBM ranker.

Walks resulted race entries from the DB and produces a pandas
:class:`~pandas.DataFrame` shaped for a rank-aware training loop:

- one row per horse per race
- group key = ``race_id`` (LightGBM ``group`` parameter)
- rank-score target = ``field_size - finish_position`` so winner has the
  highest score
- features include the current engineered signals (jockey/trainer win
  rate, surface affinity, etc.) *plus* raw values that let tree splits
  discover non-linear effects the hand-tuned log-linear model can't

Kept as pure functions so the same builder serves training and
inference (single horse/race → single-row frame).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_type

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, joinedload

from ganyan.db.models import Race, RaceEntry, RaceStatus
from ganyan.predictor.features import (
    compute_field_pace_density, extract_features,
    lookup_agf_reliability, precompute_agf_reliability_table,
)
from ganyan.scraper.parser import parse_eid_to_seconds, parse_last_six


# Engineered + raw columns used as model inputs.  Order is load-bearing:
# LightGBM models serialize a feature-name list and the predictor uses
# this constant to re-assemble inference frames in the right order.
FEATURE_COLUMNS: list[str] = [
    # Engineered (from features.extract_features).
    "speed_figure",
    "form_cycle",
    "weight_delta",
    "rest_fitness",
    "class_indicator",
    "jockey_win_rate",
    "trainer_win_rate",
    "gate_bias",
    "surface_affinity",
    "agf_edge",
    "sire_win_rate",
    "sire_surface_rate",
    "dam_win_rate",
    "dam_surface_rate",
    "jockey_track_win_rate",
    # Domain-derived signals ("sürpriz at" features).
    "surface_switch",
    "distance_delta_m",
    "equipment_changed",
    # "apprentice_jockey" removed 2026-08-13: compute_apprentice_jockey
    # hard-returns None (parser strips TJK's <sup> apprentice marker), so
    # the column was a permanently all-NaN input in every head.
    "field_pace_density",
    # Last-20-races score — engineered (vs field avg) and raw.
    "s20_edge",
    # Historical AGF-top1 hit rate for this race's regime (race_type ×
    # field-size bucket × surface).  Lets the tree learn where AGF is
    # trustworthy vs where it's noise (e.g. Handikap 16/H3 at 15% vs
    # G3 at 61% — a 4× spread confirmed on 7k+ historical races).
    "agf_reliability",
    # Late-money drift on this entry: difference between latest and
    # earliest AGF snapshot.  ``None`` for entries with <2 snapshots.
    # Becomes informative after a few days of multi-fetch scrapes.
    "late_agf_drift",
    # Late program changes — 0/1 indicators detected by comparing
    # earliest vs latest snapshot's jockey / equipment / gate fields.
    # ``late_jockey_change`` is the strongest sürpriz-at indicator we
    # can extract from public data — a regular jockey getting reported
    # (medical) or punished forces a substitution within hours of post.
    "late_jockey_change",
    "late_equipment_change",
    "late_gate_change",
    # Count of distinct tipster tickets (from external aggregators)
    # that picked this horse.  Independent crowd-wisdom signal vs the
    # parimutuel market AGF — populated as ``external_signals``
    # accumulates from scheduled scrapes.  NaN until first scrape.
    "tipster_consensus",
    # 1 when the listed jockey is on TJK's reported/penalized list,
    # 0 when scraped clean, NaN pre-scrape.  Predicts substitute-
    # rider scenarios before the morning program shows the change.
    "jockey_discipline_flag",
    # Workout-derived fitness signals from TJK İdman İstatistikleri.
    # NaN until first workout scrape captures rows for this horse.
    # Triple-feature: recency, raw speed, and frequency of training.
    "days_since_workout",
    "workout_speed_ms",
    "n_workouts_recent",
    # Steward-report existence flag — coarse "incident-rich race day"
    # indicator.  Full text mining deferred until existence flag
    # itself shows predictive value.
    "steward_report_flag",
    # Raw values — give the tree room to learn non-linear effects.
    "agf_raw",
    "hp_raw",
    "weight_kg_raw",
    "kgs_raw",
    "s20_raw",
    "gate_number",
    "age",
    # Race-level context.
    "distance_meters",
    "field_size",
    # Surface: 1.0 kum, 0.0 çim, NaN unknown (lets LightGBM's missing-
    # value branch handle "not yet published / weird surface" rather
    # than bucketing it with Sentetik under a shared sentinel.
    "surface_is_kum",
]

TARGET_COLUMN = "rank_score"
EV_TARGET_COLUMN = "ev_target"
FINISH_TIME_TARGET_COLUMN = "finish_seconds"
GROUP_COLUMN = "race_id"


@dataclass
class TrainingFrame:
    """Container returned by :func:`build_training_frame`."""

    features: pd.DataFrame  # shape (n_rows, len(FEATURE_COLUMNS))
    target: pd.Series  # rank score per row
    ev_target: pd.Series  # per-1-TL bet return per row (winners only realised)
    finish_time_target: pd.Series  # actual finish time in seconds (NaN when missing)
    groups: pd.Series  # race_id per row (used for LGBM grouping)
    race_dates: pd.Series  # race.date per row (used for temporal split)

    @property
    def race_ids_ordered(self) -> np.ndarray:
        """Unique race_ids in the order they appear (for group sizes)."""
        return self.groups.drop_duplicates().to_numpy()

    def group_sizes(self) -> np.ndarray:
        """Contiguous group sizes as required by LightGBM."""
        return self.groups.groupby(self.groups, sort=False).size().to_numpy()


def _surface_encode(surface: str | None) -> float:
    """Encode surface as 1 (kum), 0 (çim), NaN (unknown/other).

    NaN is intentional — LightGBM has a dedicated missing-value branch
    per split, so "surface not known yet" is distinguishable from a
    third surface class.  Previously we encoded both as ``-1``, which
    let the tree learn a conditional that conflated missing data with
    the Sentetik surface.
    """
    if surface is None:
        return np.nan
    s = surface.lower()
    if s.startswith("kum"):
        return 1.0
    if s.startswith("çim") or s.startswith("cim"):
        return 0.0
    return np.nan


def build_training_frame(
    session: Session,
    *,
    from_date: date_type | None = None,
    to_date: date_type | None = None,
    require_agf: bool = True,
    min_field_size: int = 3,
    race_type_prefix: str | None = None,
) -> TrainingFrame:
    """Extract a feature matrix from resulted races in the DB.

    Parameters
    ----------
    from_date, to_date:
        Optional inclusive bounds on ``race.date``.
    require_agf:
        If ``True`` (default), skip races where no entry has AGF — these
        are the historical winners-only rows from the KosuSorgulama path
        and have no value for rank training.
    min_field_size:
        Races with fewer resulted entries than this are dropped (too
        sparse for meaningful ranking).
    race_type_prefix:
        Restrict to races whose ``race_type`` starts with this string
        (e.g. ``"Handikap"`` or ``"Maiden"``).  Used by per-race-type
        specialist trainers.  ``None`` means no filter.
    """
    q = (
        session.query(Race)
        .options(
            joinedload(Race.entries).joinedload(RaceEntry.horse),
            joinedload(Race.track),
        )
        .filter(Race.status == RaceStatus.resulted)
    )
    if from_date is not None:
        q = q.filter(Race.date >= from_date)
    if to_date is not None:
        q = q.filter(Race.date <= to_date)
    if race_type_prefix is not None:
        q = q.filter(Race.race_type.like(f"{race_type_prefix}%"))

    candidate_races = q.order_by(Race.date.asc(), Race.race_number.asc()).all()
    # AGF-reliability regime tables, one per race DATE (cached), each
    # with before_date = that date.  Mirrors the inference path exactly
    # (build_race_frame uses before_date=race.date).  The old single
    # snapshot at the window's earliest date was leak-free but
    # systematically STALER than what the model sees at serve time — a
    # train/serve distribution shift on a top-importance feature.
    reliability_tables_by_date: dict = {}

    rows: list[dict] = []
    for race in candidate_races:
        entries = [
            e for e in race.entries
            if e.finish_position is not None
        ]
        if len(entries) < min_field_size:
            continue
        if require_agf and not any(e.agf is not None for e in entries):
            continue

        weights = [float(e.weight_kg) for e in entries if e.weight_kg is not None]
        hps = [float(e.hp) for e in entries if e.hp is not None]
        s20s = [float(e.s20) for e in entries if e.s20 is not None]
        # Match the bayesian predictor: relative features (class_indicator,
        # s20_edge, weight_delta) need at least half the field covered or
        # the "average" is a 1–2 horse fluke.
        cov = max(2, int(len(entries) * 0.5))
        field_avg_weight = sum(weights) / len(weights) if len(weights) >= cov else None
        field_avg_hp = sum(hps) / len(hps) if len(hps) >= cov else None
        field_avg_s20 = sum(s20s) / len(s20s) if len(s20s) >= cov else None
        field_size = len(entries)
        # Compute race-level pace density once per race from every
        # horse's last_six string — same for every row in this race.
        pace_density = compute_field_pace_density(
            [parse_last_six(e.last_six) for e in entries]
        )
        reliability_table = reliability_tables_by_date.get(race.date)
        if reliability_table is None:
            reliability_table = precompute_agf_reliability_table(
                session, before_date=race.date,
            )
            reliability_tables_by_date[race.date] = reliability_table
        agf_reliability = lookup_agf_reliability(
            reliability_table, race.race_type, field_size, race.surface,
        )

        for entry in entries:
            # Skip obvious sentinel finish values (DNF / scratched rows that
            # TJK marks with positions way outside the real field size).
            # LightGBM LambdaRank rejects negative labels, so any horse
            # whose recorded finish_position exceeds the field size would
            # otherwise produce rank_score < 0 and kill the train job.
            if entry.finish_position > field_size:
                continue
            trainer_name = entry.horse.trainer if entry.horse else None
            sire_name = entry.horse.sire if entry.horse else None
            dam_name = entry.horse.dam if entry.horse else None
            features = extract_features(
                eid_seconds=parse_eid_to_seconds(entry.eid),
                distance_meters=race.distance_meters,
                last_six_parsed=parse_last_six(entry.last_six),
                weight_kg=float(entry.weight_kg) if entry.weight_kg is not None else None,
                field_avg_weight=field_avg_weight,
                kgs=int(entry.kgs) if entry.kgs is not None else None,
                hp=float(entry.hp) if entry.hp is not None else None,
                field_avg_hp=field_avg_hp,
                s20=float(entry.s20) if entry.s20 is not None else None,
                field_avg_s20=field_avg_s20,
                session=session,
                jockey=entry.jockey,
                trainer=trainer_name,
                horse_id=entry.horse_id,
                gate_number=entry.gate_number,
                surface=race.surface,
                race_date=race.date,
                agf=float(entry.agf) if entry.agf is not None else None,
                field_size=field_size,
                sire=sire_name,
                dam=dam_name,
                track_id=race.track_id,
                equipment=entry.equipment,
                field_pace_density=pace_density,
                agf_reliability=agf_reliability,
                race_entry_id=entry.id,
                race_id_for_signals=race.id,
            )
            # Finish-time target: this horse's actual recorded time in
            # seconds.  Same TJK string format as EID — minutes.seconds.
            # hundredths.  None when the entry has no finish_time row
            # (DNF / scratched / missing data); regression head drops
            # those rows at training.
            finish_seconds = parse_eid_to_seconds(entry.finish_time)

            # EV target: realised return per 1-TL flat bet on this horse.
            # For the winner of a parimutuel ganyan pool, the payout per 1
            # TL bet is ``race.ganyan_payout_tl`` (already net of takeout
            # at TJK).  Net return is therefore ``payout - 1``.  Losers
            # forfeit the stake → return = -1.  When the actual payout
            # row is missing, fall back to the AGF-implied payout
            # ``100 / agf`` so we still get a usable (if noisier) target
            # for early-window races where TJK didn't publish the pool.
            if entry.finish_position == 1:
                if race.ganyan_payout_tl is not None:
                    ev_value = float(race.ganyan_payout_tl) - 1.0
                elif entry.agf is not None and float(entry.agf) > 0:
                    ev_value = (100.0 / float(entry.agf)) - 1.0
                else:
                    ev_value = np.nan
            else:
                ev_value = -1.0

            rows.append({
                GROUP_COLUMN: race.id,
                "race_date": race.date,
                "finish_position": entry.finish_position,
                "rank_score": field_size - entry.finish_position,
                EV_TARGET_COLUMN: ev_value,
                FINISH_TIME_TARGET_COLUMN: (
                    finish_seconds if finish_seconds is not None else np.nan
                ),
                # Engineered
                "speed_figure": features.speed_figure,
                "form_cycle": features.form_cycle,
                "weight_delta": features.weight_delta,
                "rest_fitness": features.rest_fitness,
                "class_indicator": features.class_indicator,
                "jockey_win_rate": features.jockey_win_rate,
                "trainer_win_rate": features.trainer_win_rate,
                "gate_bias": features.gate_bias,
                "surface_affinity": features.surface_affinity,
                "agf_edge": features.agf_edge,
                "sire_win_rate": features.sire_win_rate,
                "sire_surface_rate": features.sire_surface_rate,
                "dam_win_rate": features.dam_win_rate,
                "dam_surface_rate": features.dam_surface_rate,
                "jockey_track_win_rate": features.jockey_track_win_rate,
                "surface_switch": features.surface_switch,
                "distance_delta_m": features.distance_delta_m,
                "equipment_changed": features.equipment_changed,
                "field_pace_density": features.field_pace_density,
                "s20_edge": features.s20_edge,
                "agf_reliability": features.agf_reliability,
                "late_agf_drift": features.late_agf_drift,
                "late_jockey_change": features.late_jockey_change,
                "late_equipment_change": features.late_equipment_change,
                "late_gate_change": features.late_gate_change,
                "tipster_consensus": features.tipster_consensus,
                "jockey_discipline_flag": features.jockey_discipline_flag,
                "days_since_workout": features.days_since_workout,
                "workout_speed_ms": features.workout_speed_ms,
                "n_workouts_recent": features.n_workouts_recent,
                "steward_report_flag": features.steward_report_flag,
                # Raw
                "agf_raw": float(entry.agf) if entry.agf is not None else np.nan,
                "hp_raw": float(entry.hp) if entry.hp is not None else np.nan,
                "weight_kg_raw": (
                    float(entry.weight_kg) if entry.weight_kg is not None else np.nan
                ),
                "kgs_raw": int(entry.kgs) if entry.kgs is not None else np.nan,
                "s20_raw": float(entry.s20) if entry.s20 is not None else np.nan,
                "gate_number": (
                    int(entry.gate_number) if entry.gate_number is not None else np.nan
                ),
                "age": int(entry.horse.age) if entry.horse and entry.horse.age else np.nan,
                # Race-level
                "distance_meters": (
                    int(race.distance_meters) if race.distance_meters else np.nan
                ),
                "field_size": field_size,
                "surface_is_kum": _surface_encode(race.surface),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return TrainingFrame(
            features=pd.DataFrame(columns=FEATURE_COLUMNS),
            target=pd.Series(dtype="int64"),
            ev_target=pd.Series(dtype="float64"),
            finish_time_target=pd.Series(dtype="float64"),
            groups=pd.Series(dtype="int64"),
            race_dates=pd.Series(dtype="object"),
        )

    # Stable ordering by (race_id, finish_position) so groups are contiguous.
    df = df.sort_values([GROUP_COLUMN, "finish_position"]).reset_index(drop=True)

    return TrainingFrame(
        features=df[FEATURE_COLUMNS].astype("float64"),
        target=df[TARGET_COLUMN].astype("int64"),
        ev_target=df[EV_TARGET_COLUMN].astype("float64"),
        finish_time_target=df[FINISH_TIME_TARGET_COLUMN].astype("float64"),
        groups=df[GROUP_COLUMN].astype("int64"),
        race_dates=df["race_date"],
    )


def build_race_frame(session: Session, race_id: int) -> pd.DataFrame:
    """Build an inference-time feature matrix for a single race.

    Returns a DataFrame with FEATURE_COLUMNS + ``horse_id`` so callers
    can zip predictions back to entries.
    """
    race = session.get(Race, race_id)
    if race is None or not race.entries:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["horse_id"])

    entries = list(race.entries)
    weights = [float(e.weight_kg) for e in entries if e.weight_kg is not None]
    hps = [float(e.hp) for e in entries if e.hp is not None]
    s20s = [float(e.s20) for e in entries if e.s20 is not None]
    field_avg_weight = sum(weights) / len(weights) if weights else None
    field_avg_hp = sum(hps) / len(hps) if hps else None
    field_avg_s20 = sum(s20s) / len(s20s) if s20s else None
    field_size = len(entries)
    pace_density = compute_field_pace_density(
        [parse_last_six(e.last_six) for e in entries]
    )
    # Inference-time regime lookup: cutoff at the race's own date so
    # the feature is never computed on post-race information.
    agf_reliability_table = precompute_agf_reliability_table(
        session, before_date=race.date,
    )
    agf_reliability = lookup_agf_reliability(
        agf_reliability_table, race.race_type, field_size, race.surface,
    )

    rows: list[dict] = []
    for entry in entries:
        trainer_name = entry.horse.trainer if entry.horse else None
        sire_name = entry.horse.sire if entry.horse else None
        dam_name = entry.horse.dam if entry.horse else None
        features = extract_features(
            eid_seconds=parse_eid_to_seconds(entry.eid),
            distance_meters=race.distance_meters,
            last_six_parsed=parse_last_six(entry.last_six),
            weight_kg=float(entry.weight_kg) if entry.weight_kg is not None else None,
            field_avg_weight=field_avg_weight,
            kgs=int(entry.kgs) if entry.kgs is not None else None,
            hp=float(entry.hp) if entry.hp is not None else None,
            field_avg_hp=field_avg_hp,
            s20=float(entry.s20) if entry.s20 is not None else None,
            field_avg_s20=field_avg_s20,
            session=session,
            jockey=entry.jockey,
            trainer=trainer_name,
            horse_id=entry.horse_id,
            gate_number=entry.gate_number,
            surface=race.surface,
            race_date=race.date,
            agf=float(entry.agf) if entry.agf is not None else None,
            field_size=field_size,
            sire=sire_name,
            dam=dam_name,
            track_id=race.track_id,
            equipment=entry.equipment,
            field_pace_density=pace_density,
            agf_reliability=agf_reliability,
            race_entry_id=entry.id,
            race_id_for_signals=race.id,
        )
        rows.append({
            "horse_id": entry.horse_id,
            "speed_figure": features.speed_figure,
            "form_cycle": features.form_cycle,
            "weight_delta": features.weight_delta,
            "rest_fitness": features.rest_fitness,
            "class_indicator": features.class_indicator,
            "jockey_win_rate": features.jockey_win_rate,
            "trainer_win_rate": features.trainer_win_rate,
            "gate_bias": features.gate_bias,
            "surface_affinity": features.surface_affinity,
            "agf_edge": features.agf_edge,
            "sire_win_rate": features.sire_win_rate,
            "sire_surface_rate": features.sire_surface_rate,
            "dam_win_rate": features.dam_win_rate,
            "dam_surface_rate": features.dam_surface_rate,
            "jockey_track_win_rate": features.jockey_track_win_rate,
            "surface_switch": features.surface_switch,
            "distance_delta_m": features.distance_delta_m,
            "equipment_changed": features.equipment_changed,
            "field_pace_density": features.field_pace_density,
            "s20_edge": features.s20_edge,
            "agf_reliability": features.agf_reliability,
            "late_agf_drift": features.late_agf_drift,
            "late_jockey_change": features.late_jockey_change,
            "late_equipment_change": features.late_equipment_change,
            "late_gate_change": features.late_gate_change,
            "tipster_consensus": features.tipster_consensus,
            "jockey_discipline_flag": features.jockey_discipline_flag,
            "days_since_workout": features.days_since_workout,
            "workout_speed_ms": features.workout_speed_ms,
            "n_workouts_recent": features.n_workouts_recent,
            "steward_report_flag": features.steward_report_flag,
            "agf_raw": float(entry.agf) if entry.agf is not None else np.nan,
            "hp_raw": float(entry.hp) if entry.hp is not None else np.nan,
            "weight_kg_raw": (
                float(entry.weight_kg) if entry.weight_kg is not None else np.nan
            ),
            "kgs_raw": int(entry.kgs) if entry.kgs is not None else np.nan,
            "s20_raw": float(entry.s20) if entry.s20 is not None else np.nan,
            "gate_number": (
                int(entry.gate_number) if entry.gate_number is not None else np.nan
            ),
            "age": int(entry.horse.age) if entry.horse and entry.horse.age else np.nan,
            "distance_meters": (
                int(race.distance_meters) if race.distance_meters else np.nan
            ),
            "field_size": field_size,
            "surface_is_kum": _surface_encode(race.surface),
        })

    return pd.DataFrame(rows)
