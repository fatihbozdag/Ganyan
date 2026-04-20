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
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from ganyan.db.models import Race, RaceEntry, RaceStatus
from ganyan.predictor.features import compute_field_pace_density, extract_features
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
    # Domain-derived signals ("sürpriz at" features).
    "surface_switch",
    "distance_delta_m",
    "equipment_changed",
    "apprentice_jockey",
    "field_pace_density",
    # Raw values — give the tree room to learn non-linear effects.
    "agf_raw",
    "hp_raw",
    "weight_kg_raw",
    "kgs_raw",
    "gate_number",
    "age",
    # Race-level context.
    "distance_meters",
    "field_size",
    # Surface encoded as 1 for kum, 0 for çim, -1 for unknown/other.
    "surface_is_kum",
]

TARGET_COLUMN = "rank_score"
GROUP_COLUMN = "race_id"


@dataclass
class TrainingFrame:
    """Container returned by :func:`build_training_frame`."""

    features: pd.DataFrame  # shape (n_rows, len(FEATURE_COLUMNS))
    target: pd.Series  # rank score per row
    groups: pd.Series  # race_id per row (used for LGBM grouping)
    race_dates: pd.Series  # race.date per row (used for temporal split)

    @property
    def race_ids_ordered(self) -> np.ndarray:
        """Unique race_ids in the order they appear (for group sizes)."""
        return self.groups.drop_duplicates().to_numpy()

    def group_sizes(self) -> np.ndarray:
        """Contiguous group sizes as required by LightGBM."""
        return self.groups.groupby(self.groups, sort=False).size().to_numpy()


def _surface_encode(surface: str | None) -> int:
    if surface is None:
        return -1
    s = surface.lower()
    if s.startswith("kum"):
        return 1
    if s.startswith("çim") or s.startswith("cim"):
        return 0
    return -1


def build_training_frame(
    session: Session,
    *,
    from_date: date_type | None = None,
    to_date: date_type | None = None,
    require_agf: bool = True,
    min_field_size: int = 3,
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

    rows: list[dict] = []
    for race in q.order_by(Race.date.asc(), Race.race_number.asc()).all():
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
        field_avg_weight = sum(weights) / len(weights) if weights else None
        field_avg_hp = sum(hps) / len(hps) if hps else None
        field_size = len(entries)
        # Compute race-level pace density once per race from every
        # horse's last_six string — same for every row in this race.
        pace_density = compute_field_pace_density(
            [parse_last_six(e.last_six) for e in entries]
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
            features = extract_features(
                eid_seconds=parse_eid_to_seconds(entry.eid),
                distance_meters=race.distance_meters,
                last_six_parsed=parse_last_six(entry.last_six),
                weight_kg=float(entry.weight_kg) if entry.weight_kg is not None else None,
                field_avg_weight=field_avg_weight,
                kgs=int(entry.kgs) if entry.kgs is not None else None,
                hp=float(entry.hp) if entry.hp is not None else None,
                field_avg_hp=field_avg_hp,
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
                equipment=entry.equipment,
                field_pace_density=pace_density,
            )
            rows.append({
                GROUP_COLUMN: race.id,
                "race_date": race.date,
                "finish_position": entry.finish_position,
                "rank_score": field_size - entry.finish_position,
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
                "surface_switch": features.surface_switch,
                "distance_delta_m": features.distance_delta_m,
                "equipment_changed": features.equipment_changed,
                "apprentice_jockey": features.apprentice_jockey,
                "field_pace_density": features.field_pace_density,
                # Raw
                "agf_raw": float(entry.agf) if entry.agf is not None else np.nan,
                "hp_raw": float(entry.hp) if entry.hp is not None else np.nan,
                "weight_kg_raw": (
                    float(entry.weight_kg) if entry.weight_kg is not None else np.nan
                ),
                "kgs_raw": int(entry.kgs) if entry.kgs is not None else np.nan,
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
            groups=pd.Series(dtype="int64"),
            race_dates=pd.Series(dtype="object"),
        )

    # Stable ordering by (race_id, finish_position) so groups are contiguous.
    df = df.sort_values([GROUP_COLUMN, "finish_position"]).reset_index(drop=True)

    return TrainingFrame(
        features=df[FEATURE_COLUMNS].astype("float64"),
        target=df[TARGET_COLUMN].astype("int64"),
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
    field_avg_weight = sum(weights) / len(weights) if weights else None
    field_avg_hp = sum(hps) / len(hps) if hps else None
    field_size = len(entries)
    pace_density = compute_field_pace_density(
        [parse_last_six(e.last_six) for e in entries]
    )

    rows: list[dict] = []
    for entry in entries:
        trainer_name = entry.horse.trainer if entry.horse else None
        sire_name = entry.horse.sire if entry.horse else None
        features = extract_features(
            eid_seconds=parse_eid_to_seconds(entry.eid),
            distance_meters=race.distance_meters,
            last_six_parsed=parse_last_six(entry.last_six),
            weight_kg=float(entry.weight_kg) if entry.weight_kg is not None else None,
            field_avg_weight=field_avg_weight,
            kgs=int(entry.kgs) if entry.kgs is not None else None,
            hp=float(entry.hp) if entry.hp is not None else None,
            field_avg_hp=field_avg_hp,
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
            equipment=entry.equipment,
            field_pace_density=pace_density,
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
            "surface_switch": features.surface_switch,
            "distance_delta_m": features.distance_delta_m,
            "equipment_changed": features.equipment_changed,
            "apprentice_jockey": features.apprentice_jockey,
            "field_pace_density": features.field_pace_density,
            "agf_raw": float(entry.agf) if entry.agf is not None else np.nan,
            "hp_raw": float(entry.hp) if entry.hp is not None else np.nan,
            "weight_kg_raw": (
                float(entry.weight_kg) if entry.weight_kg is not None else np.nan
            ),
            "kgs_raw": int(entry.kgs) if entry.kgs is not None else np.nan,
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
