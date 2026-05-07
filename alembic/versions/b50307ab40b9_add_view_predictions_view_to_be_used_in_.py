"""add view_predictions view to be used in predictions text output

Revision ID: b50307ab40b9
Revises: 1f21042191c0
Create Date: 2026-05-07 10:02:03.712490

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b50307ab40b9'
down_revision: Union[str, Sequence[str], None] = '1f21042191c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS idx_races_date;
        DROP INDEX IF EXISTS idx_race_entries_race_id;
        DROP INDEX IF EXISTS idx_predictions_race_entry_id;
        DROP INDEX IF EXISTS idx_races_track_id;
        DROP INDEX IF EXISTS idx_race_entries_horse_id;

        CREATE INDEX idx_races_date ON races (date);
        CREATE INDEX idx_race_entries_race_id ON race_entries (race_id);
        CREATE INDEX idx_predictions_race_entry_id ON predictions (race_entry_id);
        CREATE INDEX idx_races_track_id ON races (track_id);
        CREATE INDEX idx_race_entries_horse_id ON race_entries (horse_id);

        DROP VIEW IF EXISTS view_predictions;
        CREATE VIEW
            view_predictions AS
        SELECT
            predictions.id as id,
            races.id as race_id,
            races.date as race_date,
            races.post_time as race_time,
            races.race_number as race_number,
            races.race_type as race_type,
            races.horse_type as horse_type,
            horses.id as horse_id,
            horses.name as horse_name,
            race_entries.jockey as jockey_name,
            races.horse_type as horse_type,
            concat (
                tracks.name,
                ' (',
                races.distance_meters,
                ' m, ',
                races.surface,
                ')'
            ) as track_name,
            race_entries.gate_number as gate_number,
            ROUND(predictions.probability, 2) as probability,
            predictions.race_entry_id as race_entry_id,
            predictions.model_version as model_version,
            predictions.predicted_at as predicted_at,
            predictions.confidence as confidence,
            predictions.factors as factors
        FROM
            races
            JOIN race_entries ON races.id = race_entries.race_id
            JOIN predictions ON predictions.race_entry_id = race_entries.id
            JOIN tracks ON races.track_id = tracks.id
            JOIN horses ON race_entries.horse_id = horses.id
        GROUP BY
            horses.id
        ORDER BY
            races.race_number,
            predictions.probability DESC;
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("""
        DROP VIEW view_predictions
    """)
