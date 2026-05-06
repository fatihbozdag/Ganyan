"""add unique constraints

Revision ID: 1f21042191c0
Revises: 001_initial
Create Date: 2026-05-05 14:18:20.978223

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1f21042191c0'
down_revision: Union[str, Sequence[str], None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Use batch mode for SQLite support
    with op.batch_alter_table("races") as batch_op:
        batch_op.create_unique_constraint("uq_race_track_date_num", ["track_id", "date", "race_number"])
    with op.batch_alter_table("race_entries", schema=None) as batch_op:
        batch_op.create_unique_constraint("uq_race_entries_race_horse", ["race_id", "horse_id"])
    with op.batch_alter_table("picks", schema=None) as batch_op:
        batch_op.create_unique_constraint("uq_picks_race_strategy", ["race_id", "strategy"])
    with op.batch_alter_table("regime_daily", schema=None) as batch_op:
        batch_op.create_unique_constraint("uq_regime_daily_date_strategy", ["snapshot_date", "strategy"])
    with op.batch_alter_table("multi_race_pools", schema=None) as batch_op:
        batch_op.create_unique_constraint("uq_multi_race_pool_date_track_type_idx", ["date", "track_id", "pool_type", "pool_index"])
    with op.batch_alter_table("multi_race_picks", schema=None) as batch_op:
        batch_op.create_unique_constraint("uq_multi_race_pick_date_track_type_idx_strat", ["date", "track_id", "pool_type", "pool_index", "strategy"])


def downgrade() -> None:
    uniques = {
        "races": "uq_race_track_date_num",
        "race_entries": "uq_race_entries_race_horse",
        "picks": "uq_picks_race_strategy",
        "regime_daily": "uq_regime_daily_date_strategy",
        "multi_race_pools": "uq_multi_race_pool_date_track_type_idx",
        "multi_race_picks": "uq_multi_race_pick_date_track_type_idx_strat"
    }
    
    for table, unique in uniques.items():        
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_constraint(unique, type_='unique')
