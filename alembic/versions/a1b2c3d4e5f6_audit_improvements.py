"""audit improvements: predictions table, scrape log error column, entry unique constraint

Revision ID: a1b2c3d4e5f6
Revises: 0c6e46ca4eb5
Create Date: 2026-04-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "0c6e46ca4eb5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- race_entries: enforce one entry per (race, horse) ----------------
    # Deduplicate any pre-existing rows before adding the constraint.
    op.execute(
        """
        DELETE FROM race_entries
        WHERE id NOT IN (
            SELECT MIN(id) FROM race_entries
            GROUP BY race_id, horse_id
        )
        """
    )
    op.create_unique_constraint(
        "uq_race_entries_race_horse", "race_entries", ["race_id", "horse_id"],
    )
    op.create_index(
        "ix_race_entries_jockey", "race_entries", ["jockey"],
    )

    # --- scrape_log: error capture + per-(date,track) lookup index --------
    op.add_column(
        "scrape_log",
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_scrape_log_date_track", "scrape_log", ["date", "track"],
    )

    # --- predictions audit table ------------------------------------------
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "race_entry_id", sa.Integer(),
            sa.ForeignKey("race_entries.id"), nullable=False,
        ),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column(
            "predicted_at", sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False,
        ),
        sa.Column("probability", sa.Numeric(6, 3), nullable=False),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=True),
        sa.Column("factors", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_predictions_race_entry_id", "predictions", ["race_entry_id"],
    )
    op.create_index(
        "ix_predictions_model_version", "predictions", ["model_version"],
    )
    op.create_index(
        "ix_predictions_predicted_at", "predictions", ["predicted_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_predicted_at", table_name="predictions")
    op.drop_index("ix_predictions_model_version", table_name="predictions")
    op.drop_index("ix_predictions_race_entry_id", table_name="predictions")
    op.drop_table("predictions")

    op.drop_index("ix_scrape_log_date_track", table_name="scrape_log")
    op.drop_column("scrape_log", "error_message")

    op.drop_index("ix_race_entries_jockey", table_name="race_entries")
    op.drop_constraint(
        "uq_race_entries_race_horse", "race_entries", type_="unique",
    )
