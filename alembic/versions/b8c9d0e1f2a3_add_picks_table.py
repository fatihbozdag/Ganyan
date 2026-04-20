"""add picks table for strategy-level bet tracking

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-04-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b8c9d0e1f2a3"
down_revision: Union[str, Sequence[str], None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "picks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "race_id", sa.Integer(),
            sa.ForeignKey("races.id"), nullable=False,
        ),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("combination", sa.JSON(), nullable=False),
        sa.Column("combination_names", sa.JSON(), nullable=True),
        sa.Column("stake_tl", sa.Numeric(10, 2), nullable=False),
        sa.Column("ticket_count", sa.SmallInteger(), nullable=False, server_default="1"),
        sa.Column("model_prob_pct", sa.Numeric(6, 3), nullable=True),
        sa.Column(
            "generated_at", sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False,
        ),
        sa.Column("graded", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("hit", sa.Boolean(), nullable=True),
        sa.Column("payout_tl", sa.Numeric(12, 2), nullable=True),
        sa.Column("net_tl", sa.Numeric(12, 2), nullable=True),
        sa.Column("graded_at", sa.DateTime(), nullable=True),
    )
    op.create_unique_constraint(
        "uq_picks_race_strategy", "picks", ["race_id", "strategy"],
    )
    op.create_index("ix_picks_race_id", "picks", ["race_id"])
    op.create_index("ix_picks_strategy", "picks", ["strategy"])
    op.create_index("ix_picks_generated_at", "picks", ["generated_at"])
    op.create_index("ix_picks_graded", "picks", ["graded"])


def downgrade() -> None:
    op.drop_index("ix_picks_graded", table_name="picks")
    op.drop_index("ix_picks_generated_at", table_name="picks")
    op.drop_index("ix_picks_strategy", table_name="picks")
    op.drop_index("ix_picks_race_id", table_name="picks")
    op.drop_constraint("uq_picks_race_strategy", "picks", type_="unique")
    op.drop_table("picks")
