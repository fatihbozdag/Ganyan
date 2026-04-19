"""add race pace_l800 columns (Son 800 sectional times)

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-04-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "races",
        sa.Column("pace_l800_leader_s", sa.Numeric(6, 2), nullable=True),
    )
    op.add_column(
        "races",
        sa.Column("pace_l800_runner_up_s", sa.Numeric(6, 2), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("races", "pace_l800_runner_up_s")
    op.drop_column("races", "pace_l800_leader_s")
