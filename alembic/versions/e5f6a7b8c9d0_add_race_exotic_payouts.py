"""add exotic pool payout columns to races

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-04-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_COLUMNS = [
    "ganyan_payout_tl",
    "ikili_payout_tl",
    "sirali_ikili_payout_tl",
    "uclu_payout_tl",
    "dortlu_payout_tl",
]


def upgrade() -> None:
    for col in _COLUMNS:
        op.add_column("races", sa.Column(col, sa.Numeric(12, 2), nullable=True))


def downgrade() -> None:
    for col in reversed(_COLUMNS):
        op.drop_column("races", col)
