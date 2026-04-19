"""add horse pedigree columns (tjk_at_id, sire, dam, birth_date)

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-04-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("horses", sa.Column("tjk_at_id", sa.Integer(), nullable=True))
    op.add_column("horses", sa.Column("sire", sa.String(length=200), nullable=True))
    op.add_column("horses", sa.Column("dam", sa.String(length=200), nullable=True))
    op.add_column("horses", sa.Column("birth_date", sa.Date(), nullable=True))
    op.add_column(
        "horses",
        sa.Column("profile_crawled_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_horses_tjk_at_id", "horses", ["tjk_at_id"])


def downgrade() -> None:
    op.drop_index("ix_horses_tjk_at_id", table_name="horses")
    op.drop_column("horses", "profile_crawled_at")
    op.drop_column("horses", "birth_date")
    op.drop_column("horses", "dam")
    op.drop_column("horses", "sire")
    op.drop_column("horses", "tjk_at_id")
