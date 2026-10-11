"""Preserve race-specific attributes and physical draw without inventing history."""
from alembic import op
import sqlalchemy as sa

revision = "f3a4b5c6d7e8"
down_revision = "e2f3a4b5c6d7"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("races", sa.Column("conditions_history", sa.JSON(), nullable=True))
    op.add_column("race_entries", sa.Column("age_at_race", sa.SmallInteger(), nullable=True))
    op.add_column("race_entries", sa.Column("trainer_at_race", sa.String(200), nullable=True))
    op.add_column("race_entries", sa.Column("start_gate", sa.SmallInteger(), nullable=True))
    op.add_column("agf_snapshots", sa.Column("start_gate", sa.SmallInteger(), nullable=True))
    # Existing Horse attributes and program numbers cannot reconstruct past ages,
    # trainers or physical draws. Leave unknown history NULL for a source refresh.


def downgrade():
    op.drop_column("agf_snapshots", "start_gate")
    for column in ("start_gate", "trainer_at_race", "age_at_race"):
        op.drop_column("race_entries", column)
    op.drop_column("races", "conditions_history")
