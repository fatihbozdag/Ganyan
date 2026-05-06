"""initial schema

Revision ID: 001_initial
Revises:
Create Date: 2026-05-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tracks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=100), nullable=False, unique=True),
        sa.Column("city", sa.String(length=100), nullable=True),
        sa.Column("surface_types", sa.JSON(), nullable=True),
    )

    op.create_table(
        "races",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("track_id", sa.Integer(), sa.ForeignKey("tracks.id"), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("race_number", sa.SmallInteger(), nullable=False),
        sa.Column("post_time", sa.String(length=5), nullable=True),
        sa.Column("distance_meters", sa.Integer(), nullable=True),
        sa.Column("surface", sa.String(length=50), nullable=True),
        sa.Column("race_type", sa.String(length=100), nullable=True),
        sa.Column("horse_type", sa.String(length=100), nullable=True),
        sa.Column("weight_rule", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="scheduled"),
        sa.Column("pace_l800_leader_s", sa.Numeric(6, 2), nullable=True),
        sa.Column("pace_l800_runner_up_s", sa.Numeric(6, 2), nullable=True),
        sa.Column("ganyan_payout_tl", sa.Numeric(12, 2), nullable=True),
        sa.Column("ikili_payout_tl", sa.Numeric(12, 2), nullable=True),
        sa.Column("sirali_ikili_payout_tl", sa.Numeric(12, 2), nullable=True),
        sa.Column("uclu_payout_tl", sa.Numeric(12, 2), nullable=True),
        sa.Column("dortlu_payout_tl", sa.Numeric(12, 2), nullable=True),
    )
    op.create_index("ix_races_date", "races", ["date"])
    op.create_index("ix_races_track_date", "races", ["track_id", "date"])

    op.create_table(
        "horses",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("tjk_at_id", sa.Integer(), nullable=True),
        sa.Column("age", sa.SmallInteger(), nullable=True),
        sa.Column("origin", sa.String(length=100), nullable=True),
        sa.Column("owner", sa.String(length=200), nullable=True),
        sa.Column("trainer", sa.String(length=200), nullable=True),
        sa.Column("sire", sa.String(length=200), nullable=True),
        sa.Column("dam", sa.String(length=200), nullable=True),
        sa.Column("birth_date", sa.Date(), nullable=True),
        sa.Column("profile_crawled_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_horses_tjk_at_id", "horses", ["tjk_at_id"])
    op.create_index("ix_horses_name", "horses", ["name"])

    op.create_table(
        "race_entries",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("race_id", sa.Integer(), sa.ForeignKey("races.id"), nullable=False),
        sa.Column("horse_id", sa.Integer(), sa.ForeignKey("horses.id"), nullable=False),
        sa.Column("gate_number", sa.SmallInteger(), nullable=True),
        sa.Column("jockey", sa.String(length=200), nullable=True),
        sa.Column("weight_kg", sa.Numeric(4, 1), nullable=True),
        sa.Column("hp", sa.Numeric(5, 1), nullable=True),
        sa.Column("kgs", sa.SmallInteger(), nullable=True),
        sa.Column("s20", sa.Numeric(5, 2), nullable=True),
        sa.Column("eid", sa.String(length=20), nullable=True),
        sa.Column("gny", sa.Numeric(5, 2), nullable=True),
        sa.Column("agf", sa.Numeric(5, 2), nullable=True),
        sa.Column("last_six", sa.String(length=50), nullable=True),
        sa.Column("equipment", sa.String(length=100), nullable=True),
        sa.Column("finish_position", sa.SmallInteger(), nullable=True),
        sa.Column("finish_time", sa.String(length=20), nullable=True),
        sa.Column("performance_score", sa.Numeric(5, 2), nullable=True),
        sa.Column("predicted_probability", sa.Numeric(5, 2), nullable=True),
        sa.Column("scratched", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.create_index("ix_race_entries_race_id", "race_entries", ["race_id"])
    op.create_index("ix_race_entries_horse_id", "race_entries", ["horse_id"])
    op.create_index("ix_race_entries_jockey", "race_entries", ["jockey"])

    op.create_table(
        "agf_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("race_entry_id", sa.Integer(), sa.ForeignKey("race_entries.id", ondelete="CASCADE"), nullable=False),
        sa.Column("taken_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("agf", sa.Numeric(5, 2), nullable=False),
        sa.Column("jockey", sa.String(length=200), nullable=True),
        sa.Column("equipment", sa.String(length=100), nullable=True),
        sa.Column("gate_number", sa.SmallInteger(), nullable=True),
    )
    op.create_index("ix_agf_snapshots_race_entry_id", "agf_snapshots", ["race_entry_id"])
    op.create_index("ix_agf_snapshots_taken_at", "agf_snapshots", ["taken_at"])

    op.create_table(
        "external_signals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source_name", sa.String(length=50), nullable=False),
        sa.Column("signal_type", sa.String(length=50), nullable=False),
        sa.Column("race_id", sa.Integer(), sa.ForeignKey("races.id", ondelete="CASCADE"), nullable=True),
        sa.Column("race_entry_id", sa.Integer(), sa.ForeignKey("race_entries.id", ondelete="CASCADE"), nullable=True),
        sa.Column("value", sa.Numeric(10, 3), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("captured_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_external_signals_source_type", "external_signals", ["source_name", "signal_type"])
    op.create_index("ix_external_signals_race_entry_id", "external_signals", ["race_entry_id"])
    op.create_index("ix_external_signals_race_id", "external_signals", ["race_id"])
    op.create_index("ix_external_signals_captured_at", "external_signals", ["captured_at"])

    op.create_table(
        "scrape_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("track", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("scraped_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_scrape_log_date_track", "scrape_log", ["date", "track"])

    op.create_table(
        "job_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.String(length=100), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("output_summary", sa.String(length=500), nullable=True),
    )
    op.create_index("ix_job_runs_job_id", "job_runs", ["job_id"])
    op.create_index("ix_job_runs_started_at", "job_runs", ["started_at"])
    op.create_index("ix_job_runs_status", "job_runs", ["status"])

    op.create_table(
        "picks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("race_id", sa.Integer(), sa.ForeignKey("races.id"), nullable=False),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("combination", sa.JSON(), nullable=False),
        sa.Column("combination_names", sa.JSON(), nullable=True),
        sa.Column("stake_tl", sa.Numeric(10, 2), nullable=False),
        sa.Column("ticket_count", sa.SmallInteger(), nullable=False, server_default="1"),
        sa.Column("model_prob_pct", sa.Numeric(6, 3), nullable=True),
        sa.Column("generated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("graded", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("hit", sa.Boolean(), nullable=True),
        sa.Column("payout_tl", sa.Numeric(12, 2), nullable=True),
        sa.Column("net_tl", sa.Numeric(12, 2), nullable=True),
        sa.Column("graded_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_picks_race_id", "picks", ["race_id"])
    op.create_index("ix_picks_strategy", "picks", ["strategy"])
    op.create_index("ix_picks_generated_at", "picks", ["generated_at"])
    op.create_index("ix_picks_graded", "picks", ["graded"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("race_entry_id", sa.Integer(), sa.ForeignKey("race_entries.id"), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("predicted_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("probability", sa.Numeric(6, 3), nullable=False),
        sa.Column("confidence", sa.Numeric(4, 3), nullable=True),
        sa.Column("factors", sa.JSON(), nullable=True),
    )
    op.create_index("ix_predictions_race_entry_id", "predictions", ["race_entry_id"])
    op.create_index("ix_predictions_model_version", "predictions", ["model_version"])
    op.create_index("ix_predictions_predicted_at", "predictions", ["predicted_at"])

    op.create_table(
        "regime_daily",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("snapshot_date", sa.Date(), nullable=False),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("n_winning", sa.Integer(), nullable=False),
        sa.Column("mean_payout_tl", sa.Numeric(12, 2), nullable=True),
        sa.Column("mean_pool_proxy_tl", sa.Numeric(14, 2), nullable=True),
        sa.Column("implied_takeout", sa.Numeric(6, 4), nullable=True),
        sa.Column("realized_vs_expected", sa.Numeric(6, 4), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_regime_daily_snapshot_date", "regime_daily", ["snapshot_date"])
    op.create_index("ix_regime_daily_strategy", "regime_daily", ["strategy"])

    op.create_table(
        "multi_race_pools",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("track_id", sa.Integer(), sa.ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pool_type", sa.String(length=10), nullable=False),
        sa.Column("pool_index", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("start_race_no", sa.Integer(), nullable=True),
        sa.Column("end_race_no", sa.Integer(), nullable=True),
        sa.Column("winning_combo", sa.String(length=200), nullable=True),
        sa.Column("payout_tl", sa.Numeric(14, 2), nullable=True),
        sa.Column("captured_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_multi_race_pools_date_track", "multi_race_pools", ["date", "track_id"])

    op.create_table(
        "multi_race_picks",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("track_id", sa.Integer(), sa.ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pool_type", sa.String(length=10), nullable=False),
        sa.Column("pool_index", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("start_race_no", sa.Integer(), nullable=False),
        sa.Column("end_race_no", sa.Integer(), nullable=False),
        sa.Column("kept_horses_per_leg", sa.JSON(), nullable=False),
        sa.Column("total_tickets", sa.Integer(), nullable=False),
        sa.Column("ticket_unit_tl", sa.Numeric(10, 2), nullable=False),
        sa.Column("stake_tl", sa.Numeric(10, 2), nullable=False),
        sa.Column("conviction_per_leg", sa.JSON(), nullable=True),
        sa.Column("generated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("graded", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("hit", sa.Boolean(), nullable=True),
        sa.Column("payout_tl", sa.Numeric(14, 2), nullable=True),
        sa.Column("net_tl", sa.Numeric(14, 2), nullable=True),
        sa.Column("graded_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_multi_race_picks_date_track", "multi_race_picks", ["date", "track_id"])
    op.create_index("ix_multi_race_picks_graded", "multi_race_picks", ["graded"])

def downgrade() -> None:
    op.drop_table("multi_race_picks")
    op.drop_table("multi_race_pools")
    op.drop_table("regime_daily")
    op.drop_table("predictions")
    op.drop_table("picks")
    op.drop_table("job_runs")
    op.drop_table("scrape_log")
    op.drop_table("external_signals")
    op.drop_table("agf_snapshots")
    op.drop_table("race_entries")
    op.drop_table("horses")
    op.drop_table("races")
    op.drop_table("tracks")