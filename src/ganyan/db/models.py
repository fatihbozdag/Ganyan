import enum
from datetime import date as date_type, datetime

from sqlalchemy import (
    String, SmallInteger, Integer, Numeric, Date, DateTime, Enum, JSON, Text,
    ForeignKey, UniqueConstraint, Index, func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RaceStatus(enum.Enum):
    scheduled = "scheduled"
    resulted = "resulted"
    cancelled = "cancelled"


class ScrapeStatus(enum.Enum):
    success = "success"
    failed = "failed"
    skipped = "skipped"


class JobStatus(enum.Enum):
    """Outcome of a scheduled job execution."""

    running = "running"
    success = "success"
    failed = "failed"
    missed = "missed"


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    city: Mapped[str | None] = mapped_column(String(100))
    surface_types: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    races: Mapped[list["Race"]] = relationship(back_populates="track")


class Race(Base):
    __tablename__ = "races"
    __table_args__ = (
        UniqueConstraint("track_id", "date", "race_number", name="uq_race_track_date_num"),
        Index("ix_races_date", "date"),
        Index("ix_races_track_date", "track_id", "date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    track_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"))
    date: Mapped[date_type] = mapped_column(Date)
    race_number: Mapped[int] = mapped_column(SmallInteger)
    post_time: Mapped[str | None] = mapped_column(String(5), nullable=True)  # HH:MM
    distance_meters: Mapped[int | None] = mapped_column(Integer)
    surface: Mapped[str | None] = mapped_column(String(50))
    race_type: Mapped[str | None] = mapped_column(String(100))
    horse_type: Mapped[str | None] = mapped_column(String(100))
    weight_rule: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[RaceStatus] = mapped_column(Enum(RaceStatus), default=RaceStatus.scheduled)
    # Last-800m sectional times (seconds) from the TJK results page.
    # Leader is the race winner; runner_up is the 2nd finisher.  Either
    # may be NULL when TJK publishes only one (wire-to-wire wins).
    pace_l800_leader_s: Mapped[float | None] = mapped_column(
        Numeric(6, 2), nullable=True,
    )
    pace_l800_runner_up_s: Mapped[float | None] = mapped_column(
        Numeric(6, 2), nullable=True,
    )
    # Actual parimutuel payouts (TL per 1 TL bet on the winning combo).
    # Only populated for races whose results page has been scraped and
    # whose given pool had a winner.
    ganyan_payout_tl: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True,
    )
    ikili_payout_tl: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True,
    )
    sirali_ikili_payout_tl: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True,
    )
    uclu_payout_tl: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True,
    )
    dortlu_payout_tl: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True,
    )

    track: Mapped["Track"] = relationship(back_populates="races")
    entries: Mapped[list["RaceEntry"]] = relationship(back_populates="race")


class Horse(Base):
    __tablename__ = "horses"
    __table_args__ = (
        Index("ix_horses_tjk_at_id", "tjk_at_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    tjk_at_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    age: Mapped[int | None] = mapped_column(SmallInteger)
    origin: Mapped[str | None] = mapped_column(String(100))
    owner: Mapped[str | None] = mapped_column(String(200))
    trainer: Mapped[str | None] = mapped_column(String(200))
    # Pedigree (populated by the horse-detail crawler, not the race scrape).
    sire: Mapped[str | None] = mapped_column(String(200))
    dam: Mapped[str | None] = mapped_column(String(200))
    birth_date: Mapped[date_type | None] = mapped_column(Date, nullable=True)
    profile_crawled_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True,
    )

    entries: Mapped[list["RaceEntry"]] = relationship(back_populates="horse")


class RaceEntry(Base):
    __tablename__ = "race_entries"
    __table_args__ = (
        UniqueConstraint("race_id", "horse_id", name="uq_race_entries_race_horse"),
        Index("ix_race_entries_race_id", "race_id"),
        Index("ix_race_entries_horse_id", "horse_id"),
        Index("ix_race_entries_jockey", "jockey"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id"))
    horse_id: Mapped[int] = mapped_column(ForeignKey("horses.id"))
    gate_number: Mapped[int | None] = mapped_column(SmallInteger)
    jockey: Mapped[str | None] = mapped_column(String(200))
    weight_kg: Mapped[float | None] = mapped_column(Numeric(4, 1))
    hp: Mapped[float | None] = mapped_column(Numeric(5, 1))
    kgs: Mapped[int | None] = mapped_column(SmallInteger)
    s20: Mapped[float | None] = mapped_column(Numeric(5, 2))
    eid: Mapped[str | None] = mapped_column(String(20))
    gny: Mapped[float | None] = mapped_column(Numeric(5, 2))
    agf: Mapped[float | None] = mapped_column(Numeric(5, 2))
    last_six: Mapped[str | None] = mapped_column(String(50))
    # Equipment (takı) codes this horse wears in this race.  Space-
    # separated 1-3-letter codes (KG, DB, SK, K, AG, Y, NL, ...).  First-
    # time equipment is a classic upset signal in Turkish handicapping.
    equipment: Mapped[str | None] = mapped_column(String(100))
    finish_position: Mapped[int | None] = mapped_column(SmallInteger)
    finish_time: Mapped[str | None] = mapped_column(String(20))
    performance_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    predicted_probability: Mapped[float | None] = mapped_column(Numeric(5, 2))

    race: Mapped["Race"] = relationship(back_populates="entries")
    horse: Mapped["Horse"] = relationship(back_populates="entries")


class ScrapeLog(Base):
    __tablename__ = "scrape_log"
    __table_args__ = (
        Index("ix_scrape_log_date_track", "date", "track"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date_type] = mapped_column(Date)
    track: Mapped[str] = mapped_column(String(100))
    status: Mapped[ScrapeStatus] = mapped_column(Enum(ScrapeStatus))
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class JobRun(Base):
    """One execution of a scheduled (APScheduler) job.

    Populated by the event listener in :mod:`ganyan.scheduler`.  The
    ``/ops`` dashboard and the macOS notifier both read from this
    table.  Keep every row forever — history is cheap and it's useful
    for seeing *when* a previously-working pipeline broke.
    """

    __tablename__ = "job_runs"
    __table_args__ = (
        Index("ix_job_runs_job_id", "job_id"),
        Index("ix_job_runs_started_at", "started_at"),
        Index("ix_job_runs_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(String(100))
    started_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True,
    )
    # Stored as plain String; JobStatus enum is app-side validation only.
    status: Mapped[str] = mapped_column(String(16))
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Optional short human-readable summary ("17 races stored", etc.).
    output_summary: Mapped[str | None] = mapped_column(
        String(500), nullable=True,
    )


class Pick(Base):
    """A strategy-level bet recommendation — the thing you would place.

    Distinct from :class:`Prediction` (which stores per-horse win
    probabilities from a given model).  A Pick fixes which pool, which
    ordered combination, and which strategy label ("uclu_top1", etc.)
    got recommended for a given race.  After the race resolves, a
    grader fills in ``hit``, ``payout_tl``, and ``net_tl`` so we can
    track each strategy's real-world running ROI over time.
    """

    __tablename__ = "picks"
    __table_args__ = (
        Index("ix_picks_race_id", "race_id"),
        Index("ix_picks_strategy", "strategy"),
        Index("ix_picks_generated_at", "generated_at"),
        Index("ix_picks_graded", "graded"),
        UniqueConstraint(
            "race_id", "strategy", name="uq_picks_race_strategy",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    race_id: Mapped[int] = mapped_column(ForeignKey("races.id"))
    strategy: Mapped[str] = mapped_column(String(50))
    # JSON list of horse_ids in the picked order (for Üçlü this is the
    # exact 1-2-3 ordering; for box-6 it's still the base top-3).
    combination: Mapped[list] = mapped_column(JSON)
    # Human-readable horse names at the time the pick was generated.
    combination_names: Mapped[list | None] = mapped_column(JSON, nullable=True)
    stake_tl: Mapped[float] = mapped_column(Numeric(10, 2))
    ticket_count: Mapped[int] = mapped_column(SmallInteger, default=1)
    model_prob_pct: Mapped[float | None] = mapped_column(
        Numeric(6, 3), nullable=True,
    )
    generated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )

    # Graded fields — filled by grade_picks() after the race resolves.
    graded: Mapped[bool] = mapped_column(default=False)
    hit: Mapped[bool | None] = mapped_column(nullable=True)
    payout_tl: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True,
    )
    net_tl: Mapped[float | None] = mapped_column(
        Numeric(12, 2), nullable=True,
    )
    graded_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True,
    )


class Prediction(Base):
    """Audit history of predictions.

    Unlike ``RaceEntry.predicted_probability`` (a single slot that is
    overwritten on every re-run), this table keeps every prediction the
    system has made, tagged with a model version and timestamp.  Lets us
    compare model variants on held-out races and trace accuracy over time.
    """

    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_predictions_race_entry_id", "race_entry_id"),
        Index("ix_predictions_model_version", "model_version"),
        Index("ix_predictions_predicted_at", "predicted_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    race_entry_id: Mapped[int] = mapped_column(ForeignKey("race_entries.id"))
    model_version: Mapped[str] = mapped_column(String(50))
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False,
    )
    probability: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Numeric(4, 3), nullable=True)
    factors: Mapped[dict | None] = mapped_column(JSON, nullable=True)
