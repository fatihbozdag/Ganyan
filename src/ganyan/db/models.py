import enum
from datetime import date as date_type, datetime

from sqlalchemy import (
    String, SmallInteger, Integer, Numeric, Date, DateTime, Enum, JSON,
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
    distance_meters: Mapped[int | None] = mapped_column(Integer)
    surface: Mapped[str | None] = mapped_column(String(50))
    race_type: Mapped[str | None] = mapped_column(String(100))
    horse_type: Mapped[str | None] = mapped_column(String(100))
    weight_rule: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[RaceStatus] = mapped_column(Enum(RaceStatus), default=RaceStatus.scheduled)

    track: Mapped["Track"] = relationship(back_populates="races")
    entries: Mapped[list["RaceEntry"]] = relationship(back_populates="race")


class Horse(Base):
    __tablename__ = "horses"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    age: Mapped[int | None] = mapped_column(SmallInteger)
    origin: Mapped[str | None] = mapped_column(String(100))
    owner: Mapped[str | None] = mapped_column(String(200))
    trainer: Mapped[str | None] = mapped_column(String(200))

    entries: Mapped[list["RaceEntry"]] = relationship(back_populates="horse")


class RaceEntry(Base):
    __tablename__ = "race_entries"
    __table_args__ = (
        Index("ix_race_entries_race_id", "race_id"),
        Index("ix_race_entries_horse_id", "horse_id"),
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
    finish_position: Mapped[int | None] = mapped_column(SmallInteger)
    finish_time: Mapped[str | None] = mapped_column(String(20))
    performance_score: Mapped[float | None] = mapped_column(Numeric(5, 2))
    predicted_probability: Mapped[float | None] = mapped_column(Numeric(5, 2))

    race: Mapped["Race"] = relationship(back_populates="entries")
    horse: Mapped["Horse"] = relationship(back_populates="entries")


class ScrapeLog(Base):
    __tablename__ = "scrape_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[date_type] = mapped_column(Date)
    track: Mapped[str] = mapped_column(String(100))
    status: Mapped[ScrapeStatus] = mapped_column(Enum(ScrapeStatus))
    scraped_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
