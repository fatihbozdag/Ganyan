"""Base classes for external-data plugins."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date as date_type, datetime
from typing import ClassVar

from sqlalchemy.orm import Session
from ganyan.time import utcnow


logger = logging.getLogger(__name__)


@dataclass
class ExternalSignalRow:
    """One row to be written to ``external_signals``.

    The plugin's job is to produce these.  ``race_id`` /
    ``race_entry_id`` may be ``None`` when the plugin can't resolve
    the TJK identifier inline; a downstream resolver pass can join
    on the ``payload`` keys to fill them in.
    """

    source_name: str
    signal_type: str
    race_id: int | None = None
    race_entry_id: int | None = None
    value: float | None = None
    payload: dict | None = None
    captured_at: datetime | None = None


class ExternalSource(ABC):
    """Abstract third-party data plugin.

    Subclasses set ``source_name`` and implement ``fetch_for_date``.
    The framework handles registry, CLI dispatch, and persistence;
    plugins focus on parsing.
    """

    # Stable string used as the ``source_name`` in stored rows and the
    # CLI lookup key.  Override per subclass.
    source_name: ClassVar[str] = "abstract"

    # Stable identifiers each plugin emits.  Used by feature extraction
    # to query a specific signal type from this source.  Subclasses
    # should override.
    signal_types: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def fetch_for_date(
        self, session: Session, target_date: date_type,
    ) -> list[ExternalSignalRow]:
        """Return signals collected for ``target_date``.

        ``session`` is provided so the plugin can resolve TJK-side
        identifiers (race_id, race_entry_id) when its own raw output
        carries (track, race_number, horse_number).
        """
        raise NotImplementedError


# Module-level registry.  Plugins register themselves on import (see
# ``__init__.py``) so the CLI can discover sources by name.
REGISTRY: dict[str, type[ExternalSource]] = {}


def register_source(cls: type[ExternalSource]) -> type[ExternalSource]:
    """Register an :class:`ExternalSource` subclass by ``source_name``.

    Idempotent: re-registering the same name overwrites the previous
    binding (useful in tests).  Returns ``cls`` so the function can be
    used as a decorator.
    """
    name = getattr(cls, "source_name", None)
    if not name or name == "abstract":
        raise ValueError(
            f"ExternalSource subclass {cls.__name__} must set source_name",
        )
    REGISTRY[name] = cls
    return cls


def persist_signals(
    session: Session, rows: list[ExternalSignalRow],
) -> int:
    """Persist a batch of signals to ``external_signals``.

    Returns the number of rows inserted.  Caller is responsible for
    committing; this function only flushes.
    """
    from ganyan.db.models import ExternalSignal

    if not rows:
        return 0
    objs = [
        ExternalSignal(
            source_name=r.source_name,
            signal_type=r.signal_type,
            race_id=r.race_id,
            race_entry_id=r.race_entry_id,
            value=r.value,
            payload=r.payload,
            captured_at=r.captured_at or utcnow(),
        )
        for r in rows
    ]
    session.add_all(objs)
    session.flush()
    return len(objs)
