"""Single-file halt mechanism shared by detection canaries and /advice consumers.

A flag file (default ``/tmp/ganyan-halt.flag``, override with
``GANYAN_HALT_FLAG_PATH``) holds JSON metadata about why the system is
halted. Existence of the file = halted; consumers should suppress
Kelly-sized stake recommendations and render picks/confidence as
informational only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict


DEFAULT_PATH = "/tmp/ganyan-halt.flag"


class HaltState(TypedDict):
    reason: str
    source: str
    timestamp: str


def _flag_path() -> Path:
    return Path(os.environ.get("GANYAN_HALT_FLAG_PATH", DEFAULT_PATH))


def is_halted() -> Optional[HaltState]:
    """Return halt state if halted, else None."""
    p = _flag_path()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            raise ValueError("Invalid halt state")
        return {
            "reason": data.get("reason", "unspecified"),
            "source": data.get("source", "unknown"),
            "timestamp": data.get("timestamp", ""),
        }
    except (ValueError, OSError):
        return {"reason": "halt flag present but unreadable", "source": "unknown", "timestamp": ""}


def set_halt(reason: str, source: str) -> None:
    """Write halt flag. Does NOT overwrite an existing halt — first writer wins.

    Rationale: if canary A halted at 12:00 and canary B fires at 12:30, we want
    the operator to see canary A's reason (the root cause) rather than the
    derivative symptom canary B detected.
    """
    p = _flag_path()
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reason": reason,
        "source": source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with p.open("x") as stream:
            json.dump(payload, stream, indent=2)
    except FileExistsError:
        pass


def clear_halt() -> None:
    """Remove halt flag. Idempotent."""
    p = _flag_path()
    try:
        p.unlink()
    except FileNotFoundError:
        pass
