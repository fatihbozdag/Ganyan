"""Fail closed on missing data, unavailable health, or degenerate predictions."""
import os
from urllib.request import urlopen
from ganyan.db.models import Race
from ganyan.db.session import get_session
from ganyan.predictor.halt_flag import set_halt
from ganyan.predictor.uniformity_guard import is_uniform
from ganyan.time import today


def check_data(session):
    races = session.query(Race).filter(Race.date == today()).all()
    if not races:
        raise RuntimeError("No current-day race program")
    for race in races:
        entries = [e for e in race.entries if not e.scratched]
        if not entries or all(e.agf is None for e in entries):
            raise RuntimeError(f"Missing entries/AGF for race {race.id}")
        if any(e.predicted_probability is None for e in entries):
            raise RuntimeError(f"Incomplete predictions for race {race.id}")
        if is_uniform([float(e.predicted_probability) for e in entries]):
            raise RuntimeError(f"Uniform predictions for race {race.id}")


def main():
    try:
        with urlopen(os.environ.get("GANYAN_HEALTH_URL", "http://localhost:5003/ops/health"), timeout=10) as response:
            if response.status != 200:
                raise RuntimeError("Health endpoint is degraded")
        with get_session() as session:
            check_data(session)
    except Exception as exc:
        set_halt(str(exc), "heartbeat")
        print(f"heartbeat failed: {exc}")
        return 1
    print("heartbeat OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
