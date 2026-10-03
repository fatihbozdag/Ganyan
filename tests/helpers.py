"""Explicit historical prediction records for isolated test fixtures."""
from datetime import timedelta
from uuid import uuid4
from ganyan.db.models import Race, Prediction
from ganyan.time import race_cutoff


def record_fixture_predictions(session):
    session.flush()
    for race in session.query(Race).all():
        entries = sorted([e for e in race.entries if e.predicted_probability is not None],
                         key=lambda e: (-float(e.predicted_probability), e.horse_id))
        run = uuid4().hex
        for rank, entry in enumerate(entries, 1):
            session.add(Prediction(race_entry_id=entry.id, model_version="fixture",
                predicted_at=race_cutoff(race) - timedelta(minutes=1),
                probability=entry.predicted_probability,
                factors={"_run_id": run, "_rank": rank, "_agf": entry.agf}))
