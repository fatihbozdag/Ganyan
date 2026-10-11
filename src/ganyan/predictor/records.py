"""Persist complete pre-race prediction runs; evaluate immutable records only."""
from collections import defaultdict
from types import SimpleNamespace
from uuid import uuid4

from ganyan.db.models import Prediction, Race, RaceEntry
from ganyan.time import is_upcoming, race_cutoff, utcnow


def save_predictions(session, race_id, predictions, version, artifact=None):
    if not predictions:
        return []
    race = session.get(Race, race_id)
    now = utcnow()
    if race is None or not is_upcoming(race, now):
        raise ValueError("Only upcoming races with a known post time can be recorded")
    entries = {e.horse_id: e for e in race.entries if not e.scratched}
    if not predictions or {p.horse_id for p in predictions} != set(entries):
        raise ValueError("A recorded prediction must cover the complete active field")
    import math
    if len(predictions) != len(entries) or any(not math.isfinite(p.probability) or not 0 <= p.probability <= 100 for p in predictions):
        raise ValueError("Invalid prediction probabilities")
    if abs(sum(p.probability for p in predictions) - 100) > 0.1:
        raise ValueError("Prediction probabilities must sum to 100")
    from ganyan.predictor.ml.artifacts import pipeline_digest
    artifact = {**(artifact or {}), "pipeline_sha256": pipeline_digest()}
    run_id = uuid4().hex
    for rank, pred in enumerate(predictions, 1):
        entry = entries[pred.horse_id]
        factors = {**pred.contributing_factors,
                   "_run_id": run_id, "_rank": rank, "_artifact": artifact or {},
                   "_cutoff": race_cutoff(race).isoformat(),
                   "_agf": float(entry.agf) if entry.agf is not None else None}
        session.add(Prediction(race_entry_id=entry.id, model_version=version[:50],
                               predicted_at=now, probability=pred.probability,
                               confidence=pred.confidence, factors=factors))
        entry.predicted_probability = pred.probability
    # Remove obsolete active slots for withdrawals; history rows remain intact.
    for entry in race.entries:
        if entry.scratched:
            entry.predicted_probability = None
    return predictions


def recorded_entries(session, race_id):
    """Latest complete pre-post run, independent of mutable probability slots."""
    race = session.get(Race, race_id)
    if race is None:
        return []
    entries = {e.id: e for e in race.entries if not e.scratched}
    rows = (session.query(Prediction).filter(Prediction.race_entry_id.in_(entries),
            Prediction.predicted_at < race_cutoff(race))
            .order_by(Prediction.predicted_at.desc(), Prediction.id.desc()).all())
    groups = defaultdict(dict)
    for row in rows:
        factors = row.factors or {}
        # Legacy rows lack a reliable run/rank identity. Do not manufacture one.
        if factors.get("_run_id") and factors.get("_rank"):
            groups[factors["_run_id"]][row.race_entry_id] = row
    for run in groups.values():
        if set(run) != set(entries):
            continue
        result = []
        for eid, row in run.items():
            entry = entries[eid]
            attrs = {c.name: getattr(entry, c.name) for c in RaceEntry.__table__.columns}
            attrs.update(horse=entry.horse, predicted_probability=float(row.probability),
                         predicted_rank=row.factors["_rank"], agf=row.factors.get("_agf"))
            result.append(SimpleNamespace(**attrs))
        return sorted(result, key=lambda e: e.predicted_rank)
    return []
