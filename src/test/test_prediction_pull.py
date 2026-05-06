import pytest
from sqlalchemy import select

from ganyan.db.models import Prediction
from ganyan.db.session import get_session_factory


def get_predictions(limit: int = 10):
    factory = get_session_factory()
    session = factory()
    try:
        stmt = select(Prediction).limit(limit)
        result = session.execute(stmt)
        return result.scalars().all()
    finally:
        session.close()


def test_pull_predictions():
    predictions = get_predictions(5)
    assert isinstance(predictions, list)


if __name__ == "__main__":
    preds = get_predictions()
    for p in preds:
        print(f"Prediction(id={p.id}, probability={p.probability}, model={p.model_version}, race_entry={p.race_entry})")