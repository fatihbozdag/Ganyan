#!/bin/bash
cd /Users/fatihbozdag/Downloads/Cursor-Projects/ganyan
echo "=== bayes smoke train start $(date) ==="
uv run python - <<'PY'
import os
import time
from datetime import date
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from ganyan.predictor.bayes.trainer import train_full

t0 = time.time()
eng = create_engine(os.environ.get('DATABASE_URL', "sqlite:///data/ganyan.db"))
with Session(eng) as s:
    train_full(s, date(2026,4,1), date(2026,4,28),
               Path("models/bayes_pl_smoke"), n_iter=3_000)
print(f"smoke train done in {time.time()-t0:.1f}s")
PY
echo "=== bayes smoke train done $(date) ==="
