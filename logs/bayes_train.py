"""One-off Bayesian training script (not part of the package)."""
import os
import argparse
import time
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ganyan.predictor.bayes.trainer import train_full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_date", required=True)
    p.add_argument("--to", dest="to_date", required=True)
    p.add_argument("--out", dest="output_base", required=True)
    p.add_argument("--iters", type=int, default=30_000)
    args = p.parse_args()

    fd = date.fromisoformat(args.from_date)
    td = date.fromisoformat(args.to_date)
    out = Path(args.output_base)

    print(f"=== bayes train start {time.strftime('%H:%M:%S')} ===")
    print(f"window: {fd} → {td}, iters: {args.iters}, out: {out}")

    t0 = time.time()
    eng = create_engine(os.environ.get('DATABASE_URL', "sqlite:///data/ganyan.db"))
    with Session(eng) as s:
        train_full(s, fd, td, out, n_iter=args.iters)
    elapsed = time.time() - t0
    print(f"=== bayes train done in {elapsed:.1f}s ===")


if __name__ == "__main__":
    main()
