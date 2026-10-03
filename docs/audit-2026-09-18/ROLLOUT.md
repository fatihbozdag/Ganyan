# Rollout and prediction-accuracy protocol

## Prepare the corrected pipeline

1. Back up the production database and model directory. Stop the one scheduler before
   deploying the schema/code pair. The repair session did not migrate production,
   refit its models, refresh source data, or modify its ledger.
2. Install with `uv sync --frozen --all-extras`, then apply `uv run alembic upgrade head`.
   Migration `f3a4b5c6d7e8` adds nullable race-specific age/trainer, physical draw, and
   condition history. It deliberately does not infer historical values from today's
   horse profile or from the horse's program number.
3. Refresh source-backed card attributes in a controlled backfill. Newly captured
   historical signals are not retroactively available before a race. Legacy signal
   timestamps were sometimes local time: do not subtract three hours globally;
   verify each source/run's clock before correcting archival timestamps.
4. Set a stable `SECRET_KEY`. Use loopback for a local dashboard; remote mutations
   require `MUTATION_TOKEN` in an `Authorization: Bearer ...` header. Local HTMX uses
   session CSRF. `.env.example` disables scheduler and launch refresh; enable them
   deliberately in one process after verification.
5. Confirm `/ops/health`, scraper job outcomes, and complete pre-race prediction
   records. Missing post times prevent recording; obtain the real post time.

## Fit and compare candidates

Predeclare three chronological periods: fitting, model selection/calibration, and a
final forward test. The training command internally separates fitting, selection,
and a diagnostic holdout by date. **Every date used by that command is development
information** and must precede the separate promotion window. Repeatedly choosing
models against the final window turns it into selection data: reserve a new window.

Example command shapes (replace the dates with available, disjoint periods):

```sh
uv run ganyan train --from YYYY-MM-DD --to YYYY-MM-DD --model-name accuracy_v2
uv run ganyan model-gate --candidate candidates/accuracy_v2 \
  --from YYYY-MM-DD --to YYYY-MM-DD --output models/candidates/accuracy_v2.gate.json
uv run ganyan model-promote --candidate candidates/accuracy_v2 \
  --gate models/candidates/accuracy_v2.gate.json
```

The gate requires exact development-date provenance for both models, current candidate
feature schema/code, at least 1,500 actually scored paired races spanning 365 days,
no more than 5% incomplete-result exclusions, at least +1 percentage point top-1
improvement and paired McNemar p < .05. It reports an AGF baseline. Prediction errors
fail the gate. Retrospective mode is diagnostic and never authorizes promotion.
The manifest switch is atomic and serialized; weights are installed in immutable
content-addressed releases. A passing report is invalid after code, candidate, or
incumbent changes. Never invent missing legacy metadata to make a gate pass.

Compare the default ranker, a regularized linear model, and bounded feature ablations
on selection data first. Keep AGF-favourite and chance baselines; report field-size,
race-type, track, and time-period results alongside overall top-1, top-3, log loss
and calibration. Evaluate the **whole serving ensemble separately** before rollout:
a passing single-head gate does not prove an ensemble gain.

## Historical limits and ledger handling

Historical card fields such as AGF, equipment and jockey are reconstructed from the
stored card, not a complete archival pre-race feature snapshot. Dynamic signals now
obey capture cutoffs, but this cannot prove every old card value was available at
its claimed prediction time. Use newly recorded prospective runs for operational
accuracy claims. Their run ID, ordering, probabilities, AGF, artifact and code hash
are recorded together; unidentifiable legacy rows are excluded from that evaluation.

Old age/trainer/draw gaps remain missing until source-backed recovery. Corrected
features change inputs to legacy weights; loading their hashes proves identity,
not accuracy under the new feature definitions. The bundled manifest retains ten
compatible legacy heads and excludes the weather-dependent linear Plackett–Luce
head whose required features are absent. Refit and revalidate before production use.

New Plase losses settle when the pool is confirmed; existing incomplete Plase rows
can use the existing `resettle_plase_picks` maintenance function. Previously overwritten
coupons cannot be reconstructed without backups. Review old multi-race rows marked
as hits with zero payout against published pools; do not automatically rewrite a
financial ledger. New coupons are immutable, require explicit pool indexes/windows,
and stay pending until payout is published. Automatic coupon creation skips programs
without verified windows; supply `ganyan multi-picks`' explicit window and pool index.

Docker needs a mounted model directory and database configuration; its build was
not run here because Docker is unavailable. The Python wheel/source build and
PostgreSQL migration were verified. GUI command wiring was reviewed/compiled;
interactive desktop behavior still requires a GUI smoke test.
