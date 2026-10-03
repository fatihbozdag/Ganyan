# Audit repair status — 2026-09-18

The repair plan is implemented in the working tree. No production refit,
promotion, schema migration, source backfill, or ledger rewrite was performed.
There is no measured new accuracy uplift yet. Existing `AGENTS.md` was preserved.

## Findings and implemented repairs

| Audit | Repair | Verification |
|---|---|---|
| F01 | Shared payout units; missing pools excluded independently of hits; charge only generated tickets | Audit settlement regressions |
| F02 | Confirmed Plase pools settle losing tickets; incomplete prior Plase rows can be resettled | Plase-loss regression |
| F03 | Outcome-independent expected tie accuracy for tree/linear metrics | Constant scorer gives 25% top-1 / 75% top-3 in four-horse fields |
| F04 | Exact development provenance; final paired count/date span; exclusion limits; code/artifact hashes | Window, overlap, one-race rejection and promotion tests |
| F05 | UTC capture times; pre-post/as-of filters for snapshots and external signals | Post-race snapshot and tipster regressions |
| F06 | Explicit approved heads; candidate-only training; verified, serialized atomic promotion; GUI uses CLI gate | Manifest, artifact, promotion tests; 10 approved heads load |
| F07 | Authoritative TJK horse IDs override same-name cache hits | Same-name/different-ID test; PostgreSQL identity check |
| F08 | Immutable coupons; explicit pool identity and verified race windows | Multi-race persistence tests |
| F09 | Scheduler and external-plugin failures propagate; partial track fetches fail | Scheduler and plugin failure regressions |
| F10 | Loopback default; CSRF-protected local writes; bearer auth for remote writes | Local/cross-origin/remote rejection tests |
| F11 | Complete, pre-post prediction runs record rank, run ID, AGF, artifact/code identity; evaluation reads these | Mutable-slot overwrite and post-result recording rejection tests |
| F12 | Multi-race grading waits for matching pool window and payout | Late-payout regression |
| F13 | Updated race conditions retained with observation history | Surface/distance/post-time refresh test |
| F14 | Shared ML field construction; scratches excluded; incomplete labels rejected; historical age/trainer columns | Train/serve equality and incomplete Bayesian field tests |
| F15 | Physical start gate separated from program/bet number | Program/results parser assertions; missing-draw regression |
| F16 | Health requires results and recorded predictions; heartbeat fails on missing/degenerate data | Missing-results/predictions regression |

Additional prediction repairs: finish-time models now rank lower times first;
selection/calibration are separate from the diagnostic final holdout; missing
Bayesian features are neutral after scaling; speed/pace baselines use prior dates;
a race-constant, unidentifiable Bayesian intercept is fixed at zero. Ensemble
records identify the models actually loaded, and confidence uses applicable heads.
Halt handling fails closed for malformed files and suppresses the whole advice
response if a later race triggers it. Recent workout speed now uses the newest
workout before selecting its closest distance split.

## Verification

- Full pytest suite: **301 passed, 1 skipped** (19.78 seconds). The skip is the
  opt-in PostgreSQL test verified separately. Ordinary tests use synthetic
  SQLite data; leakage tests no longer depend on the configured live database.
- PostgreSQL 15 integration: **1 passed**, applying the complete migration chain,
  persisting the new fields and duplicate-name/distinct-ID horses, then downgrading
  and upgrading the new migration in an isolated temporary schema.
- Package: wheel and source archive built; approved manifest included in wheel.
- CLI: main help, `model-gate --help`, and `model-promote --help` load.
- Python compilation and `git diff --check` pass.
- CI added for Python 3.12, PostgreSQL 16, full tests and package build.

## Rollout limits

Follow [ROLLOUT.md](ROLLOUT.md) before deploying. Missing historical attributes,
uncertain legacy timestamps, unknown model-development dates and overwritten
coupons require source/backups; code cannot reconstruct them reliably. Historical
cards are not complete point-in-time feature archives. Legacy prediction rows
without trustworthy run identity are intentionally excluded from measured live
accuracy. Existing weights need revalidation with corrected feature definitions.

The manifest excludes one legacy linear Plackett–Luce head requiring unavailable
weather features. A single-head promotion gate is not an ensemble accuracy test.
Docker runtime/build and interactive GUI operation were not verified here; Docker
is unavailable, while GUI source and command wiring were compiled/reviewed.

The original `REPORT.md` and reproduction artifacts remain historical audit
evidence; `reproduce.py` asserts old defects and is not the repaired test suite.
Use `tests/test_audit_regressions.py` for desired-behavior regressions.
