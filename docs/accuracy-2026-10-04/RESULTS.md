# Forward accuracy test — 2026-10-04

> **Correction (2026-10-05).** The "vs AGF" columns below are biased in
> the models' favour. The runner counted races with no AGF at all (247 in
> the selection year, 91 in the test year) as misses for the AGF favourite
> while still scoring the models on them. On races that have AGF, the
> favourite's top-1 is **34.31%** (selection year) and **34.21%** (test
> year), not 33.00% / 33.71%. Against that, every arm in the test year is
> roughly 0.3–0.5 pp *below* the market. The baseline comparisons (model vs
> model on the same races) are unaffected. Corrected market comparisons
> are in the "Round 2" section; the runner was fixed in the same change.

Primary metric: top-1 hit rate (did the model's #1 pick win). Paired exact
McNemar test on the same races. Market baseline: the AGF favourite.

## Setup

- **Data:** TJK cards + full-field results rebuilt from scratch in a cloud
  session, 2023-01-01 → 2026-10-02 (24,221 races, 24,201 resulted), pedigree
  crawled for 16,314 / 16,315 horses. This is **not** the production database.
- **Code:** branch `audit-2026-09-18-repairs-ie4xic` at `8ed8b3e` (feature
  schema 2, 41 features), including the result-data fixes in `b6c75cf`
  (late withdrawals marked scratched; non-finishers ranked last).
- **Runner:** `scripts/accuracy_experiment.py` with the frame cache from
  `scripts/build_frame_cache.py`. Every head records its exact training window;
  scoring uses `ml/gate.py`'s exclusion rule (field ≥ 4, complete result).
- **Protocol:** arms were chosen on a validation year, the comparisons were
  fixed and approved, then the test year was scored once.

| Phase | Train on | Score on | Scored races | Excluded |
|---|---|---|---|---|
| Selection | 2023-01-01 → 2024-09-30 | 2024-10-01 → 2025-09-30 | 6,478 | 109 (1.7%) |
| Final test | 2023-01-01 → 2025-09-30 | 2025-10-01 → 2026-10-02 | 6,245 | 71 (1.1%) |

Arms (holdout = most recent share of the training window kept out of fitting):

- `ranker_90d` — **baseline**: single LambdaRank ranker with `ganyan train`
  defaults (last 90 days, 20% holdout).
- `ranker_all` — same ranker on all history (20% holdout).
- `ranker_all_h05` — all history, 5% holdout.
- `ensemble_default` — the 10 approved heads, each with its command's defaults
  (ranker/value/finish-time 90 days, specialists + linear all history), ordered
  by the production #1-vote rule.
- `ensemble_all_h05` — the same heads, all on all history, 5% holdout.
- `ensemble_all_h05_meanprob` — the `ensemble_all_h05` heads ordered by mean
  probability instead of the vote count.

## Selection year (2024-10-01 → 2025-09-30, n = 6,478)

AGF favourite: **33.00%**.

| Arm | Top-1 | Top-3 | Δ vs baseline | p | Δ vs AGF | p |
|---|---|---|---|---|---|---|
| ranker_90d (baseline) | 31.99% | 66.42% | — | — | −1.02 pp | 0.026 |
| ranker_all | **34.04%** | 68.60% | +2.05 pp | 1.4e-5 | +1.03 pp | 0.0004 |
| ranker_all_h05 | 33.88% | 68.57% | +1.90 pp | 8.0e-5 | +0.88 pp | 0.003 |
| ensemble_default | 33.74% | 64.80% | +1.76 pp | 1.7e-5 | +0.74 pp | 0.043 |
| ensemble_all_h05 | 33.81% | 64.74% | +1.82 pp | 1.4e-4 | +0.80 pp | 0.008 |
| ensemble_all_h05_meanprob | 32.88% | 67.18% | +0.90 pp | 0.071 | −0.12 pp | 0.76 |

Dropped after selection: `ranker_all_h05` (no gain over `ranker_all`) and
mean-probability ordering (worse top-1 than the vote rule).

## Final test year (2025-10-01 → 2026-10-02, n = 6,245, 367 days)

Comparisons fixed before scoring: primary `ranker_all` vs `ranker_90d`;
secondary `ensemble_all_h05` vs `ensemble_default`; every arm vs AGF.
AGF favourite: **33.71%**.

| Arm | Top-1 | Top-3 | Δ vs baseline | p | Δ vs AGF | p |
|---|---|---|---|---|---|---|
| ranker_90d (baseline) | 32.46% | 67.77% | — | — | −1.25 pp | 0.008 |
| ranker_all | 33.66% | 67.72% | **+1.20 pp** | **0.011** | −0.05 pp | 0.90 |
| ensemble_default | 33.92% | 65.00% | +1.46 pp | 0.0006 | +0.21 pp | 0.57 |
| ensemble_all_h05 | 33.96% | 65.17% | +1.51 pp | 0.002 | +0.26 pp | 0.42 |

Secondary: `ensemble_all_h05` vs `ensemble_default` = **+0.05 pp, p = 0.92**
(no difference).

## Conclusions

1. **Proven improvement:** training the single ranker on all history instead
   of the 90-day default raises forward top-1 by **+1.20 pp** (32.46% →
   33.66%, p = 0.011, 6,245 races over 367 days). That clears the gate's bar
   (≥365 days, ≥1500 races, ≥+1 pp, p < .05), and the selection year
   pointed the same way (+2.05 pp).
2. **No model beats the market favourite.** In the test year every arm is
   within ±0.3 pp of the AGF favourite (best +0.26 pp, p = 0.42). On this
   data the models do not reliably beat simply picking the AGF favourite.
   The 90-day default ranker was significantly *worse* than the favourite in
   both years.
3. **The ensemble adds nothing measurable over one all-history ranker** on
   top-1 (33.96% vs 33.66%; not tested head-to-head, so not a claim), and it is
   ~2.5 pp worse on top-3 (65.2% vs 67.7%) in both years.
4. **The ensemble's training windows do not matter** (+0.05 pp, p = 0.92).

## Limits

- Rebuilt data, not the production database. Snapshot-based features (late AGF
  drift, late program changes) and external signals (tipsters, workouts,
  discipline, steward reports) are empty here because they cannot be
  reconstructed after the fact; production has some of them.
- These are not the deployed weights, which have no recorded training window
  and cannot be gated. The baseline is today's code with the CLI defaults.
- One training run per arm (no seed variation); one test year.
- Production decisions should rerun `ganyan model-gate` on the production
  database.

## Reproduce

```bash
uv run python scripts/build_frame_cache.py --from 2023-01-01 --to 2026-10-02 \
    --workers 4 --out logs/experiments/frames.pkl
uv run python scripts/accuracy_experiment.py --data-from 2023-01-01 \
    --train-to 2025-09-30 --score-from 2025-10-01 --score-to 2026-10-02 \
    --out logs/experiments/final --frame-cache logs/experiments/frames.pkl \
    --arms ranker_90d ranker_all ensemble_default ensemble_all_h05
```

Raw reports: `selection_report.json`, `final_report_primary.json`,
`final_report_secondary.json` in this directory.

## Round 2 — repaired weights and new approaches (selection year only)

Changes since round 1: `weight_kg` was missing for ~40% of runners because
overweight cells ("55+0.70Fazla Kilo") failed to parse (fixed in
`97fffaf`; all results re-scraped, 0% missing now), and the market
comparison now uses only races with AGF.

Protocol: approaches judged on the selection year only. The 2025-10 → 2026-10
test year was **not** scored again. Proof requires a new untouched window
(≥365 days from 2026-10-03).

Selection year 2024-10-01 → 2025-09-30, train ≤ 2024-09-30. n = 6,478
(6,231 with AGF). Baseline: `ranker_all`. AGF favourite on AGF races:
**34.31%**.

| Arm | Top-1 | Δ vs ranker_all | p | Top-1 on AGF races | Δ vs AGF | p |
|---|---|---|---|---|---|---|
| ranker_all (baseline) | 33.58% | — | — | 34.04% | −0.27 pp | 0.060 |
| ranker_90d | 31.97% | −1.61 pp | 0.0003 | 32.79% | −1.52 pp | 0.0004 |
| ranker_win | 33.50% | −0.08 pp | 0.86 | 34.33% | +0.02 pp | 1.0 |
| ranker_hp_leaves15 | 33.76% | +0.19 pp | 0.17 | 34.23% | −0.08 pp | 0.33 |
| ranker_hp_leaves63 | 33.42% | −0.15 pp | 0.42 | 33.99% | −0.32 pp | 0.033 |
| ranker_hp_lr02 | 33.53% | −0.05 pp | 0.83 | 34.12% | −0.19 pp | 0.13 |
| ranker_hp_ff07 | 33.84% | +0.26 pp | 0.064 | 34.31% | +0.00 pp | 1.0 |
| blend_ranker_all | 33.82% | +0.25 pp | 0.072 | 34.30% | −0.02 pp | 1.0 |
| blend_ranker_win | 33.50% | −0.08 pp | 0.77 | 34.33% | +0.02 pp | 1.0 |
| ensemble_all_h05 | 33.61% | +0.03 pp | 0.96 | 33.80% | −0.51 pp | 0.080 |

Market blend weights (conditional logit, fitted on 3,353 races,
2024-04-02 → 2024-09-30, using a model trained only before that window):
`blend_ranker_all` AGF 1.022 / model 0.006; `blend_ranker_win` AGF 0.980 /
model 0.050. The fit puts essentially all weight on the market.

### Round 2 conclusions

1. **Nothing beats the AGF favourite.** The best arms tie it to within
   0.02 pp, and the blend fit gives the model almost zero weight once AGF is
   known. On this data the models add no measurable information about the
   winner beyond the market's own favourite.
2. **Repairing weight did not change top-1** (`ranker_all` 33.58% vs 34.04%
   in round 1 on the old data; well within noise).
3. **No tuning, objective or blend variant beats the all-history ranker**
   (largest +0.26 pp, p = 0.064).
4. The 90-day window remains clearly worse (−1.61 pp, p = 0.0003),
   consistent with round 1.
