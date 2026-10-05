# Frozen candidates and evaluation plan — fixed 2026-10-05

This plan was written and committed before any race in the evaluation
window had been run under it. Nothing below may be changed after
evaluation data is looked at; deviations must be reported as such.

## What is frozen

Artifacts: `models/candidates/frozen-2026-10-02/` (`frozen.json` holds every
hash). All heads trained on the rebuilt TJK data 2023-01-01 → 2026-10-02,
feature schema 2, pipeline digest `c826e8bb7969c96c…` (full value in
`frozen.json`), code at commit `5b84d04`.

| Id | Arm | What it is | Model SHA-256 |
|---|---|---|---|
| C1 | `ranker_hp_ff07` | All-history LambdaRank ranker, `min_data_in_leaf=200`, `feature_fraction=0.7` | `a2a6b9fc…` |
| C2 | `blend_ranker_all` | Conditional logit `0.5433·log(AGF) + 0.4185·log p(ranker_all)`; weights fitted on 2026-04-04 → 2026-10-02 (3,246 races) with a ranker trained only before that window | `054691b4…` (+ blend-fit head) |
| R | `ranker_all` | Reference: all-history ranker, defaults (the new `ganyan train` default) | `054691b4…` |
| M | AGF favourite | Market reference: highest AGF in the race | — |

## Why these, and what to expect

Chosen on the 2024-10 → 2025-09 selection year (`RESULTS.md`, Round 2):
C1 had the best single-model top-1 (33.84%, +0.26 pp vs R, p = 0.064) and
tied the AGF favourite on races with AGF (34.31% vs 34.31%); C2 (as
`blend_ranker_all`) was second (+0.25 pp vs R) and also tied the market.
**The prior evidence predicts a null result against the market.** The
blend weights are unstable: the selection-year fit gave AGF 1.02 / model
0.006, the frozen fit gives 0.54 / 0.42.

## Evaluation window

From **2026-10-03** to the first date on which both hold: ≥ 365 days
covered and ≥ 1,500 scored races (expected ~2027-10-03). No interim
looks are reported as results.

## Scoring rules

- Races: resulted, ≥ 4 runners after scratches, complete placings, a
  winner (the `ml/gate.py` rule); abort if > 5% of races are excluded.
- Late withdrawals ("Koşmaz") count as scratched; non-finishers rank last.
- Metric: top-1 (the model's #1 pick won). Secondary: top-3.
- Comparisons with M use only races where at least one runner has AGF.
- Features are built as of each race's post time; snapshot and external
  features stay as available at that time.
- Code must have the frozen pipeline digest; any feature-code change
  invalidates the run (the runner refuses a mismatched cache or head).

## Hypotheses and tests

Two-sided exact McNemar on paired races, Bonferroni-corrected over the
three primary tests (each α = 0.05 / 3 = 0.0167):

1. **H1** C1 top-1 > M (races with AGF).
2. **H2** C2 top-1 > M (races with AGF).
3. **H3** C1 top-1 > R (all scored races).

All three are reported whatever the outcome, with effect sizes and
discordant counts. A "better than the market" claim requires H1 or H2 to
pass; promotion via `ganyan model-promote` remains a separate gate on the
production database.

## How to run

```bash
uv run python scripts/build_frame_cache.py --from 2023-01-01 --to <END> \
    --workers 4 --out logs/experiments/frames_eval.pkl
uv run python scripts/accuracy_experiment.py --data-from 2023-01-01 \
    --train-to 2026-10-02 --score-from 2026-10-03 --score-to <END> \
    --out models/candidates/frozen-2026-10-02 \
    --frame-cache logs/experiments/frames_eval.pkl \
    --arms ranker_all ranker_hp_ff07 blend_ranker_all
```

The runner reuses the frozen heads (it trains nothing when `active.json`
exists) and reads C2's weights from `frozen.json` instead of refitting.
H1/H2 come from `delta_vs_agf_pp` / `mcnemar_p_vs_agf`; H3 needs a second
run with `--arms ranker_hp_ff07 ranker_all` (baseline = first arm), or a
paired count from the two arms' per-race hits.
