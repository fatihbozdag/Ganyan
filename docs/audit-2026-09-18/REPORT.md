# Ganyan Repository Audit — 2026-09-18

## Assessment

The repository has useful modular boundaries, a substantial unit-test suite, historical prediction rows, and several operational guards. However, current evaluation numbers, payout summaries, and model-promotion controls are not reliable enough to serve as acceptance criteria without correction. The most consequential defects are inconsistent payout accounting, outcome-dependent evaluation ties, a bypassable OOS gate, and mutable historical records.

This is a repository audit, not a new estimate of predictive performance or betting profitability. No production training, scraping, migration, deployment, or model replacement was performed. Application source files and the existing `AGENTS.md` were not changed.

## Scope and verification

- Baseline: Git HEAD `4631430`, plus the existing untracked local GUI and artifacts. Untracked content is not necessarily present in another clone.
- Reviewed paths: ingestion/parser/backfill and external signals; database models/migrations; feature construction, LightGBM and Bayesian paths; ensemble loading/persistence; prediction evaluation and single/multi-race ledgers; CLI/Flask entry points; scheduler/canaries; GUI training/promotion; packaging and operational documentation.
- `.venv/bin/pytest -q`: **269 passed, 1 failed, 1 setup error**, 46 warnings, 86.06 seconds, Python 3.14.0. Both unsuccessful checks in `tests/test_predictor/test_ml/test_leakage.py` failed to connect to PostgreSQL on localhost:5432. They did not demonstrate failed leakage assertions. Their fixture calls the application's default `get_session()`, rather than an isolated test database.
- `.venv/bin/alembic heads`: one head, `e2f3a4b5c6d7`. Migrations were not applied.
- `.venv/bin/python docs/audit-2026-09-18/reproduce.py`: **13 defect reproductions completed** using an in-memory SQLite database, mock predictions, and mocked gate execution. See [results](reproduction-results.json) and [script](reproduce.py). Its assertions intentionally characterize existing defects; they are not desired-behavior regression tests. The synthetic scheduler exception in [stderr](reproduction-stderr.txt) is intentional.
- No live PostgreSQL data inspection, fresh model fit/backtest, external TJK rule verification, Docker build, dependency vulnerability scan, or interactive GUI execution. Static findings are identified below. The audit does not establish the frequency or numerical impact of these defects in the live dataset.

## High-priority findings

### F01 — Exotic backtesting still doubles Üçlü payouts (P1, reproduced)

**Locations:** `src/ganyan/predictor/exotic_evaluate.py:201` and `:292`; compare `src/ganyan/predictor/picks.py:406` and its `BIRIM_TL_BY_STRATEGY` mapping.

Both exotic evaluation paths multiply the published pool payout directly by ticket stake. The ledger applies the repository's documented two-TL Üçlü unit divisor. With a published payout of 10 and a 100-TL winning ticket, the backtester reports **1,000 TL** while the ledger reports **500 TL**. Backtest ROI therefore contradicts the corrected ledger. Additionally, missing-payout races are excluded only when the model wins, and stake uses requested `top_n` even if fewer combinations exist.

**Repair:** centralize pool-unit settlement for ledger and both backtest paths; determine pool eligibility independently of the prediction outcome; charge only actual combinations. Add agreement tests across all settlement consumers before interpreting existing backtest results.

### F02 — Losing Plase picks disappear from financial results (P1, reproduced)

**Locations:** `src/ganyan/predictor/picks.py:359`, `:379`, `:457`, `:537`; `src/ganyan/scraper/tjk_api.py:1193`.

The scraper assigns Plase payouts to placed horses only. The grader nevertheless requires a payout on the *picked losing horse* before recording its loss. A losing pick in a race with confirmed payouts becomes `graded=True, hit=False, net_tl=None`. Resettlement repeats the same condition, so the loss remains absent. Summaries count only priced stakes for ROI: the isolated 100-TL loss reports **0% ROI**, and mixed samples can exclude losers preferentially.

**Repair:** represent pool availability at race level; settle every losing pick to zero payout when that pool is confirmed. Distinguish pending results from absent pools, then recompute affected ledger totals from retained evidence.

### F03 — Training evaluation uses finish order to resolve score ties (P1, reproduced)

**Locations:** `src/ganyan/predictor/ml/features.py:390`; `src/ganyan/predictor/ml/trainer.py:418`.

Training-frame rows are sorted by actual finish position. Evaluation then sorts only on predicted score, with no outcome-independent tie rule. A model returning identical scores for every horse receives **100% top-1 accuracy on two four-horse races** in the reproduction. Live inference instead resolves ties using AGF and horse ID. This can inflate reported holdout/CV accuracy for sparse or degenerate models; no claim is made that all existing reported accuracy is inflated by the same amount.

**Repair:** retain horse identifiers and use the exact serving tie policy, or a documented neutral tie metric. Re-evaluate affected artifacts after correcting the metric.

### F04 — The OOS gate can approve one scored race and overlapping training (P1, reproduced)

**Locations:** `logs/oos_model_gate.py:94`, `:96`, `:118`, `:151`.

The minimum sample check runs against queried races, before missing winners, prediction failures, and empty predictions are excluded. It is never repeated on scored races. Training-window separation is asserted in the docstring but never checked in model metadata. A mocked run with 1,500 queried races, only **one scored pair**, and overlapping training metadata returns **`swap: true`**. The gate also reports the nominal date window rather than enforcing temporal coverage of scored observations.

**Repair:** validate artifact training provenance, compute eligible paired coverage, enforce minimum size and time coverage on the final cohort, and fail closed on excessive exclusions. Bind gate results to model and feature-code hashes. Historical disjoint evaluation should be distinguished from forward deployment validation.

### F05 — Historical feature extraction lacks an as-of cutoff (P1, reproduced)

**Locations:** `src/ganyan/predictor/features.py:1078`, `:1123`; external-signal queries at `:812`, `:857`, `:947`, `:1007`; `src/ganyan/scraper/backfill.py:25`.

Snapshot-derived features read the first and last stored observations without limiting `taken_at` to the prediction time. Historical backfills also create snapshots at scrape time. Adding a snapshot a month after a race changes that race's AGF-drift feature from unavailable to **+80 points**. External-signal features similarly lack a capture-time boundary; workout dates are not restricted to precede the target race. Thus outcome-column blacklist tests do not establish temporal integrity.

**Repair:** pass an explicit prediction timestamp through feature construction; require both event-time and availability-time eligibility. Preserve source provenance and distinguish historical retrieval from observations actually available before the race. Add future-observation invariance tests.

### F06 — Training candidates and GUI backups silently join production inference (P1, static)

**Locations:** `src/ganyan/predictor/ml/ensemble.py:120`; `src/ganyan/predictor/ml/trainer.py:731`; `GanyanGUI/ganyan_gui/tabs/model.py:675`, `:701`, `:741`; `ops/README.md:72`.

The ensemble loads every root-level `models/*.meta.json`. A separately named training candidate in that directory immediately gains a vote. GUI promotion saves production backups there too, so backups become additional heads. The GUI's OOS button passes a model filename to `exotics-backtest --model`, but `_build_predictor` accepts only `bayesian`, `ml`, or `ensemble`. Promotion checks holdout metadata and user confirmation, not the documented OOS gate. The operations guide still advertises bare training as safe alongside the daemon, although bare training overwrites live weights.

**Repair:** explicitly manifest approved heads; isolate candidates and backups; make promotion require a matching validated gate artifact and publish model/metadata together. Wire the GUI to the actual gate and correct conflicting operational instructions.

### F07 — Name caching defeats stable horse identity (P1, reproduced)

**Locations:** `src/ganyan/scraper/backfill.py:230`, `:390`, `:492`.

The get-or-create helper respects TJK AtId, but ingestion first reuses a horse cached by name without checking that ID. If one existing horse has a name and a newly encountered horse shares it with a different AtId, the existing row is reused. The reproduction ingests AtId **222** but stores the entry against **111**, mixing histories and potentially overwriting mutable attributes.

**Repair:** key cache resolution by AtId and permit name fallback only for unresolved compatible identities. Cover a single pre-existing same-name horse as well as already-ambiguous names in regression tests.

### F08 — Multi-race persistence can rewrite settled history and confuse pool windows (P1, reproduced/static)

**Locations:** `src/ganyan/predictor/multi_race_picks.py:182`, `:225`, `:271`; `src/ganyan/scheduler.py:156`; `src/ganyan/cli/main.py:2015`.

Persisting a matching date/track/pool/strategy unconditionally replaces selections, resets settlement, and retains the original generation timestamp. Reproduction replaces a settled R1–R6 coupon with R3–R8 under the same ID. CLI/scheduler calls default to `pool_index=1` regardless of the selected starting race; grading matches that index without checking the race window. On a multi-pool card, the intended window can therefore be compared with another pool's winners.

**Repair:** identify pools and their explicit leg windows, freeze coupons once betting begins, preserve graded rows, and use immutable revisions for pre-race updates. Reject ambiguous pool/window mappings.

### F09 — Scheduled failures are recorded as successful execution (P1, reproduced)

**Locations:** `src/ganyan/scheduler.py:92`, `:121`, `:301`, `:370`, `:632`.

Jobs catch errors and return normally, including complete scrape failures; some per-race failures are silently rolled back. APScheduler's execution listener interprets a normal return as `JobStatus.success`, so failure counters and notifications miss these incidents. A synthetic network failure in the AGF job reproduces a normal return.

**Repair:** propagate fatal job errors after cleanup, explicitly record partial failure counts, and distinguish successful empty results from failed ingestion. Exercise the job/listener/health chain together in tests.

### F10 — Network-facing mutation endpoints have no access control (P1 when reachable, reproduced/static)

**Locations:** `src/ganyan/web/app.py:173`; `src/ganyan/web/routes.py:589`, `:662`, `:725`, `:813`.

The default server binds all interfaces, and scrape/predict POST routes have neither authentication nor CSRF/origin validation. An unauthenticated request carrying an unrelated Origin reaches `/predict/today` and returns 200 in the isolated client. With populated data, the route writes predictions; scraping routes trigger remote work and database updates. Actual network exposure was not measured.

**Repair:** default local installations to loopback, require authenticated mutation access where shared, and protect browser-submitted actions against CSRF. Do not rely on hiding the historical-scrape UI to secure its route.

### F11 — Current-day prediction refresh can rewrite completed-race evaluation (P1, static)

**Locations:** `src/ganyan/web/routes.py:668`, `:683`, `:686`; `src/ganyan/predictor/evaluate.py:85`; `src/ganyan/predictor/ml/predictor.py:210`.

The dashboard's prediction action selects every race today, including completed races, and writes single-model probabilities into the same fields the ensemble and evaluation use. Evaluation reads these mutable fields without enforcing a pre-post prediction timestamp or fixed model identity. A later refresh can therefore change historical dashboard accuracy and replace ensemble state, while previously generated picks remain unchanged.

**Repair:** restrict operational refresh to eligible upcoming races, separate exploratory scoring from recorded recommendations, and evaluate immutable timestamped prediction runs made before post time. Persist the ranking policy as well as probabilities.

## Medium-priority findings

### F12 — Multi-race wins settle permanently at zero before payouts arrive (P2, reproduced)

**Location:** `src/ganyan/predictor/multi_race_picks.py:291`.

If a winning combination exists but its payout is missing, the grader sets `hit=True`, payout zero, and `graded=True`. A later payout is ignored by the pending grader. Reproduction gives a winning one-TL coupon a **−1 TL** net and leaves it there after a 1,000-TL pool payout arrives. Keep financial settlement pending until payout is known and add an explicit resettlement path.

### F13 — Program refresh ignores changed race conditions (P2, reproduced)

**Location:** `src/ganyan/scraper/backfill.py:214`.

Existing races only receive a previously absent post time. A refreshed surface, distance, or revised existing post time is ignored. Reproduction submits turf/1,600m/15:00 over sand/1,200m/14:00 and retains all old values. Update validated mutable race fields and preserve their change history; prediction cutoffs must use the revised post time.

### F14 — Training and serving construct different field features (P2, reproduced/static)

**Locations:** `src/ganyan/predictor/ml/features.py:232`, `:412`; `src/ganyan/predictor/ml/predictor.py:139`.

Training requires minimum coverage before computing relative weight/HP/S20 features; inference averages any available values. One known HP among three horses yields missing training class indicators but a numeric serving indicator. Serving also builds field statistics before excluding scratched horses from prediction rows. Share one field-construction implementation, filter actual runners first, and test training/serving equivalence.

### F15 — Program number is used as physical starting-gate bias (P2, static)

**Locations:** `src/ganyan/scraper/tjk_api.py:110`, `:132`; `src/ganyan/predictor/features.py:1141`.

The parser deliberately stores the betting program number in `gate_number`, explicitly excluding the physical start gate. `compute_gate_bias` then interprets that value as an inside/outside draw position. Store program number and physical gate separately; keep coupon identifiers stable and calculate draw features only from the actual start position.

### F16 — Health can be green with no results or predictions (P2, reproduced)

**Locations:** `src/ganyan/web/routes.py:1008`; `scripts/heartbeat.sh:50`.

`_compute_health(today, None, None, 0)` returns `ok`: missing results are not rejected and `last_prediction_at` is unused. The heartbeat's count checks can also pass when there are no current-day entries or races. Combined with swallowed job failures, a nonfunctional data pipeline can appear healthy. Define time-aware expected data and prediction freshness checks and test empty/stale states.

## Additional limitations and maintenance risks

- **Historical horse attributes:** training reads mutable `Horse.age` and current trainer; reverse chronological backfills overwrite them. Per-entry age/trainer snapshots are needed for reproducible historical features (`ml/features.py:369`, `features.py:399`).
- **Bayesian model:** the track-distance additive term is identical for every horse in a race and cancels from Plackett–Luce probabilities (`bayes/model.py`). Missing values inserted as zero are subsequently z-scored, so they are not generally neutral despite comments. Historical speed/pace baselines are computed through the full frame end date rather than separately as of each race. These merit targeted model-design tests before refitting.
- **Scratch consistency:** the web Bayesian advice input includes all race entries (`web/routes.py:1408`), unlike the ML predictors' runner filtering. Harmonize field selection across the gate and recommendation paths.
- **Provenance and schema:** model version strings based on objective/iteration or head count are not immutable artifact identities. `reindex` silently supplies NaN for unavailable model features. Validate required feature schema and record artifact hashes/head manifests with prediction runs.
- **Operational portability:** cron triggers use Istanbul time but job bodies use host-local `date.today()`/`datetime.now()`. Re-prediction stops at 20:35 while AGF snapshots continue later. Operational documentation still lists the disabled monthly retrain job. Align clocks, coverage, and runbooks.
- **Packaging:** Docker installs the editable project before copying `src/`, omits model artifacts, and has no declared production-model mount/bootstrap flow. A clean image build and prediction smoke test remain unverified. GUI runtime dependency installation bypasses `uv.lock` and advertises Python 3.8 although the package requires 3.12+.
- **Testing/deployment:** SQLite `create_all` fixtures do not exercise PostgreSQL migration-only constraints, and two tests depend on the configured application database. No repository CI workflow or enforced coverage threshold was found. Add a disposable PostgreSQL migration/integration job and isolate expensive model smoke tests.
- **Scope boundary:** local logs/model outputs were inspected selectively for operational gate provenance; this was not a line-by-line review of every historical experiment, a security penetration test, or verification of published TJK payout rules.

## Recommended repair order

1. Fix payout accounting, tie handling, and final-cohort OOS validation (F01–F04). Mark existing affected figures as requiring recomputation; preserve original artifacts.
2. Make model promotion explicit and historical inputs/records immutable (F05–F08, F11). Add artifact identities and as-of prediction timestamps.
3. Repair scheduler failure reporting, mutation access, and health checks (F09–F10, F16).
4. Correct deferred settlement, program updates, field equivalence, and gate semantics (F12–F15); add focused regression tests for every reproduced case.
5. Run isolated PostgreSQL migration/integration checks and a clean packaging smoke test. Only after code and ledger corrections should a separately authorized model evaluation or refit establish new performance claims.
