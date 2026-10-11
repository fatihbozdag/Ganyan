# Audit repair plan

Objective: improve the reliability of winner prediction and establish an honest,
repeatable measure of accuracy before promoting new models.

1. **Measurement and settlement (F01–F04, F12):** shared payout units, complete
   Plase settlement, neutral score ties, strict final-cohort/provenance gates.
2. **Data and features (F05, F07, F13–F15):** horse identity, race/entry snapshots,
   physical draw separation, pre-race availability boundaries, shared field
   construction, and historical Bayesian feature integrity.
3. **Prediction and model lifecycle (F06, F08, F11):** approved-head manifest,
   isolated candidates, verified atomic promotion, immutable pre-race prediction
   records and coupons, explicit pool windows, GUI integration.
4. **Operations and access (F09–F10, F16):** truthful failures, Istanbul clocks,
   current predictions, authenticated/CSRF-protected writes, strict health checks.
5. **Verification and delivery:** desired-behavior regression tests, isolated DB
   tests, migration/package smoke checks, CI, corrected runbooks and final status.

Acceptance: each finding has a regression test or explicit verification evidence;
the full suite passes; live artifacts remain intact. Production data migration,
ledger repair, model fitting/evaluation, and promotion are separate rollout actions.
No accuracy uplift will be claimed without a new held-out evaluation.

Progress and final commands/results will be recorded in `REPAIR_STATUS.md`.
