"""ML package: both the production ML-driven pipeline and a separate,
earlier experimental comparison. Read-only with respect to `routesentry/`
in both cases -- everything here imports from `routesentry/` to reuse its
leakage-free baseline definitions (vs_own_history, vs_similar_routes) and
its evidence-validation/verdict logic; nothing in `routesentry/` is ever
modified or written to by this package.

Two distinct things live here -- don't confuse them:

- `official_pipeline.py` + `signal.py` + `calibration.py`: the PRODUCTION
  pipeline. This IS wired into `run.py` (its default mode) and IS what
  writes `submission/output.csv` / `submission/evidence_ledger.csv`. See
  `docs/model_selection.md` and `docs/ml_driven_pipeline.md`.
- `pipeline.py` + `reporting.py` + `synthetic.py` + `chronological.py` +
  `models.py` + `features.py` + `config.py`: shared by both the production
  pipeline above AND a separate, earlier, exploratory comparison of
  Isolation Forest vs. LOF vs. the rule-based rule. That comparison is run
  independently via `scripts/run_ml_evaluation.py`, writes its own
  `submission/experimental_ml_*` files, and never touches
  `submission/output.csv`. See `submission/ml_evaluation_report.md`.
"""
