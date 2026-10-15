# Evaluation Report

**The production pipeline is now ML-driven** (`run.py`'s default — a calibrated LOF model decides
the statistical signal; see `docs/model_selection.md`). The original rule-based pipeline remains
fully intact and selectable (`run.py --mode rule_based`), and its output is preserved as a benchmark
artifact. This report covers both, clearly labeled, since both are real, tested, running code paths.

## 1. Data processed (shared by both pipelines)

- `shipment_records.csv`: 2,940 rows read, **0 skipped** (all rows passed
  validation — no non-positive quantity/distance/cost, no invalid dates, no
  unrecognised `route_type`, no blank origin/destination).
- `context_notes.csv`: 10 notes read, 0 skipped.
- 7 distinct routes, each with a single consistent `route_type`:
  - Short: Mumbai-Pune, Delhi-Jaipur
  - Medium: Ahmedabad-Mumbai, Chennai-Bangalore, Kolkata-Bhubaneswar
  - Long: Mumbai-Delhi, Delhi-Chennai
- Date range: 2024-01-01 to 2025-12-28 (104 ISO weeks).
- 0 duplicate `shipment_id` values found in the real data.

## 2. ML-driven pipeline output (`submission/output.csv`, the current default)

- **728** route-week rows (same universe as the rule-based pipeline — every
  (route, week_of) pair with at least one shipment).
- Statistical-signal eligibility: 644 rows scored by the calibrated LOF model, 28 rows ineligible
  (insufficient per-route history), 56 rows excluded (insufficient pooled cross-route training data
  for that fold) — see `docs/ml_driven_pipeline.md`.
- Verdicts: **7 "Yes"** (unexplained anomalies), **3 "No (justified)"** (explained anomalies),
  **718 "No"** (not anomalous or not assessed).
- Notes actually cited: `N001` (×2), `N002` (×1) — the same two notes the rule-based pipeline cites,
  since the evidence-validation layer is identical code in both pipelines.
- Calibrated LOF contamination: **0.02**, selected via a chronological train/validation grid search
  (`docs/model_selection.md`), never touching the held-out test period.

## 3. Rule-based pipeline output (`submission/benchmark_rule_based_output.csv`, preserved)

- Same 728 rows. Verdicts: **23 "Yes"**, **3 "No (justified)"** (same 2 notes, same rows),
  **702 "No"**. Flat +15 percentage-point threshold, documented in `docs/threshold_analysis.md`.

## 4. Unit tests

Command: `python3 -m pytest tests/ -v`

**136 / 136 passed**, covering both pipelines:

| Area | File | Count |
|---|---|---|
| Week boundaries, date-range overlap | `test_dates.py` | 8 |
| Data validation (missing cols, bad rows, duplicates) | `test_data_io.py` | 11 |
| Aggregation + both baselines (ratio-of-sums, no leakage, peer exclusion) | `test_weekly.py` | 9 |
| Rule-based statistical anomaly rule | `test_anomaly.py` | 6 |
| Evidence validation / hallucination guardrails (shared by both pipelines) | `test_evidence.py` | 16 |
| Verdict combination logic (shared) | `test_verdict.py` | 7 |
| Rule-based output schema contract | `test_output_contract.py` | 8 |
| Rule-based golden examples from the brief | `test_golden_examples.py` | 3 |
| Rule-based in-process reproducibility | `test_reproducibility.py` | 3 |
| Regression: rule-based pipeline unchanged by the ML pivot | `test_baseline_preserved.py` | 3 |
| ML feature construction, leakage-free | `test_ml_features.py` | 8 |
| ML model wrappers, score direction | `test_ml_models.py` | 6 |
| ML chronological walk-forward evaluation | `test_ml_chronological.py` | 5 |
| ML leakage (direct train/test boundary checks) | `test_ml_leakage.py` | 3 |
| ML synthetic controlled suite | `test_ml_synthetic.py` | 5 |
| ML output contract (experimental comparison) | `test_ml_output_contract.py` | 7 |
| ML in-process reproducibility (experimental comparison) | `test_ml_reproducibility.py` | 3 |
| ML signal (production): rising gate, calibrated score | `test_ml_signal.py` | 7 |
| ML calibration: chronological split, grid search | `test_ml_calibration.py` | 6 |
| ML official (production) pipeline integration | `test_ml_official_pipeline.py` | 9 |
| ML golden-example divergence (documented, test-pinned) | `test_ml_golden_divergence.py` | 3 |

## 5. Golden examples (brief's own worked examples)

**Rule-based (`--mode rule_based`):** all 3 rows reproduce exactly on `cost_per_tonne_km`,
`vs_own_history`, `vs_similar_routes`, `flagged`, and `matched_note_id`. See
`tests/test_golden_examples.py`.

**ML-driven (default):** 2 of 3 reproduce exactly (Delhi-Jaipur, Ahmedabad-Mumbai); the third
(Mumbai-Pune, 2025-09-15) now reads `"No"` instead of `"Yes"` — a direct, documented consequence of
the calibrated threshold being more conservative. All three numeric fields still match exactly in
every row. See `tests/test_ml_golden_divergence.py` and `docs/model_selection.md`.

## 6. Adversarial evaluation (evidence layer — identical code in both pipelines)

14/14 synthetic hallucination/misleading-retrieval cases pass — see
`submission/adversarial_evaluation.md`. This is unaffected by the ML pivot since
`routesentry/evidence.py` was not modified.

## 7. Reproducibility

Both pipelines checked by running the actual CLI 3 times in 3 separate processes and hashing the
output:

- Rule-based: byte-identical (`docs/reproducibility_report.md`).
- ML-driven: byte-identical (`docs/ml_driven_reproducibility_report.md`) — the model fits are seeded
  (`RANDOM_SEED=42`) and the feature rows are explicitly sorted before every split, so this is a
  verified result, not an assumption that seeding alone would guarantee.

## 8. Why this is enough to trust the submission

1. The numeric core (aggregation + both baselines) — shared by both pipelines — is tested against
   hand-computed expected values, not just against the real dataset's output.
2. The evidence layer is tested against the exact failure modes the brief calls out by name, and is
   literally the same code in both pipelines — nothing about switching the statistical-signal source
   touches it.
3. The ML-driven threshold was calibrated on a train/validation split that never saw the held-out
   test period, and never saw the brief's 3 sample rows as a target to fit against.
4. Reproducibility for both pipelines is checked by actually running the CLI 3 times in 3 separate
   processes and hashing the output.
5. Where the two pipelines disagree, the disagreement is quantified and explained
   (`docs/model_selection.md`), not hidden.

## 9. Known limitations

**Shared by both pipelines:**
- `classify_direction`'s keyword rules were built by reading the 10 real notes; a materially
  different notes corpus would need new keyword patterns (an explicit, auditable edit).
- No LLM or stretch-goal natural-language Q&A interface is implemented.

**Rule-based only:**
- The flat 15-percentage-point threshold does not distinguish a sustained trend from a one-off spike
  (`docs/threshold_analysis.md`) — the Mumbai-Pune route is flagged "Yes" repeatedly across much of
  2024–2025 because it drifts upward against its peers, evaluated independently each week.

**ML-driven only:**
- Covers fewer rows (84 "not evaluated" vs. 7) and raises fewer statistical signals (10 vs. 26) than
  the rule-based rule — see `submission/ml_limitations.md` for the full, honest accounting, including
  the one brief-example divergence this causes.
