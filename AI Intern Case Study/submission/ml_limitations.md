# ML Limitations

**Scope note:** this document was originally written for the separate, exploratory comparison in
`submission/ml_evaluation_report.md` (Isolation Forest vs. LOF vs. the rule-based baseline, neither
integrated). RouteSentry's production pipeline is now ML-driven (`ml/official_pipeline.py`, LOF,
calibrated contamination=0.02, `run.py`'s default) — see `docs/model_selection.md` and
`docs/ml_driven_pipeline.md` for that design. Everything below still applies to the production
pipeline too, plus the production-specific points in the new final section.

**Note on regeneration:** this file is rewritten from scratch by `scripts/run_ml_evaluation.py`.
This scope note and the "Production-pipeline-specific limitations" section at the end are hand-added
and not part of that script's template — re-running it will remove them; re-add both after.

## What this evaluation cannot prove

- **No real-world accuracy claim.** There are no ground-truth labels for which real route-weeks are
  genuinely anomalous (as opposed to statistically unusual) — this case study's inputs only provide 10
  free-text context notes, not a labeled anomaly set. Every real-data number in
  `submission/ml_evaluation_report.md` is an agreement/disagreement count, a score distribution, or a
  correlation — never a precision/recall/F1/accuracy figure.
- **The synthetic suite (section 8 of the evaluation report) is a wiring sanity check, not a
  performance benchmark.** Both models detect obviously-extreme, clearly-labeled synthetic cases
  reliably (100% / 100% detection rate for IF/LOF at
  +80% injected magnitude) — this says the pipeline is correctly wired, not
  that either model would catch a real, subtler anomaly at a comparable rate.
- **Neither model understands "rising."** Isolation Forest and LOF are undirected outlier detectors:
  15 of this run's 17 "ML flags, baseline doesn't" cases are **falling**
  cost weeks, which the brief explicitly excludes from scope ("a route's cost is rising..."). A
  route-type-specific or sign-constrained model would be needed to make the ML alerts directly
  comparable to the baseline's own semantics.
- **Small data regime.** 700 eligible route-week rows across 7 routes is a small dataset
  for Isolation Forest / LOF by the standards either algorithm is usually applied at; score stability
  across neighbouring folds was not separately stress-tested beyond the contamination sensitivity run
  in section 9.
- **Contamination is an assumption, not a measurement.** 5% was
  chosen as a documented, round policy value (`ml/config.py`), not fit to any known true anomaly rate —
  section 9's sensitivity table shows alert counts move considerably (roughly in proportion) as this
  assumption changes, for both models.
- **`prior_volatility_pct` with few prior weeks is noisy.** At the minimum eligible
  `n_prior_weeks=4`, a standard deviation of 4 points is itself a low-precision
  estimate of the route's true volatility.
- **LOF's `n_neighbors` is a heuristic**, not tuned: `clamp(train_size // 10, 5, 20)`.
  Early folds with a small training set use a small, less statistically stable neighbourhood.

## What would be needed to resolve these

A labeled set of real, independently-confirmed cost anomalies (ideally spanning both genuinely-justified
and genuinely-unjustified rises, at a scale of dozens-to-hundreds of examples) to measure actual
precision/recall per model, and a larger multi-year shipment history to evaluate score stability across
more chronological folds than the ~13 this dataset's ~104 weeks allow at an 8-week fold width.

## Production-pipeline-specific limitations (added with the ML-driven pivot)

- **The calibration grid search couldn't discriminate on detection rate.** Every candidate
  contamination (0.02–0.10) achieved 100% synthetic detection, so the selected value (0.02) was
  effectively chosen to minimize real-data false alerts on the validation period, not from a genuine
  detection/false-alarm trade-off. See `docs/model_selection.md` §2 for the full grid and a concrete
  fix (a tighter synthetic injection magnitude).
- **Coverage gap vs. the rule-based rule.** 84 of 728 route-weeks (11.5%) get no model assessment at
  all — the pooled, cross-route training set needed to fit LOF takes longer to accumulate than one
  route's own history, which the rule-based rule could use with as little as 1 prior week. These
  rows resolve to `"No"` with an explicit "not evaluated" reason, never a guessed normal.
- **Known false-negative vs. the brief's own example.** Mumbai-Pune's 2025-09-15 row (`vs_own_history
  = +9.2%`, `vs_similar_routes = +23.6%`) is flagged `"Yes"` in the brief's sample output and under
  `--mode rule_based`, but resolves to `"No"` under the calibrated ML-driven default (LOF score 2.096,
  below the 0.02-contamination boundary). This is a direct, documented, test-pinned consequence of
  calibrating for fewer real false alerts (`tests/test_ml_golden_divergence.py`), not a defect — but
  it is a real sensitivity cost, and it is the single clearest number to cite if asked "what did
  moving to ML actually cost you." Traced to the raw shipments and empirically re-checked against 5
  alternative contamination values and a per-route-type model variant — none within the honestly
  calibrated false-alert ceiling resolves it; see `docs/model_selection.md` §§3.1-3.2 for the full
  trace and why the only settings that *would* flip it were already independently rejected.
- **Model selection weighted one criterion over another by judgment, not by a single provably-correct
  rule.** LOF's much larger separation margin (Cohen's d 24.35 vs. 4.80) was weighted above Isolation
  Forest's smaller stability edge (CV 1.08 vs. 1.18) because the two models' instability appeared
  correlated with real anomaly clustering rather than independent model noise — a reasoned judgment
  call, not a mathematically forced conclusion, and documented as such in `docs/model_selection.md`.
