# 10-Minute Walkthrough — Speaker Notes

## 0–1 min: Problem and motivation

"Freight cost per route can quietly creep up — sometimes for a real reason (fuel, tolls, weather,
festivals), sometimes not. RouteSentry watches every route week by week, decides whether a cost
rise looks statistically out of the ordinary, and — critically — checks a small notes corpus for a
genuine reason *before* calling it unexplained. If it can't find a real reason, it says so instead
of guessing. The production pipeline now makes that first decision with a trained anomaly-detection
model rather than a fixed percentage rule — the original rule-based version is still there,
selectable, and is what I validated the model against."

## 1–3 min: Architecture and data flow

Walk the pipeline diagram in `README.md` §6: load+validate → weekly aggregation (route × week →
cost per tonne-km, same exact formulas either way) → feature construction → a calibrated LOF model
decides the statistical signal (gated so it can never flag a *falling* cost) → TF-IDF candidate
retrieval over the 10 notes → deterministic evidence validation → final verdict → two CSVs
(`output.csv` for grading, `evidence_ledger.csv` for full audit).

Emphasize: everything from note retrieval onward is the *same code* in both the ML-driven and
rule-based pipelines — I only replaced the statistical-signal step, which is why nothing about
evidence validation needed to be re-proven when I made this change.

## 3–5 min: Cost calculations and baselines (unchanged by the ML pivot)

- Formula: `Σfreight_cost_inr / Σ(quantity_tonnes × distance_km)` per route-week — a ratio of sums
  (see `test_weekly.py::test_ratio_of_sums_not_average_of_ratios`).
- `vs_own_history`: trailing 8-week average of *prior* weeks only — no look-ahead, and these are the
  same numbers fed to the ML model as features.
- `vs_similar_routes`: same-week average across the route's own peer type, excluding itself.
- Real example: **Ahmedabad-Mumbai, week of 2025-01-20** — cost 3.29/tonne-km, +29.5% vs its own
  history, +22.5% vs similar Medium routes that week.

## 5–6 min: Why LOF, and how the threshold was calibrated

- Isolation Forest and LOF were both already built for an earlier side-by-side comparison. Choosing
  one for production needed a real head-to-head: LOF's separation between normal and synthetic
  labeled-anomalous scores (Cohen's d = 24.35) was ~5x larger than Isolation Forest's (4.80) — show
  `docs/model_selection.md`.
- LOF's threshold (`contamination`) was calibrated on a chronological train (420 rows) /
  validation (140 rows) split of real data, with the final 140-row test period held out and never
  touched. Selected value: **0.02** — the most conservative candidate that kept the real false-alert
  rate on validation under 5%.
- Be upfront about the limitation: every candidate hit 100% synthetic detection, so the grid search
  effectively picked the lowest-false-alert option rather than a genuine precision/recall trade-off.

## 6–8 min: Evidence retrieval, audit trail, and hallucination prevention

- Show the real justified row: Ahmedabad-Mumbai 2025-01-20 matches `N002` ("a regional festival
  week... temporary surcharge... Ahmedabad-Mumbai corridor"), dated exactly that week — identical
  under both pipelines, since the evidence layer never changed. Open `evidence_ledger.csv` and show
  the full candidate list, including rejected notes and why.
- Show a real unexplained row: **Delhi-Jaipur, week of 2024-11-11** — flagged `"Yes"` under both
  pipelines, `matched_note_id` blank. The closest note (road-condition improvement) is correctly
  rejected as not describing a cost increase.
- Name the guardrails: a note saying "not significantly affected" or "improved road conditions" can
  never become a justification, even as the closest semantic match. A global "All Routes" note that
  excludes the affected routes is rejected despite its `applies_to` field.
- The "rising" gate is enforced structurally, not left to the model: LOF is undirected and would
  otherwise flag unusually *low*-cost weeks too — which the brief doesn't ask for.

## 8–9 min: Evaluation, tests, and reproducibility

- 136 unit tests (111 rule-based/shared + 25 new for the ML-driven path: the rising gate overriding
  the model, leakage-free calibration, the detector actually being invoked, the documented golden-row
  divergence). 14/14 adversarial hallucination cases pass (evidence layer, unchanged).
- Both pipelines reproducible across 3 separate processes, byte-identical
  (`docs/reproducibility_report.md`, `docs/ml_driven_reproducibility_report.md`) — the ML path fixes
  a random seed for every model fit rather than being seedless, but is equally deterministic.
- $0 / 0 tokens / 0 LLM calls either way — LOF is a classical scikit-learn model trained locally, not
  an LLM.

## 9–10 min: What changed, what it cost, and what's next

- The ML-driven pipeline raises far fewer statistical signals than the old rule (10 vs. 26) because
  the calibration explicitly optimized for fewer real false alerts — a genuine, disclosed trade-off,
  not a free win.
- **Concrete cost:** one of the brief's own 3 example rows (Mumbai-Pune, 2025-09-15) is no longer
  flagged under the calibrated threshold. Numbers match exactly; only the verdict differs. Pinned
  down by a test so it can't silently change again.
- The original rule-based pipeline isn't gone — `run.py --mode rule_based` runs it unchanged, and its
  output is preserved as `submission/benchmark_rule_based_output.csv`.
- Future: a labeled set of real, confirmed anomalies would turn both the model-selection comparison
  and the calibration grid search into real accuracy measurements instead of synthetic-proxy ones.
