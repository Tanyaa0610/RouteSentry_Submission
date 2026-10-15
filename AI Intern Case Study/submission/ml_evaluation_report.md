# ML Evaluation Report — Isolation Forest & LOF (Parallel Evaluation Mode)

**SUPERSEDED, READ THIS FIRST:** this report's own §12 ("Final integration decision," below)
concluded that neither model should be integrated into the official output. That conclusion was
true when written and is no longer RouteSentry's actual state: a later, separate, explicit decision
made a calibrated LOF model (not Isolation Forest) the **default** production statistical signal —
see `docs/model_selection.md` for why, `docs/ml_driven_pipeline.md` for the design, and
`submission/evaluation_report.md` §§2-3 for current real-data results from both pipelines.
`submission/output.csv` is now the ML-driven pipeline's output, not the rule-based pipeline's
(preserved separately at `submission/benchmark_rule_based_output.csv`).

Everything below this notice — the experimental setup, the real-data comparison, the synthetic
suite, the sensitivity analysis — is unchanged and still accurate **as a description of that
earlier, separate comparison**; it is also the actual evidence the later model-selection decision in
`docs/model_selection.md` was built on (the Cohen's-d separation-margin analysis there, in
particular, is computed from this report's own synthetic suite). Only this document's own §12
conclusion is out of date.

**Note on regeneration:** this report, along with `submission/ml_feature_documentation.md` and
`submission/ml_limitations.md`, is rewritten from scratch by `scripts/run_ml_evaluation.py` every
time it runs. The SUPERSEDED notice above (and the corresponding note in `ml_limitations.md`) are
hand-added and are NOT part of that script's template — re-running the script will remove them, and
they must be re-added afterward. They are deliberately kept out of the generator itself so the
generator doesn't need to know about a later pivot that postdates it.

These two unsupervised models were originally run alongside, and separate from, the then-official
rule-based RouteSentry pipeline (`routesentry/`). They are an experimental comparison, not the
original pipeline replacement this section once described — see the "Final integration decision"
section below, read in light of the notice above.

## 1. Dataset and evaluation-window sizes

- Shipments loaded: 2940
- Route-week feature rows: 728
- Eligible rows (>= 4 prior weeks, peer average available): 700
- Ineligible (insufficient history): 28
- Excluded (eligible, but fold's training set below MIN_TRAIN_SIZE=40): 56
- Actually scored by both models: 644
- Chronological fold width: 8 weeks

## 2. Feature engineering and history requirements

Features (see `submission/ml_feature_documentation.md` for full definitions): `cost_per_tonne_km, vs_own_history_pct, vs_similar_routes_pct, wow_change_pct, prior_volatility_pct, n_prior_weeks, route_type_short, route_type_medium, route_type_long`.
A row is eligible only with >= 4 strictly-prior observed weeks for that route AND an available same-week peer average. All history-based features are computed from weeks strictly before the scored week; see `tests/test_ml_leakage.py`.

## 3. Chronological split methodology

Expanding window: at each fold boundary, TRAIN = all eligible rows with `week_of` strictly before the boundary; TEST = eligible rows in the next 8-week block. A `StandardScaler` is fit on TRAIN only, then applied to both TRAIN and TEST. Each model is fit on scaled TRAIN only and scores scaled TEST. Folds with fewer than 40 eligible training rows are not scored (reported as `insufficient_training_data`) rather than fit on too little data. See `ml/chronological.py`.

## 4. Model parameters and thresholds

- Random seed: 42 (both models, and the synthetic suite's sampling)
- Isolation Forest: `n_estimators=200`, `contamination=0.05`
- LOF: `novelty=True`, `contamination=0.05`, `n_neighbors = clamp(train_size // 10, 5, 20)` per fold (see `ml/models.py::choose_lof_k`)
- Score convention (both models): **higher score = more anomalous** (sklearn's own `decision_function`/`score_samples` are negated to enforce this consistently).
- Score distribution on real, scored rows (644 rows): IF p50/p90/p95/p99/max = -0.108 / -0.030 / 0.009 / 0.072 / 0.126; LOF p50/p90/p95/p99/max = 1.031 / 1.523 / 2.247 / 6.912 / 9.739

## 5. Alert counts (real data, scored rows only)

| Method | Alerts | Rate |
|---|---|---|
| Rule-based baseline (statistical signal, pre-evidence) | 26 | 4.0% |
| Isolation Forest | 39 | 6.1% |
| LOF | 37 | 5.7% |

Note: the baseline's own final CSV verdict (`Yes`/`No (justified)`/`No`) additionally runs evidence validation on top of this statistical signal; the comparison here is against the pre-evidence statistical signal only, since Isolation Forest/LOF never see note text (see section 6).

## 6. Agreement and disagreement

- IF <-> LOF agreement rate: 97.8% (630/644)
- IF <-> baseline agreement rate: 95.2% (613/644)
- LOF <-> baseline agreement rate: 94.6% (609/644)
- Pearson correlation, IF score vs. baseline's vs_own_history_pct: 0.226; vs. vs_similar_routes_pct: 0.279
- Pearson correlation, LOF score vs. baseline's vs_own_history_pct: 0.190; vs. vs_similar_routes_pct: 0.102

Full three-way breakdown (baseline_signal, if_prediction, lof_prediction) -> count:

| baseline | IF | LOF | count |
|---|---|---|---|
| False | False | False | 590 |
| False | False | True | 6 |
| False | True | False | 5 |
| False | True | True | 17 |
| True | False | False | 9 |
| True | True | False | 3 |
| True | True | True | 14 |

## 7. Representative disagreement cases

**Baseline flags, neither ML model does** (9 cases; showing up to 3):

- Mumbai-Pune 2024-06-10 -- vs_own_history=+3.2%, vs_similar_routes=+16.6%, wow_change=+4.6%, if_score=-0.027, lof_score=1.060, baseline_verdict=Yes
- Mumbai-Pune 2024-10-28 -- vs_own_history=+3.1%, vs_similar_routes=+15.1%, wow_change=+4.6%, if_score=-0.062, lof_score=1.008, baseline_verdict=Yes
- Mumbai-Pune 2024-12-16 -- vs_own_history=+3.5%, vs_similar_routes=+15.3%, wow_change=+3.5%, if_score=-0.000, lof_score=1.137, baseline_verdict=Yes

**Both ML models flag, baseline doesn't** (17 cases: 2 rising, 15 falling; showing up to 3 of each):

*Falling-cost cases (out of scope for the brief's "rising and unjustified" framing):*
- Delhi-Jaipur 2024-11-25 -- vs_own_history=-8.0%, vs_similar_routes=-10.9%, wow_change=-25.3%, if_score=0.064, lof_score=9.739, baseline_verdict=No
- Delhi-Jaipur 2024-12-02 -- vs_own_history=-8.5%, vs_similar_routes=-11.5%, wow_change=-0.4%, if_score=0.028, lof_score=8.870, baseline_verdict=No
- Delhi-Jaipur 2024-12-09 -- vs_own_history=-3.0%, vs_similar_routes=-6.0%, wow_change=+5.8%, if_score=0.013, lof_score=8.884, baseline_verdict=No

*Rising-cost cases (the operationally relevant subset):*
- Ahmedabad-Mumbai 2025-03-10 -- vs_own_history=+1.0%, vs_similar_routes=-15.9%, wow_change=+1.8%, if_score=0.001, lof_score=3.809, baseline_verdict=No
- Chennai-Bangalore 2025-05-05 -- vs_own_history=+1.5%, vs_similar_routes=+10.8%, wow_change=+9.6%, if_score=0.009, lof_score=2.814, baseline_verdict=No

Interpretation: the large majority of "ML-only" alerts are **falling**-cost weeks (`vs_own_history_pct <= 0`) that an undirected outlier detector flags as statistically unusual in either direction, while the rule-based baseline deliberately only considers a **rising** cost an anomaly (per the brief: "cost is rising and looks out of the ordinary"). The baseline-only misses tend to be modest, peer-comparison-driven rises on a route that has already been drifting upward for a while — by the time of scoring, the expanding-window models have absorbed that drift into their learned "normal", so a continuation of the trend no longer looks like an outlier to them, whereas the baseline's fixed +15pp-vs-peers rule still fires on every qualifying week independently.

## 8. Synthetic controlled evaluation (separate from all of the above)

**This section uses synthetic, labeled data only — it is not a measurement of real-world accuracy.** 30 synthetic anomalies were built from real row templates with `vs_own_history_pct`, `vs_similar_routes_pct`, and `wow_change_pct` all overridden to +80% (far beyond the real data's observed range — see `docs/threshold_analysis.md`), mixed with a held-out set of 120 real normal rows never used in training (580 real rows).

| Model | Synthetic detection rate | False-alert rate on real normal holdout |
|---|---|---|
| Isolation Forest | 100.0% | 2.5% |
| LOF | 100.0% | 3.3% |

Mean scores: IF normal=-0.145 vs. synthetic=0.100; LOF normal=1.144 vs. synthetic=10.501.

Both models detect this obviously-extreme, clearly-labeled synthetic case reliably. This confirms the pipeline wiring (features -> scaler -> model -> score direction) behaves as intended; it says nothing about how well either model would catch a real, subtler anomaly.

## 9. Sensitivity to contamination

Full chronological evaluation re-run at each contamination value (same seed, same folds):

| contamination | IF alerts | LOF alerts |
|---|---|---|
| 0.03 | 17 | 30 |
| 0.05 | 39 | 37 |
| 0.1 | 66 | 71 |

Alert counts scale roughly with the contamination parameter for both models, as expected — neither model's alert set is a fixed, parameter-insensitive set of "true" anomalies. This is exactly why section 10 does not pick a winner by alert count.

## 10. Runtime and reproducibility

Full pipeline run (chronological eval + sensitivity re-runs + synthetic suite): 8.7s on this machine.
Reproducibility: every model fit uses `RANDOM_SEED` (42); `StandardScaler`, `IsolationForest`, and `LocalOutlierFactor` are all deterministic given a fixed seed and fixed input order (feature rows are explicitly sorted by `(week_of, route)` before any split). See `tests/test_ml_reproducibility.py`, which asserts two full runs in-process produce identical scores, and `scripts/check_reproducibility.py --ml` for a 3-process check analogous to the rule-based baseline's own.

## 11. Limitations and whether either model adds meaningful value

See `submission/ml_limitations.md` for the full discussion. Summary:
- Of 17 "ML flags, baseline doesn't" cases, only 2 are even rising-cost weeks at all, and those are marginal (a percent or two above zero) — essentially no clear case was found where an ML model caught a meaningful rising-and-unjustified anomaly that the deterministic baseline's own two prescribed comparisons missed.
- Of 9 cases the baseline alone flags, the ML models' lower scores are explainable (adaptive models absorbing a sustained trend) rather than clearly "wrong" — this is a difference in philosophy (fixed rule vs. adaptive window), not evidence either is more correct.
- **No ground-truth labels exist for real route-weeks**, so no accuracy/precision/recall claim is made anywhere in this report for real data — only agreement/disagreement counts, score distributions, and the clearly-separate synthetic sanity check.

## 12. Final integration decision (AT THE TIME THIS REPORT WAS WRITTEN — now superseded, see notice at top)

**Neither model is integrated into the official flagging path.** The measured results do not establish a reliable improvement over the deterministic baseline for this specific task ("a route's cost is rising in a way that doesn't look justified"): the ML models are undirected (they flag unusual cost in either direction, not just rises) and, when restricted to their rising-cost alerts, essentially reproduce the baseline's own signal rather than add new, verifiably-missed cases. `submission/output.csv` remains exactly the deterministic rule-based pipeline's output, unchanged by any of this work.
What would change this: a labeled dataset of real route-weeks independently confirmed as genuine cost problems (not just statistically unusual), large enough to measure real precision/recall per model — none exists in this case study's inputs.

**What actually happened next:** a separate, later, explicit decision made LOF the production default
anyway, with the "undirected model" problem solved structurally (a hard, non-ML "must be rising" gate
in `ml/signal.py`) rather than by picking a different model, and with an honest accounting of what
that costs (fewer alerts, one brief-example divergence) rather than a claim of improvement. See
`docs/model_selection.md` for the real decision and its evidence.
