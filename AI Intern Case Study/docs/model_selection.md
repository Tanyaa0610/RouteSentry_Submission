# Model Selection and Threshold Calibration — ML-Driven Primary Pipeline

This document records the actual, computed evidence behind two decisions in the ML-driven pipeline
(`ml/official_pipeline.py`, now `run.py`'s default): **which model** (Isolation Forest vs. LOF)
became the primary production detector, and **how its decision threshold** was set. Both were
decided from real, run, numbers — not convenience or a round default.

## 1. Why LOF over Isolation Forest

Both models were already built and evaluated side by side in the earlier experimental comparison
(`submission/ml_evaluation_report.md`). Picking a primary required a head-to-head criterion beyond
"both roughly agree" — two were computed:

### A. Standardized separation margin (Cohen's d) on the labeled synthetic stress test

The only labeled evaluation available (see `submission/ml_limitations.md` for why no real labels
exist) is the synthetic suite: 30 injected anomalies (+80pp on all three comparison features) vs. a
held-out set of real "normal" rows. Rather than just compare detection/false-alert rates at one
contamination value, the actual separation between the two score distributions was measured as an
effect size — how many standard deviations apart the normal and synthetic-anomaly score
distributions are:

| Model | Normal mean | Normal std | Synthetic mean | **Cohen's d** |
|---|---|---|---|---|
| Isolation Forest | -0.1454 | 0.0511 | 0.0995 | **4.80** |
| LOF | 1.1440 | 0.3842 | 10.5008 | **24.35** |

LOF's separation is roughly **5x larger**. This matters more than a single detection-rate number at
one arbitrary threshold: it means LOF's score distribution leaves far more room to set a threshold
that catches real anomalies without also catching normal weeks, i.e. it is the fundamentally better
discriminator in this feature space, not just luckier at one setting.

### B. Fold-to-fold alert-rate stability on real data

Computed across the 12 real chronological folds (`ml/chronological.py`'s own walk-forward
evaluation, pre-calibration, both models at contamination=0.05 for a fair comparison):

| Model | Mean alert rate | Std | Coefficient of variation |
|---|---|---|---|
| Isolation Forest | 0.064 | 0.069 | **1.078** |
| LOF | 0.061 | 0.072 | **1.180** |

Isolation Forest is marginally more stable here (~9% lower relative variance). However, the two
models' per-fold rates move almost in lockstep (`[0, 0, 0, 0, .143, .125, .196, .054, 0, .018, .089,
.143]` for IF vs. `[0, 0, .018, 0, .125, .143, .214, .036, 0, 0, .054, .143]` for LOF) — both are
near-zero in the same early folds and spike in the same later folds. That pattern looks like real
temporal clustering of anomalous weeks in the data (e.g. the Chennai-Bangalore flood period, the
Delhi-Jaipur late-2024 stretch), not model-specific instability, which weakens how much weight this
criterion should carry relative to (A).

### Decision

**LOF was selected as the primary model.** The ~9% stability edge for Isolation Forest is a much
smaller, more ambiguous signal than LOF's 5x larger, unambiguous separation margin on the only
labeled test available. Isolation Forest remains fully built, tested, and available
(`ml/models.py::fit_isolation_forest`) for anyone who wants to re-run this comparison or disagrees
with the weighting above — nothing about this choice is hard-coded elsewhere; it only affects which
score `ml/official_pipeline.py` reads.

## 2. Threshold calibration (contamination)

LOF's `contamination` parameter sets the fraction of its training data treated as anomalous, which
in turn sets its internal decision boundary. This cannot be calibrated against real labels — none
exist — so it is calibrated against a **chronological train/validation split of real data**, with
the **final test period held out and never touched** during calibration.

### Split (by calendar week, not by row count, so no route's history straddles a boundary)

| Period | Rows | Week range |
|---|---|---|
| Train | 420 | 2024-01-29 .. 2025-03-17 |
| Validation | 140 | 2025-03-24 .. 2025-08-04 |
| Held-out test | 140 | 2025-08-11 .. 2025-12-22 |

### Grid search (`ml/calibration.py`)

A LOF model is fit on Train at each candidate contamination, then scored against (a) genuine
Validation rows (measuring the real false-alert rate) and (b) 30 synthetic anomalies injected into
Validation-period row templates (measuring synthetic detection rate):

| contamination | synthetic detection | validation false-alert rate |
|---|---|---|
| 0.02 | 100% | **1.4%** |
| 0.03 | 100% | 1.4% |
| 0.05 | 100% | 2.1% |
| 0.08 | 100% | 6.4% |
| 0.10 | 100% | 7.9% |

**Selection rule:** among candidates with a validation false-alert rate ≤ 5%, pick the highest
synthetic detection rate, tie-broken toward the lowest (most conservative) contamination.

**Selected: contamination = 0.02.**

### Limitation of this calibration — stated plainly

Every candidate achieved 100% synthetic detection. The +80pp injected magnitude (chosen in the
earlier experimental work to be obviously, unambiguously extreme — see
`docs/threshold_analysis.md`) turned out to be too easy to discriminate between contamination
values on that axis alone. In practice, this grid search reduced to "pick the lowest contamination
whose real-data false-alert rate stays under the ceiling" — a legitimate, conservative, and fully
data-driven rule, but one that didn't get to use the synthetic criterion's full discriminating power.
A tighter, more realistic synthetic magnitude (e.g. +20-30pp, closer to the real data's own ~99th
percentile per `docs/threshold_analysis.md`) would let the grid search actually trade off detection
vs. false alarms, and is a concrete follow-up rather than a correctness issue with the result
reported here.

## 3. Consequence: divergence from the brief's own 3 worked examples

The brief's `sample_output_format_v2.csv` gives 3 example rows. Under the rule-based pipeline (still
fully intact, `run.py --mode rule_based`) all 3 reproduce exactly
(`tests/test_golden_examples.py`). Under the calibrated ML-driven pipeline, 2 of 3 still match
exactly; the third does not:

| Route / week | Rule-based verdict | ML-driven verdict | Why |
|---|---|---|---|
| Delhi-Jaipur, 2024-11-11 | Yes | Yes | LOF score 6.668, well above the 0.02-contamination boundary |
| Ahmedabad-Mumbai, 2025-01-20 | No (justified), N002 | No (justified), N002 | LOF score 6.484, well above boundary; evidence layer unchanged |
| Mumbai-Pune, 2025-09-15 | Yes | **No** | LOF score 2.096, below the calibrated boundary |

All three numeric fields (`cost_per_tonne_km`, `vs_own_history`, `vs_similar_routes`) are unchanged
in every row — only the verdict on the third row differs. This is a real, intended consequence of
how the two pipelines define "unusual" — not a bug — and is pinned down by
`tests/test_ml_golden_divergence.py` so it can never silently change back without a test failing and
forcing this document to be updated.

### 3.1 Full root-cause trace for Mumbai-Pune, 2025-09-15 (re-verified 2026-10-15)

Before accepting this as an inherent trade-off, every stage of the pipeline was traced by hand and
every plausible fix was tried empirically. Nothing below is asserted without having actually been run.

**Data and aggregation — confirmed correct.** The 4 raw shipments in this route-week
(`SHP02162..SHP02165`) sum to a hand-computed `cost_per_tonne_km` of 3.98080998..., identical to the
pipeline's own `WeeklyMetric.cost_per_tonne_km` to full float precision. `vs_own_history_pct` (9.25%,
8 full prior weeks) and `vs_similar_routes_pct` (23.63%, against Mumbai-Pune's one Short-route peer,
Delhi-Jaipur) also match exactly. **No aggregation or feature-computation bug exists.**

**Why the rule-based pipeline flags it.** `vs_similar_routes_pct` (23.63%) clears the flat +15pp bar
on its own, and the route is rising — `routesentry/anomaly.py`'s rule fires exactly as designed.

**Why the calibrated LOF model doesn't.** This route-week falls in chronological fold 10 (test
window 2025-08-11..2025-10-06, trained on the 560 eligible route-weeks strictly before it). Within
that fold's own 56 test rows, Mumbai-Pune's LOF score (2.0963) is the **highest of all 56** — by the
test fold's own internal ranking, this is the single most unusual row. It is still not flagged,
because LOF's decision boundary is fit from the **training** set's score distribution, not the test
fold's: at `contamination=0.02`, the boundary sits at approximately the training set's own 98th
percentile (score ≈3.08, with 9 of 560 training rows exceeding it — close to the expected
0.02 × 560 ≈ 11.2). Mumbai-Pune's test score of 2.0963 ranks at only the **95.2nd percentile of that
training distribution** — 27 of the 560 training route-weeks scored higher. In plain terms: by
August 2025 the model has already seen enough route-weeks at or above this level (including
Mumbai-Pune's own earlier weeks during its documented multi-month upward drift — see
`docs/threshold_analysis.md`) that a repeat of the same magnitude no longer looks exceptional to it,
even though it's still comfortably above the rule-based pipeline's fixed, history-independent +15pp
bar.

**Alternatives actually tried, with real numbers, before ruling out a fix:**

| Alternative tried | Result | Why rejected |
|---|---|---|
| Contamination 0.03 (within the ≤5% honest false-alert ceiling) | Mumbai-Pune still **not** flagged (fold alerts: 0/56) | Doesn't change the outcome |
| Contamination 0.05 (within the ≤5% ceiling; the original grid search's next candidate) | Mumbai-Pune still **not** flagged (fold alerts: 0/56) | Doesn't change the outcome |
| Contamination 0.08 | Mumbai-Pune **is** flagged (fold alerts: 2/56) | But this contamination's own validation false-alert rate is 6.4% — it already failed the calibration's own ≤5% acceptance ceiling in §2's grid search. Adopting it here, after already knowing it fails that test, would mean picking the threshold *because* it fixes this one row, which is exactly the sample-specific reverse-engineering this investigation was required to avoid. |
| Contamination 0.10 | Mumbai-Pune **is** flagged (fold alerts: 3/56) | Same objection, worse (7.9% false-alert rate) |
| Fit LOF per `route_type` instead of pooling all 7 routes (Short routes only: Mumbai-Pune + Delhi-Jaipur, 160 training rows) | Mumbai-Pune's score is *higher* in this narrower comparison (2.4988) but **still not flagged** at contamination 0.02, 0.03, or 0.05 | Rules out "pooling across route types dilutes the signal" as the cause — the same local-density judgment holds even compared only against its one true peer route |

**Conclusion: no principled, globally-applied adjustment within the honestly-calibrated bounds
resolves this divergence.** The only settings that do (contamination ≥0.08) were already excluded by
the calibration's own, independently-derived false-alert ceiling before this specific row was ever
examined — using them now would be adopting a worse-calibrated threshold specifically because it
produces the expected answer on one known example, which this investigation was explicitly required
not to do. The existing calibrated behavior (contamination=0.02) is therefore preserved unchanged.

### 3.2 Why this is a defensible trade-off, not a shortcoming to hide

The rule-based and ML-driven pipelines encode genuinely different definitions of "unusual," and
Mumbai-Pune 2025-09-15 is the clearest case where that difference actually bites:

- **Rule-based:** compares only against this route's own trailing 8 weeks and this week's one peer —
  a short, non-adaptive memory. A sustained drift gets flagged every single week it continues,
  because the 8-week trailing average itself keeps drifting upward too slowly to catch up.
- **ML-driven:** compares against the full multivariate distribution of everything the model has
  seen so far across all routes — a long, adaptive memory. Once a drift has been underway long
  enough, the model has "seen enough of it" to stop treating a continuation as exceptional.

Neither comparison is more correct in the abstract; they answer different questions ("is this
different from this route's own recent past?" vs. "is this different from everything observed so
far?"). The brief's own 3 examples happen to validate the rule-based definition exactly (unsurprising
— the rule-based rule is what they were almost certainly generated against), which is precisely why
`run.py --mode rule_based` remains available, unchanged, and still passes all 3 exactly.

This divergence is also the clearest illustration of the trade-off this whole pivot represents: the
ML-driven pipeline raises far fewer, more conservative alerts (10 statistically-flagged route-weeks
total vs. the rule-based pipeline's 26) because its threshold was calibrated specifically to minimize
false alerts on held-out real data — at the cost of also missing a case the brief itself considered
worth flagging.
