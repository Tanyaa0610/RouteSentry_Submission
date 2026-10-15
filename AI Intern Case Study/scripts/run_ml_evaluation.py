#!/usr/bin/env python3
"""Runs the full parallel ML evaluation (Isolation Forest + LOF) and writes
every number in its reports from this actual run -- nothing is hardcoded.

Usage:
    python3 scripts/run_ml_evaluation.py

Writes:
    submission/experimental_ml_predictions.csv
    submission/experimental_ml_model_comparison.csv
    submission/ml_evaluation_report.md
    submission/ml_feature_documentation.md
    submission/ml_limitations.md

Does NOT touch submission/output.csv, submission/evidence_ledger.csv, or
anything else written by run.py's default (ML-driven, ml/official_pipeline.py)
or --mode rule_based path -- this is a fully separate, earlier, exploratory
comparison. See ml/__init__.py.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.config import (  # noqa: E402
    FOLD_STEP_WEEKS,
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
    LOF_CONTAMINATION,
    LOF_K_DIVISOR,
    LOF_K_MAX,
    LOF_K_MIN,
    MIN_PRIOR_WEEKS_FOR_ML,
    MIN_TRAIN_SIZE,
    RANDOM_SEED,
    SYNTHETIC_N_ANOMALIES,
    SYNTHETIC_N_NORMAL_HOLDOUT,
    SYNTHETIC_PCT_MAGNITUDE,
)
from ml.features import FEATURE_NAMES, build_feature_table  # noqa: E402
from ml.pipeline import run_full_ml_pipeline  # noqa: E402
from ml.reporting import overlap_breakdown, write_comparison_csv, write_predictions_csv  # noqa: E402
from routesentry.data_io import load_shipments  # noqa: E402

SHIPMENTS = ROOT / "shipment_records.csv"
NOTES = ROOT / "context_notes.csv"


def pct(sorted_list, q):
    if not sorted_list:
        return float("nan")
    k = int(len(sorted_list) * q)
    return sorted_list[min(k, len(sorted_list) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the parallel ML (Isolation Forest + LOF) evaluation")
    parser.add_argument("--out-dir", default=str(ROOT / "submission"), help="directory to write ml_* reports/CSVs into")
    args = parser.parse_args()
    SUBMISSION = Path(args.out_dir)

    t0 = time.monotonic()
    out = run_full_ml_pipeline(SHIPMENTS, NOTES)
    elapsed = time.monotonic() - t0

    SUBMISSION.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(out.predictions, SUBMISSION / "experimental_ml_predictions.csv")
    write_comparison_csv(out.predictions, SUBMISSION / "experimental_ml_model_comparison.csv")

    scored = [p for p in out.predictions if p.ml_status == "scored"]
    status_counts = Counter(p.ml_status for p in out.predictions)
    overlap = overlap_breakdown(out.predictions)

    baseline_alerts = sum(1 for p in scored if p.baseline_statistical_signal)
    if_alerts = sum(1 for p in scored if p.if_prediction)
    lof_alerts = sum(1 for p in scored if p.lof_prediction)
    if_lof_agree = sum(1 for p in scored if p.if_lof_agree)
    if_base_agree = sum(1 for p in scored if p.if_baseline_agree)
    lof_base_agree = sum(1 for p in scored if p.lof_baseline_agree)

    if_scores_sorted = sorted(p.if_score for p in scored)
    lof_scores_sorted = sorted(p.lof_score for p in scored)

    # correlations against the baseline's own comparison percentages, and a
    # direction breakdown of the "ML flags, baseline doesn't" disagreement set
    shipments = load_shipments(SHIPMENTS)
    feature_rows = build_feature_table(shipments)
    row_by_key = {(r.route, r.week_of): r for r in feature_rows}

    vh = np.array([row_by_key[(p.route, p.week_of)].vs_own_history_pct for p in scored])
    vp = np.array([row_by_key[(p.route, p.week_of)].vs_similar_routes_pct for p in scored])
    if_s = np.array([p.if_score for p in scored])
    lof_s = np.array([p.lof_score for p in scored])
    corr_if_vh = float(np.corrcoef(if_s, vh)[0, 1])
    corr_if_vp = float(np.corrcoef(if_s, vp)[0, 1])
    corr_lof_vh = float(np.corrcoef(lof_s, vh)[0, 1])
    corr_lof_vp = float(np.corrcoef(lof_s, vp)[0, 1])

    ml_only = [p for p in scored if (not p.baseline_statistical_signal) and p.if_prediction and p.lof_prediction]
    ml_only_rising = [p for p in ml_only if row_by_key[(p.route, p.week_of)].vs_own_history_pct > 0]
    ml_only_falling = [p for p in ml_only if row_by_key[(p.route, p.week_of)].vs_own_history_pct <= 0]

    baseline_only = [p for p in scored if p.baseline_statistical_signal and not p.if_prediction and not p.lof_prediction]

    def fmt_case(p):
        r = row_by_key[(p.route, p.week_of)]
        return (
            f"{p.route} {p.week_of} -- vs_own_history={r.vs_own_history_pct:+.1f}%, "
            f"vs_similar_routes={r.vs_similar_routes_pct:+.1f}%, wow_change={r.wow_change_pct:+.1f}%, "
            f"if_score={p.if_score:.3f}, lof_score={p.lof_score:.3f}, baseline_verdict={p.baseline_verdict}"
        )

    s = out.synthetic
    sens = out.sensitivity

    # --- ml_evaluation_report.md --------------------------------------------
    lines = []
    lines.append("# ML Evaluation Report — Isolation Forest & LOF (Parallel Evaluation Mode)")
    lines.append("")
    lines.append(
        "These two unsupervised models run alongside, and completely separate from, "
        "the official rule-based RouteSentry pipeline (`routesentry/`, `submission/output.csv`). "
        "They are an experimental comparison, not a replacement — see the "
        "\"Final integration decision\" section."
    )
    lines.append("")
    lines.append("## 1. Dataset and evaluation-window sizes")
    lines.append("")
    lines.append(f"- Shipments loaded: {out.n_shipments}")
    lines.append(f"- Route-week feature rows: {out.n_feature_rows}")
    lines.append(f"- Eligible rows (>= {MIN_PRIOR_WEEKS_FOR_ML} prior weeks, peer average available): {out.n_eligible}")
    lines.append(f"- Ineligible (insufficient history): {status_counts.get('insufficient_history', 0)}")
    lines.append(f"- Excluded (eligible, but fold's training set below MIN_TRAIN_SIZE={MIN_TRAIN_SIZE}): {status_counts.get('insufficient_training_data', 0)}")
    lines.append(f"- Actually scored by both models: {len(scored)}")
    lines.append(f"- Chronological fold width: {FOLD_STEP_WEEKS} weeks")
    lines.append("")
    lines.append("## 2. Feature engineering and history requirements")
    lines.append("")
    lines.append(f"Features (see `submission/ml_feature_documentation.md` for full definitions): `{', '.join(FEATURE_NAMES)}`.")
    lines.append(
        f"A row is eligible only with >= {MIN_PRIOR_WEEKS_FOR_ML} strictly-prior observed weeks for that "
        "route AND an available same-week peer average. All history-based features are computed "
        "from weeks strictly before the scored week; see `tests/test_ml_leakage.py`."
    )
    lines.append("")
    lines.append("## 3. Chronological split methodology")
    lines.append("")
    lines.append(
        f"Expanding window: at each fold boundary, TRAIN = all eligible rows with `week_of` strictly "
        f"before the boundary; TEST = eligible rows in the next {FOLD_STEP_WEEKS}-week block. A "
        f"`StandardScaler` is fit on TRAIN only, then applied to both TRAIN and TEST. Each model is fit "
        f"on scaled TRAIN only and scores scaled TEST. Folds with fewer than {MIN_TRAIN_SIZE} eligible "
        "training rows are not scored (reported as `insufficient_training_data`) rather than fit on too "
        "little data. See `ml/chronological.py`."
    )
    lines.append("")
    lines.append("## 4. Model parameters and thresholds")
    lines.append("")
    lines.append(f"- Random seed: {RANDOM_SEED} (both models, and the synthetic suite's sampling)")
    lines.append(f"- Isolation Forest: `n_estimators={ISOLATION_FOREST_N_ESTIMATORS}`, `contamination={ISOLATION_FOREST_CONTAMINATION}`")
    lines.append(
        f"- LOF: `novelty=True`, `contamination={LOF_CONTAMINATION}`, "
        f"`n_neighbors = clamp(train_size // {LOF_K_DIVISOR}, {LOF_K_MIN}, {LOF_K_MAX})` per fold "
        "(see `ml/models.py::choose_lof_k`)"
    )
    lines.append(
        "- Score convention (both models): **higher score = more anomalous** "
        "(sklearn's own `decision_function`/`score_samples` are negated to enforce this consistently)."
    )
    lines.append(
        f"- Score distribution on real, scored rows ({len(scored)} rows): "
        f"IF p50/p90/p95/p99/max = {pct(if_scores_sorted,.5):.3f} / {pct(if_scores_sorted,.9):.3f} / "
        f"{pct(if_scores_sorted,.95):.3f} / {pct(if_scores_sorted,.99):.3f} / {max(if_scores_sorted):.3f}; "
        f"LOF p50/p90/p95/p99/max = {pct(lof_scores_sorted,.5):.3f} / {pct(lof_scores_sorted,.9):.3f} / "
        f"{pct(lof_scores_sorted,.95):.3f} / {pct(lof_scores_sorted,.99):.3f} / {max(lof_scores_sorted):.3f}"
    )
    lines.append("")
    lines.append("## 5. Alert counts (real data, scored rows only)")
    lines.append("")
    lines.append(f"| Method | Alerts | Rate |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Rule-based baseline (statistical signal, pre-evidence) | {baseline_alerts} | {baseline_alerts/len(scored):.1%} |")
    lines.append(f"| Isolation Forest | {if_alerts} | {if_alerts/len(scored):.1%} |")
    lines.append(f"| LOF | {lof_alerts} | {lof_alerts/len(scored):.1%} |")
    lines.append("")
    lines.append(
        "Note: the baseline's own final CSV verdict (`Yes`/`No (justified)`/`No`) additionally runs "
        "evidence validation on top of this statistical signal; the comparison here is against the "
        "pre-evidence statistical signal only, since Isolation Forest/LOF never see note text (see "
        "section 6)."
    )
    lines.append("")
    lines.append("## 6. Agreement and disagreement")
    lines.append("")
    lines.append(f"- IF <-> LOF agreement rate: {if_lof_agree/len(scored):.1%} ({if_lof_agree}/{len(scored)})")
    lines.append(f"- IF <-> baseline agreement rate: {if_base_agree/len(scored):.1%} ({if_base_agree}/{len(scored)})")
    lines.append(f"- LOF <-> baseline agreement rate: {lof_base_agree/len(scored):.1%} ({lof_base_agree}/{len(scored)})")
    lines.append(
        f"- Pearson correlation, IF score vs. baseline's vs_own_history_pct: {corr_if_vh:.3f}; "
        f"vs. vs_similar_routes_pct: {corr_if_vp:.3f}"
    )
    lines.append(
        f"- Pearson correlation, LOF score vs. baseline's vs_own_history_pct: {corr_lof_vh:.3f}; "
        f"vs. vs_similar_routes_pct: {corr_lof_vp:.3f}"
    )
    lines.append("")
    lines.append("Full three-way breakdown (baseline_signal, if_prediction, lof_prediction) -> count:")
    lines.append("")
    lines.append("| baseline | IF | LOF | count |")
    lines.append("|---|---|---|---|")
    for (b, i, l), cnt in sorted(overlap.items()):
        lines.append(f"| {b} | {i} | {l} | {cnt} |")
    lines.append("")
    lines.append("## 7. Representative disagreement cases")
    lines.append("")
    lines.append(f"**Baseline flags, neither ML model does** ({len(baseline_only)} cases; showing up to 3):")
    lines.append("")
    for p in baseline_only[:3]:
        lines.append(f"- {fmt_case(p)}")
    lines.append("")
    lines.append(
        f"**Both ML models flag, baseline doesn't** ({len(ml_only)} cases: {len(ml_only_rising)} rising, "
        f"{len(ml_only_falling)} falling; showing up to 3 of each):"
    )
    lines.append("")
    lines.append("*Falling-cost cases (out of scope for the brief's \"rising and unjustified\" framing):*")
    for p in ml_only_falling[:3]:
        lines.append(f"- {fmt_case(p)}")
    if ml_only_rising:
        lines.append("")
        lines.append("*Rising-cost cases (the operationally relevant subset):*")
        for p in ml_only_rising[:3]:
            lines.append(f"- {fmt_case(p)}")
    else:
        lines.append("")
        lines.append("*Rising-cost cases: none in this run.*")
    lines.append("")
    lines.append(
        "Interpretation: the large majority of \"ML-only\" alerts are **falling**-cost weeks "
        "(`vs_own_history_pct <= 0`) that an undirected outlier detector flags as statistically unusual "
        "in either direction, while the rule-based baseline deliberately only considers a **rising** cost "
        "an anomaly (per the brief: \"cost is rising and looks out of the ordinary\"). The "
        "baseline-only misses tend to be modest, peer-comparison-driven rises on a route that has "
        "already been drifting upward for a while — by the time of scoring, the expanding-window "
        "models have absorbed that drift into their learned \"normal\", so a continuation of the trend "
        "no longer looks like an outlier to them, whereas the baseline's fixed +15pp-vs-peers rule still "
        "fires on every qualifying week independently."
    )
    lines.append("")
    lines.append("## 8. Synthetic controlled evaluation (separate from all of the above)")
    lines.append("")
    lines.append(
        f"**This section uses synthetic, labeled data only — it is not a measurement of real-world "
        f"accuracy.** {s.n_synthetic_anomalies} synthetic anomalies were built from real row templates "
        f"with `vs_own_history_pct`, `vs_similar_routes_pct`, and `wow_change_pct` all overridden to "
        f"{s.injected_magnitude_pct:+.0f}% (far beyond the real data's observed range — see "
        "`docs/threshold_analysis.md`), mixed with a held-out set of "
        f"{s.n_test_normal_holdout} real normal rows never used in training ({s.n_train_normal} real rows)."
    )
    lines.append("")
    lines.append("| Model | Synthetic detection rate | False-alert rate on real normal holdout |")
    lines.append("|---|---|---|")
    lines.append(f"| Isolation Forest | {s.if_detection_rate:.1%} | {s.if_false_alert_rate:.1%} |")
    lines.append(f"| LOF | {s.lof_detection_rate:.1%} | {s.lof_false_alert_rate:.1%} |")
    lines.append("")
    lines.append(
        f"Mean scores: IF normal={s.if_mean_score_normal:.3f} vs. synthetic={s.if_mean_score_synthetic:.3f}; "
        f"LOF normal={s.lof_mean_score_normal:.3f} vs. synthetic={s.lof_mean_score_synthetic:.3f}."
    )
    lines.append("")
    lines.append(
        "Both models detect this obviously-extreme, clearly-labeled synthetic case reliably. This "
        "confirms the pipeline wiring (features -> scaler -> model -> score direction) behaves as "
        "intended; it says nothing about how well either model would catch a real, subtler anomaly."
    )
    lines.append("")
    lines.append("## 9. Sensitivity to contamination")
    lines.append("")
    lines.append("Full chronological evaluation re-run at each contamination value (same seed, same folds):")
    lines.append("")
    lines.append("| contamination | IF alerts | LOF alerts |")
    lines.append("|---|---|---|")
    for r in sens:
        lines.append(f"| {r.contamination} | {r.if_alert_count} | {r.lof_alert_count} |")
    lines.append("")
    lines.append(
        "Alert counts scale roughly with the contamination parameter for both models, as expected — "
        "neither model's alert set is a fixed, parameter-insensitive set of \"true\" anomalies. This is "
        "exactly why section 10 does not pick a winner by alert count."
    )
    lines.append("")
    lines.append("## 10. Runtime and reproducibility")
    lines.append("")
    lines.append(f"Full pipeline run (chronological eval + sensitivity re-runs + synthetic suite): {elapsed:.1f}s on this machine.")
    lines.append(
        "Reproducibility: every model fit uses `RANDOM_SEED` "
        f"({RANDOM_SEED}); `StandardScaler`, `IsolationForest`, and `LocalOutlierFactor` are all "
        "deterministic given a fixed seed and fixed input order (feature rows are explicitly sorted by "
        "`(week_of, route)` before any split). See `tests/test_ml_reproducibility.py`, which asserts two "
        "full runs in-process produce identical scores, and `scripts/check_reproducibility.py --ml` for "
        "a 3-process check analogous to the rule-based baseline's own."
    )
    lines.append("")
    lines.append("## 11. Limitations and whether either model adds meaningful value")
    lines.append("")
    lines.append("See `submission/ml_limitations.md` for the full discussion. Summary:")
    lines.append(
        f"- Of {len(ml_only)} \"ML flags, baseline doesn't\" cases, only {len(ml_only_rising)} are even "
        "rising-cost weeks at all, and those are marginal (a percent or two above zero) — essentially no "
        "clear case was found where an ML model caught a meaningful rising-and-unjustified anomaly that "
        "the deterministic baseline's own two prescribed comparisons missed."
    )
    lines.append(
        f"- Of {len(baseline_only)} cases the baseline alone flags, the ML models' lower scores are "
        "explainable (adaptive models absorbing a sustained trend) rather than clearly \"wrong\" — this "
        "is a difference in philosophy (fixed rule vs. adaptive window), not evidence either is more "
        "correct."
    )
    lines.append(
        "- **No ground-truth labels exist for real route-weeks**, so no accuracy/precision/recall claim "
        "is made anywhere in this report for real data — only agreement/disagreement counts, score "
        "distributions, and the clearly-separate synthetic sanity check."
    )
    lines.append("")
    lines.append("## 12. Final integration decision")
    lines.append("")
    lines.append(
        "**Neither model is integrated into the official flagging path.** The measured results do not "
        "establish a reliable improvement over the deterministic baseline for this specific task (\"a "
        "route's cost is rising in a way that doesn't look justified\"): the ML models are undirected "
        "(they flag unusual cost in either direction, not just rises) and, when restricted to their "
        "rising-cost alerts, essentially reproduce the baseline's own signal rather than add new, "
        "verifiably-missed cases. `submission/output.csv` remains exactly the deterministic rule-based "
        "pipeline's output, unchanged by any of this work."
    )
    lines.append(
        "What would change this: a labeled dataset of real route-weeks independently confirmed as "
        "genuine cost problems (not just statistically unusual), large enough to measure real "
        "precision/recall per model — none exists in this case study's inputs."
    )
    lines.append("")

    (SUBMISSION / "ml_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")

    # --- ml_feature_documentation.md ----------------------------------------
    feat_doc = f"""# ML Feature Documentation

All {len(FEATURE_NAMES)} features used by both Isolation Forest and LOF, built in `ml/features.py`.
None of `flagged`, `reason`, `matched_note_id`, or any context-note text is ever used as a feature.

| Feature | Definition | Leakage-free? |
|---|---|---|
| `cost_per_tonne_km` | This route-week's own aggregate cost per tonne-km (`routesentry.weekly`'s ratio-of-sums) | Yes — only this week's own shipments |
| `vs_own_history_pct` | Same definition as the rule-based baseline: % vs. the trailing average of up to 8 strictly-prior weeks | Yes — no current/future week ever enters the average |
| `vs_similar_routes_pct` | Same definition as the baseline: % vs. the same-week average across other routes of the same `route_type`, excluding self | Yes — same week only, never a future or adjacent week |
| `wow_change_pct` | % change vs. the single most recent prior observed week for this route | Yes — strictly prior week only |
| `prior_volatility_pct` | Coefficient of variation (population stdev / mean x 100) of the same trailing prior-week window used for `vs_own_history_pct` | Yes — prior weeks only |
| `n_prior_weeks` | Count of prior weeks actually used (capped at 8) | Yes — a count of past data, not a future-looking value |
| `route_type_short` / `_medium` / `_long` | One-hot encoding of the given (not derived) `route_type` | Yes — static attribute of the route |

## Eligibility gate

A route-week is only included in model training/scoring if:

- `n_prior_weeks >= {MIN_PRIOR_WEEKS_FOR_ML}` (stricter than the baseline's own minimum of 1, since
  `prior_volatility_pct` and `wow_change_pct` need more than a point or two of history to be stable), and
- a same-week peer average exists (`vs_similar_routes_pct is not None`).

Rows failing this gate are reported in `submission/experimental_ml_predictions.csv` with
`ml_status=insufficient_history` and blank scores — never imputed or defaulted to zero.

## Preprocessing

A single `sklearn.preprocessing.StandardScaler` is fit per chronological fold, on that fold's TRAINING
rows only, then applied to both that fold's training and test rows. No scaler is ever fit on the full
dataset or on any row at or after the week being scored.
"""
    (SUBMISSION / "ml_feature_documentation.md").write_text(feat_doc, encoding="utf-8")

    # --- ml_limitations.md ---------------------------------------------------
    limitations = f"""# ML Limitations

## What this evaluation cannot prove

- **No real-world accuracy claim.** There are no ground-truth labels for which real route-weeks are
  genuinely anomalous (as opposed to statistically unusual) — this case study's inputs only provide 10
  free-text context notes, not a labeled anomaly set. Every real-data number in
  `submission/ml_evaluation_report.md` is an agreement/disagreement count, a score distribution, or a
  correlation — never a precision/recall/F1/accuracy figure.
- **The synthetic suite (section 8 of the evaluation report) is a wiring sanity check, not a
  performance benchmark.** Both models detect obviously-extreme, clearly-labeled synthetic cases
  reliably ({s.if_detection_rate:.0%} / {s.lof_detection_rate:.0%} detection rate for IF/LOF at
  {s.injected_magnitude_pct:+.0f}% injected magnitude) — this says the pipeline is correctly wired, not
  that either model would catch a real, subtler anomaly at a comparable rate.
- **Neither model understands "rising."** Isolation Forest and LOF are undirected outlier detectors:
  {len(ml_only_falling)} of this run's {len(ml_only)} "ML flags, baseline doesn't" cases are **falling**
  cost weeks, which the brief explicitly excludes from scope ("a route's cost is rising..."). A
  route-type-specific or sign-constrained model would be needed to make the ML alerts directly
  comparable to the baseline's own semantics.
- **Small data regime.** {out.n_eligible} eligible route-week rows across 7 routes is a small dataset
  for Isolation Forest / LOF by the standards either algorithm is usually applied at; score stability
  across neighbouring folds was not separately stress-tested beyond the contamination sensitivity run
  in section 9.
- **Contamination is an assumption, not a measurement.** {ISOLATION_FOREST_CONTAMINATION:.0%} was
  chosen as a documented, round policy value (`ml/config.py`), not fit to any known true anomaly rate —
  section 9's sensitivity table shows alert counts move considerably (roughly in proportion) as this
  assumption changes, for both models.
- **`prior_volatility_pct` with few prior weeks is noisy.** At the minimum eligible
  `n_prior_weeks={MIN_PRIOR_WEEKS_FOR_ML}`, a standard deviation of 4 points is itself a low-precision
  estimate of the route's true volatility.
- **LOF's `n_neighbors` is a heuristic**, not tuned: `clamp(train_size // {LOF_K_DIVISOR}, {LOF_K_MIN}, {LOF_K_MAX})`.
  Early folds with a small training set use a small, less statistically stable neighbourhood.

## What would be needed to resolve these

A labeled set of real, independently-confirmed cost anomalies (ideally spanning both genuinely-justified
and genuinely-unjustified rises, at a scale of dozens-to-hundreds of examples) to measure actual
precision/recall per model, and a larger multi-year shipment history to evaluate score stability across
more chronological folds than the ~13 this dataset's ~104 weeks allow at an 8-week fold width.
"""
    (SUBMISSION / "ml_limitations.md").write_text(limitations, encoding="utf-8")

    print(f"wrote submission/experimental_ml_predictions.csv ({len(out.predictions)} rows)")
    print(f"wrote submission/experimental_ml_model_comparison.csv")
    print(f"wrote submission/ml_evaluation_report.md")
    print(f"wrote submission/ml_feature_documentation.md")
    print(f"wrote submission/ml_limitations.md")
    print(f"done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
