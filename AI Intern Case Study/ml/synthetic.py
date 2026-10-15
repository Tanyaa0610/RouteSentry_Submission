"""Separate, controlled, LABELED synthetic evaluation.

This is NOT a measurement of real-world accuracy. It answers one narrow
question: "if an obviously extreme, clearly-labeled anomaly existed, would
each fitted model actually flag it, without also flagging normal weeks at
a high rate?" It uses synthetic ground-truth labels that exist only here;
nothing from this module feeds into the real-data comparison or any
`submission/` file other than `ml_evaluation_report.md`'s own clearly
separate "Synthetic controlled evaluation" section.

Method: take real ELIGIBLE feature rows as the only source of realistic
"normal" feature vectors. Fit models on a training subset of normal rows.
Score a held-out set of normal rows (expected: mostly non-anomalous) plus a
separate set of synthetic rows built by taking a real row as a template and
replacing its comparison percentages with an extreme value far outside the
real data's observed range (see docs/threshold_analysis.md for that range).
Detection rate and false-alert rate are computed only within this labeled
synthetic set.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

import numpy as np

from ml.config import (
    RANDOM_SEED,
    SYNTHETIC_N_ANOMALIES,
    SYNTHETIC_N_NORMAL_HOLDOUT,
    SYNTHETIC_PCT_MAGNITUDE,
)
from ml.features import FeatureRow
from ml.models import FittedScaler, fit_isolation_forest, fit_lof, score_isolation_forest, score_lof


@dataclass
class SyntheticEvaluationResult:
    n_train_normal: int
    n_test_normal_holdout: int
    n_synthetic_anomalies: int
    injected_magnitude_pct: float
    if_detection_rate: float
    if_false_alert_rate: float
    lof_detection_rate: float
    lof_false_alert_rate: float
    if_mean_score_normal: float
    if_mean_score_synthetic: float
    lof_mean_score_normal: float
    lof_mean_score_synthetic: float


def _make_synthetic_row(template: FeatureRow, magnitude: float) -> FeatureRow:
    """A synthetic anomalous row: same route/route_type/history-depth as a
    real template, but with all three comparison signals pushed far beyond
    the real data's observed range in the same (rising) direction."""
    return replace(
        template,
        vs_own_history_pct=magnitude,
        vs_similar_routes_pct=magnitude,
        wow_change_pct=magnitude,
    )


def run_synthetic_evaluation(
    eligible_rows: List[FeatureRow],
    n_anomalies: int = SYNTHETIC_N_ANOMALIES,
    n_normal_holdout: int = SYNTHETIC_N_NORMAL_HOLDOUT,
    magnitude: float = SYNTHETIC_PCT_MAGNITUDE,
    seed: int = RANDOM_SEED,
) -> SyntheticEvaluationResult:
    pool = sorted(eligible_rows, key=lambda r: (r.week_of, r.route))
    if len(pool) < n_normal_holdout + 10:
        raise ValueError(
            f"not enough eligible real rows ({len(pool)}) to run the synthetic "
            f"suite with n_normal_holdout={n_normal_holdout}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(pool))
    holdout_idx = indices[:n_normal_holdout]
    train_idx = indices[n_normal_holdout:]

    train_rows = [pool[i] for i in train_idx]
    holdout_rows = [pool[i] for i in holdout_idx]

    template_idx = rng.choice(len(holdout_rows), size=n_anomalies, replace=True)
    synthetic_rows = [_make_synthetic_row(holdout_rows[i], magnitude) for i in template_idx]

    X_train = np.array([r.feature_vector() for r in train_rows])
    X_holdout = np.array([r.feature_vector() for r in holdout_rows])
    X_synthetic = np.array([r.feature_vector() for r in synthetic_rows])

    scaler = FittedScaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_holdout_s = scaler.transform(X_holdout)
    X_synthetic_s = scaler.transform(X_synthetic)

    if_model = fit_isolation_forest(X_train_s, random_state=seed)
    if_holdout_scores, if_holdout_preds = score_isolation_forest(if_model, X_holdout_s)
    if_synth_scores, if_synth_preds = score_isolation_forest(if_model, X_synthetic_s)

    lof_model = fit_lof(X_train_s)
    lof_holdout_scores, lof_holdout_preds = score_lof(lof_model, X_holdout_s)
    lof_synth_scores, lof_synth_preds = score_lof(lof_model, X_synthetic_s)

    return SyntheticEvaluationResult(
        n_train_normal=len(train_rows),
        n_test_normal_holdout=len(holdout_rows),
        n_synthetic_anomalies=len(synthetic_rows),
        injected_magnitude_pct=magnitude,
        if_detection_rate=float(np.mean(if_synth_preds)),
        if_false_alert_rate=float(np.mean(if_holdout_preds)),
        lof_detection_rate=float(np.mean(lof_synth_preds)),
        lof_false_alert_rate=float(np.mean(lof_holdout_preds)),
        if_mean_score_normal=float(np.mean(if_holdout_scores)),
        if_mean_score_synthetic=float(np.mean(if_synth_scores)),
        lof_mean_score_normal=float(np.mean(lof_holdout_scores)),
        lof_mean_score_synthetic=float(np.mean(lof_synth_scores)),
    )
