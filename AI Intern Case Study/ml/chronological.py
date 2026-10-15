"""Leakage-free chronological (expanding-window) evaluation.

For each fold boundary date T (stepping forward by FOLD_STEP_WEEKS):
  - TRAIN = every ELIGIBLE feature row with week_of strictly before T.
  - TEST  = every ELIGIBLE feature row with T <= week_of < T + FOLD_STEP_WEEKS.
  - A StandardScaler is fit on TRAIN ONLY, then applied to both TRAIN and TEST.
  - Isolation Forest and LOF are each fit on the scaled TRAIN ONLY, then used
    to score the scaled TEST rows.
  - The window then advances; TEST rows from this fold become part of a
    later fold's TRAIN set (once their week is in the past), never the
    reverse.

No feature, scaler statistic, or model parameter for a given TEST row is
ever computed from a row dated on or after that TEST row's own week. This
is the property tests/test_ml_chronological.py checks directly.

If TRAIN has fewer than MIN_TRAIN_SIZE eligible rows, that fold's TEST rows
are reported as excluded (not scored with an under-fit model) rather than
silently skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional

import numpy as np

from ml.config import (
    FOLD_STEP_WEEKS,
    ISOLATION_FOREST_CONTAMINATION,
    LOF_CONTAMINATION,
    MIN_TRAIN_SIZE,
    RANDOM_SEED,
)
from ml.features import FeatureRow
from ml.models import FittedScaler, fit_isolation_forest, fit_lof, score_isolation_forest, score_lof


@dataclass
class MLResult:
    route: str
    week_of: date
    eligible: bool
    ineligibility_reason: str
    fold_index: Optional[int]
    train_size: Optional[int]
    excluded_insufficient_training: bool
    if_score: Optional[float]
    if_prediction: Optional[bool]
    lof_score: Optional[float]
    lof_prediction: Optional[bool]


def run_chronological_evaluation(
    rows: List[FeatureRow],
    fold_step_weeks: int = FOLD_STEP_WEEKS,
    min_train_size: int = MIN_TRAIN_SIZE,
    random_seed: int = RANDOM_SEED,
    if_contamination: float = ISOLATION_FOREST_CONTAMINATION,
    lof_contamination: float = LOF_CONTAMINATION,
) -> List[MLResult]:
    results: List[MLResult] = []

    ineligible = [r for r in rows if not r.eligible]
    for r in ineligible:
        results.append(
            MLResult(
                route=r.route, week_of=r.week_of, eligible=False,
                ineligibility_reason=r.ineligibility_reason, fold_index=None,
                train_size=None, excluded_insufficient_training=False,
                if_score=None, if_prediction=None, lof_score=None, lof_prediction=None,
            )
        )

    eligible = sorted([r for r in rows if r.eligible], key=lambda r: (r.week_of, r.route))
    if not eligible:
        return results

    first_week = eligible[0].week_of
    last_week = eligible[-1].week_of
    step = timedelta(weeks=fold_step_weeks)

    fold_index = 0
    boundary = first_week
    while boundary <= last_week:
        fold_end = boundary + step
        train_rows = [r for r in eligible if r.week_of < boundary]
        test_rows = [r for r in eligible if boundary <= r.week_of < fold_end]
        if not test_rows:
            boundary = fold_end
            continue

        if len(train_rows) < min_train_size:
            for r in test_rows:
                results.append(
                    MLResult(
                        route=r.route, week_of=r.week_of, eligible=True,
                        ineligibility_reason="", fold_index=fold_index,
                        train_size=len(train_rows), excluded_insufficient_training=True,
                        if_score=None, if_prediction=None, lof_score=None, lof_prediction=None,
                    )
                )
            fold_index += 1
            boundary = fold_end
            continue

        X_train = np.array([r.feature_vector() for r in train_rows])
        X_test = np.array([r.feature_vector() for r in test_rows])

        scaler = FittedScaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if_model = fit_isolation_forest(X_train_scaled, contamination=if_contamination, random_state=random_seed)
        if_scores, if_preds = score_isolation_forest(if_model, X_test_scaled)

        lof_model = fit_lof(X_train_scaled, contamination=lof_contamination)
        lof_scores, lof_preds = score_lof(lof_model, X_test_scaled)

        for r, if_s, if_p, lof_s, lof_p in zip(test_rows, if_scores, if_preds, lof_scores, lof_preds):
            results.append(
                MLResult(
                    route=r.route, week_of=r.week_of, eligible=True,
                    ineligibility_reason="", fold_index=fold_index,
                    train_size=len(train_rows), excluded_insufficient_training=False,
                    if_score=float(if_s), if_prediction=bool(if_p),
                    lof_score=float(lof_s), lof_prediction=bool(lof_p),
                )
            )

        fold_index += 1
        boundary = fold_end

    results.sort(key=lambda r: (r.week_of, r.route))
    return results
