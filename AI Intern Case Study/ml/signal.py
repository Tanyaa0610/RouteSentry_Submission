"""The ML-driven replacement for routesentry/anomaly.py's flat-threshold
rule. This is the ONLY place in the ML-driven pipeline that decides
whether a route-week is a statistical anomaly -- it is deliberately kept
separate from evidence validation (routesentry/evidence.py, untouched) and
from the final verdict assembly (routesentry/verdict.py, untouched and
reused as-is).

Two gates, both required:

1. A literal reading of the brief's own words -- "Flag any route where the
   cost is RISING" -- is kept as a hard, non-ML gate. Isolation
   Forest/LOF are undirected outlier detectors (see
   submission/ml_evaluation_report.md): left alone, they flag unusual cost
   in either direction, and on this dataset most of what they'd add beyond
   this gate turns out to be unusually *low*-cost weeks. That is
   explicitly out of scope for this brief, so it is not something the ML
   model is asked to decide -- it is enforced here, structurally, before
   the model's score is even consulted.
2. The calibrated LOF score for this route-week, produced by
   ml/chronological.py's leakage-free walk-forward evaluation (a model
   fit only on strictly-prior pooled training rows), compared against the
   contamination-calibrated decision boundary from ml/calibration.py.

Returns the existing `routesentry.models.StatisticalSignal` type unchanged,
so `routesentry.verdict.decide` -- the single place a signal and evidence
are combined into a final verdict -- needs no modification at all.
"""
from __future__ import annotations

from typing import Optional

from routesentry.models import StatisticalSignal, WeeklyMetric

from ml.chronological import MLResult


def evaluate_ml_signal(metric: WeeklyMetric, ml_result: Optional[MLResult], contamination: float) -> StatisticalSignal:
    if metric.vs_own_history_pct is None:
        return StatisticalSignal(
            is_anomaly=False,
            rule_triggered=(
                "not evaluated: no prior weeks available for this route yet, "
                "so 'rising vs. own history' cannot be assessed"
            ),
        )

    rising = metric.vs_own_history_pct > 0
    if not rising:
        return StatisticalSignal(
            is_anomaly=False,
            rule_triggered=f"not rising: vs_own_history={metric.vs_own_history_pct:.1f}% <= 0% (required gate, not ML-decided)",
        )

    if ml_result is None or not ml_result.eligible:
        return StatisticalSignal(
            is_anomaly=False,
            rule_triggered="not evaluated: insufficient history for the ML feature set (see ml/features.py eligibility rule)",
        )
    if ml_result.excluded_insufficient_training or ml_result.lof_score is None:
        return StatisticalSignal(
            is_anomaly=False,
            rule_triggered=(
                f"not evaluated: this route-week's chronological fold had only "
                f"{ml_result.train_size if ml_result.train_size is not None else 0} pooled training rows "
                "available (below MIN_TRAIN_SIZE), so no LOF model could be reliably fit yet"
            ),
        )

    if ml_result.lof_prediction:
        return StatisticalSignal(
            is_anomaly=True,
            rule_triggered=(
                f"LOF anomaly score={ml_result.lof_score:.3f} exceeds the calibrated decision boundary "
                f"(contamination={contamination}) fit on this fold's {ml_result.train_size} prior pooled "
                f"training rows; route is rising vs. own history ({metric.vs_own_history_pct:.1f}%)"
            ),
        )

    return StatisticalSignal(
        is_anomaly=False,
        rule_triggered=(
            f"rising ({metric.vs_own_history_pct:.1f}%) but LOF score={ml_result.lof_score:.3f} is within "
            f"the calibrated normal range for this fold (contamination={contamination})"
        ),
    )
