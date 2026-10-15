"""Calibrates the anomaly-detection threshold (LOF's `contamination`
parameter) using ONLY a chronological train/validation split of the real
data, plus synthetic injected anomalies scoped to the validation period.
The final held-out TEST period is never touched here -- it is reserved for
reporting real-data results after the threshold is already fixed, exactly
so the threshold choice cannot leak information from the period it will
later be evaluated against.

No real ground-truth labels exist for any route-week (see
submission/ml_limitations.md), so there is no real precision/recall to
optimize directly. The selection rule below is therefore necessarily a
policy choice, not a discovered optimum -- documented as such.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ml.config import RANDOM_SEED, SYNTHETIC_PCT_MAGNITUDE
from ml.features import FeatureRow
from ml.models import FittedScaler, fit_lof, score_lof
from ml.synthetic import _make_synthetic_row

# Chronological three-way split, by calendar week (not by row count, so no
# route's rows are split across periods in a way that would let a route's
# own later history leak into its own earlier calibration).
CALIBRATION_TRAIN_FRACTION = 0.60
CALIBRATION_VALIDATION_FRACTION = 0.20
# remaining ~0.20 is the held-out TEST period -- reported, never used to pick the threshold

CONTAMINATION_GRID = [0.02, 0.03, 0.05, 0.08, 0.10]
# Selection rule: among grid points whose validation false-alert rate is at
# or below this ceiling, pick the one with the highest synthetic detection
# rate; tie-break on the lowest (most conservative) contamination. If no
# candidate meets the ceiling, fall back to the one with the lowest
# false-alert rate. This is a documented policy, not a fitted optimum.
MAX_ACCEPTABLE_VALIDATION_FALSE_ALERT_RATE = 0.05
N_SYNTHETIC_VALIDATION_ANOMALIES = 30


@dataclass
class ContaminationCandidate:
    contamination: float
    synthetic_detection_rate: float
    validation_false_alert_rate: float


@dataclass
class CalibrationResult:
    selected_contamination: float
    grid: List[ContaminationCandidate]
    n_train: int
    n_validation: int
    n_test: int
    train_week_range: Tuple[str, str]
    validation_week_range: Tuple[str, str]
    test_week_range: Tuple[str, str]
    selection_rule: str


def chronological_three_way_split(
    eligible_rows: List[FeatureRow],
    train_fraction: float = CALIBRATION_TRAIN_FRACTION,
    validation_fraction: float = CALIBRATION_VALIDATION_FRACTION,
) -> Tuple[List[FeatureRow], List[FeatureRow], List[FeatureRow]]:
    sorted_rows = sorted(eligible_rows, key=lambda r: (r.week_of, r.route))
    weeks = sorted({r.week_of for r in sorted_rows})
    n_weeks = len(weeks)
    train_cutoff = weeks[int(n_weeks * train_fraction)]
    val_cutoff = weeks[int(n_weeks * (train_fraction + validation_fraction))]

    train = [r for r in sorted_rows if r.week_of < train_cutoff]
    validation = [r for r in sorted_rows if train_cutoff <= r.week_of < val_cutoff]
    test = [r for r in sorted_rows if r.week_of >= val_cutoff]
    return train, validation, test


def calibrate_lof_contamination(
    eligible_rows: List[FeatureRow],
    seed: int = RANDOM_SEED,
) -> CalibrationResult:
    train, validation, test = chronological_three_way_split(eligible_rows)
    if len(train) < 20 or len(validation) < 10:
        raise ValueError(
            f"not enough data to calibrate: {len(train)} train / {len(validation)} "
            "validation rows -- need at least 20/10"
        )

    X_train = np.array([r.feature_vector() for r in train])
    X_val = np.array([r.feature_vector() for r in validation])
    scaler = FittedScaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    rng = np.random.default_rng(seed)
    n_synth = min(N_SYNTHETIC_VALIDATION_ANOMALIES, len(validation))
    template_idx = rng.choice(len(validation), size=n_synth, replace=True)
    synthetic_val_rows = [_make_synthetic_row(validation[i], SYNTHETIC_PCT_MAGNITUDE) for i in template_idx]
    X_synth = np.array([r.feature_vector() for r in synthetic_val_rows])
    X_synth_s = scaler.transform(X_synth)

    grid: List[ContaminationCandidate] = []
    for c in CONTAMINATION_GRID:
        model = fit_lof(X_train_s, contamination=c)
        _val_scores, val_preds = score_lof(model, X_val_s)
        _synth_scores, synth_preds = score_lof(model, X_synth_s)
        grid.append(
            ContaminationCandidate(
                contamination=c,
                synthetic_detection_rate=float(np.mean(synth_preds)),
                validation_false_alert_rate=float(np.mean(val_preds)),
            )
        )

    acceptable = [g for g in grid if g.validation_false_alert_rate <= MAX_ACCEPTABLE_VALIDATION_FALSE_ALERT_RATE]
    if acceptable:
        best = max(acceptable, key=lambda g: (g.synthetic_detection_rate, -g.contamination))
        rule = (
            f"highest synthetic detection rate among candidates with validation "
            f"false-alert rate <= {MAX_ACCEPTABLE_VALIDATION_FALSE_ALERT_RATE:.0%}, "
            "tie-broken toward the lowest (most conservative) contamination"
        )
    else:
        best = min(grid, key=lambda g: g.validation_false_alert_rate)
        rule = (
            f"no candidate met the {MAX_ACCEPTABLE_VALIDATION_FALSE_ALERT_RATE:.0%} false-alert "
            "ceiling on validation data; fell back to the candidate with the lowest "
            "validation false-alert rate"
        )

    return CalibrationResult(
        selected_contamination=best.contamination,
        grid=grid,
        n_train=len(train),
        n_validation=len(validation),
        n_test=len(test),
        train_week_range=(train[0].week_of.isoformat(), train[-1].week_of.isoformat()),
        validation_week_range=(validation[0].week_of.isoformat(), validation[-1].week_of.isoformat()),
        test_week_range=(test[0].week_of.isoformat(), test[-1].week_of.isoformat()) if test else ("", ""),
        selection_rule=rule,
    )
