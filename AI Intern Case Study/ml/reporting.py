"""CSV and Markdown writers for the ML evaluation. Every number written
here comes from an actual field on a PredictionRecord / SensitivityResult /
SyntheticEvaluationResult produced by ml/pipeline.py -- nothing in this
module computes or invents a number of its own beyond simple counting and
aggregation of those already-computed values.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

from ml.pipeline import MLPipelineOutput, PredictionRecord, SensitivityResult


PREDICTIONS_COLUMNS = [
    "route", "week_of", "ml_status",
    "isolation_forest_score", "isolation_forest_prediction",
    "lof_score", "lof_prediction",
    "baseline_statistical_signal", "baseline_verdict", "baseline_matched_note_id",
    "if_lof_agree", "if_baseline_agree", "lof_baseline_agree",
]


def write_predictions_csv(predictions: List[PredictionRecord], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(PREDICTIONS_COLUMNS)
        for p in predictions:
            writer.writerow([
                p.route,
                p.week_of.isoformat(),
                p.ml_status,
                "" if p.if_score is None else f"{p.if_score:.4f}",
                "" if p.if_prediction is None else p.if_prediction,
                "" if p.lof_score is None else f"{p.lof_score:.4f}",
                "" if p.lof_prediction is None else p.lof_prediction,
                p.baseline_statistical_signal,
                p.baseline_verdict,
                p.baseline_matched_note_id,
                "" if p.if_lof_agree is None else p.if_lof_agree,
                "" if p.if_baseline_agree is None else p.if_baseline_agree,
                "" if p.lof_baseline_agree is None else p.lof_baseline_agree,
            ])


COMPARISON_COLUMNS = [
    "route", "n_scored", "baseline_alert_count", "if_alert_count", "lof_alert_count",
    "if_lof_agreement_rate", "if_baseline_agreement_rate", "lof_baseline_agreement_rate",
]


def write_comparison_csv(predictions: List[PredictionRecord], path: str | Path) -> None:
    path = Path(path)
    by_route: dict[str, List[PredictionRecord]] = defaultdict(list)
    for p in predictions:
        if p.ml_status == "scored":
            by_route[p.route].append(p)

    rows = []
    for route in sorted(by_route):
        recs = by_route[route]
        n = len(recs)
        baseline_n = sum(1 for r in recs if r.baseline_statistical_signal)
        if_n = sum(1 for r in recs if r.if_prediction)
        lof_n = sum(1 for r in recs if r.lof_prediction)
        if_lof_rate = sum(1 for r in recs if r.if_lof_agree) / n if n else 0.0
        if_base_rate = sum(1 for r in recs if r.if_baseline_agree) / n if n else 0.0
        lof_base_rate = sum(1 for r in recs if r.lof_baseline_agree) / n if n else 0.0
        rows.append([route, n, baseline_n, if_n, lof_n, f"{if_lof_rate:.3f}", f"{if_base_rate:.3f}", f"{lof_base_rate:.3f}"])

    all_recs = [p for recs in by_route.values() for p in recs]
    n = len(all_recs)
    if n:
        rows.append([
            "ALL_ROUTES", n,
            sum(1 for r in all_recs if r.baseline_statistical_signal),
            sum(1 for r in all_recs if r.if_prediction),
            sum(1 for r in all_recs if r.lof_prediction),
            f"{sum(1 for r in all_recs if r.if_lof_agree) / n:.3f}",
            f"{sum(1 for r in all_recs if r.if_baseline_agree) / n:.3f}",
            f"{sum(1 for r in all_recs if r.lof_baseline_agree) / n:.3f}",
        ])

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COMPARISON_COLUMNS)
        writer.writerows(rows)


def overlap_breakdown(predictions: List[PredictionRecord]) -> "Counter[tuple]":
    scored = [p for p in predictions if p.ml_status == "scored"]
    return Counter(
        (bool(p.baseline_statistical_signal), bool(p.if_prediction), bool(p.lof_prediction))
        for p in scored
    )
