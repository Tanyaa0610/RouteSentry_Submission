"""Orchestrates the full parallel ML evaluation: load data -> build
features -> run the rule-based baseline (for comparison only, completely
unmodified) -> run chronological Isolation Forest / LOF evaluation -> run
the sensitivity analysis -> run the synthetic controlled suite -> assemble
one merged prediction record per route-week.

This module writes nothing by itself -- see scripts/run_ml_evaluation.py
for the CSV/Markdown writers. It never touches `submission/output.csv`,
`submission/evidence_ledger.csv`, or anything `routesentry/pipeline.run`
writes by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from routesentry.data_io import load_shipments
from routesentry.models import LedgerEntry
from routesentry.pipeline import run as run_baseline_pipeline

from ml.chronological import MLResult, run_chronological_evaluation
from ml.config import SENSITIVITY_CONTAMINATIONS
from ml.features import FeatureRow, build_feature_table
from ml.synthetic import SyntheticEvaluationResult, run_synthetic_evaluation


@dataclass
class PredictionRecord:
    route: str
    week_of: date
    ml_status: str  # "scored" | "insufficient_history" | "insufficient_training_data"
    if_score: Optional[float]
    if_prediction: Optional[bool]
    lof_score: Optional[float]
    lof_prediction: Optional[bool]
    baseline_statistical_signal: bool
    baseline_verdict: str
    baseline_matched_note_id: str
    if_lof_agree: Optional[bool]
    if_baseline_agree: Optional[bool]
    lof_baseline_agree: Optional[bool]


def _ml_status(r: MLResult) -> str:
    if not r.eligible:
        return "insufficient_history"
    if r.excluded_insufficient_training:
        return "insufficient_training_data"
    return "scored"


def merge_predictions(ml_results: List[MLResult], baseline_ledger: List[LedgerEntry]) -> List[PredictionRecord]:
    baseline_by_key: Dict[Tuple[str, date], LedgerEntry] = {
        (e.route, e.week_of): e for e in baseline_ledger
    }
    records: List[PredictionRecord] = []
    for r in ml_results:
        entry = baseline_by_key.get((r.route, r.week_of))
        if entry is None:
            continue  # should not happen -- same input drives both
        if_pred = r.if_prediction
        lof_pred = r.lof_prediction
        baseline_signal = entry.statistical_signal
        records.append(
            PredictionRecord(
                route=r.route,
                week_of=r.week_of,
                ml_status=_ml_status(r),
                if_score=r.if_score,
                if_prediction=if_pred,
                lof_score=r.lof_score,
                lof_prediction=lof_pred,
                baseline_statistical_signal=baseline_signal,
                baseline_verdict=entry.final_verdict,
                baseline_matched_note_id=entry.accepted_note_id or "",
                if_lof_agree=(if_pred == lof_pred) if if_pred is not None and lof_pred is not None else None,
                if_baseline_agree=(if_pred == baseline_signal) if if_pred is not None else None,
                lof_baseline_agree=(lof_pred == baseline_signal) if lof_pred is not None else None,
            )
        )
    records.sort(key=lambda p: (p.week_of, p.route))
    return records


@dataclass
class SensitivityResult:
    contamination: float
    if_alert_count: int
    lof_alert_count: int


def run_sensitivity_analysis(rows: List[FeatureRow], contaminations: List[float] = None) -> List[SensitivityResult]:
    contaminations = contaminations or SENSITIVITY_CONTAMINATIONS
    results = []
    for c in contaminations:
        ml_results = run_chronological_evaluation(rows, if_contamination=c, lof_contamination=c)
        if_count = sum(1 for r in ml_results if r.if_prediction is True)
        lof_count = sum(1 for r in ml_results if r.lof_prediction is True)
        results.append(SensitivityResult(contamination=c, if_alert_count=if_count, lof_alert_count=lof_count))
    return results


@dataclass
class MLPipelineOutput:
    predictions: List[PredictionRecord]
    sensitivity: List[SensitivityResult]
    synthetic: SyntheticEvaluationResult
    n_shipments: int
    n_feature_rows: int
    n_eligible: int


def run_full_ml_pipeline(shipments_path: str | Path, notes_path: str | Path) -> MLPipelineOutput:
    shipments = load_shipments(shipments_path)
    baseline_ledger = run_baseline_pipeline(shipments_path, notes_path)

    rows = build_feature_table(shipments)
    eligible_rows = [r for r in rows if r.eligible]

    ml_results = run_chronological_evaluation(rows)
    predictions = merge_predictions(ml_results, baseline_ledger)

    sensitivity = run_sensitivity_analysis(rows)
    synthetic = run_synthetic_evaluation(eligible_rows)

    return MLPipelineOutput(
        predictions=predictions,
        sensitivity=sensitivity,
        synthetic=synthetic,
        n_shipments=len(shipments),
        n_feature_rows=len(rows),
        n_eligible=len(eligible_rows),
    )
