"""The ML-driven production pipeline: the final anomaly flag comes from a
calibrated LOF score, not the rule-based +15pp threshold in
routesentry/anomaly.py.

Everything else is reused from routesentry/ UNCHANGED:
  - routesentry.weekly.build_weekly_metrics    -- exact brief-mandated
    cost_per_tonne_km / vs_own_history / vs_similar_routes definitions.
  - routesentry.retrieval + routesentry.pipeline._gather_candidates --
    identical note-retrieval candidate generation.
  - routesentry.evidence.evaluate_candidate     -- identical, untouched
    evidence validation (route match, date overlap, direction of impact).
  - routesentry.verdict.decide                  -- identical combination of
    (signal, evidence) into the final Yes / No / No (justified) verdict and
    reason text.
  - routesentry.pipeline.write_output_csv / write_evidence_ledger_csv --
    identical CSV schema, column order, and formatting.

The ONLY thing this module replaces is the statistical-signal step
(routesentry.anomaly.evaluate_signal -> ml.signal.evaluate_ml_signal),
which is why `routesentry/` needed zero modifications for this pivot --
see tests/test_rule_based_mode_preserved.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from routesentry.data_io import load_notes, load_shipments
from routesentry.models import LedgerEntry
from routesentry.pipeline import _gather_candidates
from routesentry.retrieval import TfidfIndex
from routesentry.verdict import decide
from routesentry.weekly import build_weekly_metrics

from ml.calibration import calibrate_lof_contamination
from ml.chronological import run_chronological_evaluation
from ml.config import PRIMARY_LOF_CONTAMINATION
from ml.features import build_feature_table
from ml.signal import evaluate_ml_signal

logger = logging.getLogger("ml.official_pipeline")


def run(
    shipments_path: str | Path,
    notes_path: str | Path,
    contamination: float = PRIMARY_LOF_CONTAMINATION,
    calibrate: bool = False,
) -> List[LedgerEntry]:
    """Run the ML-driven pipeline end to end.

    `contamination` defaults to the value already calibrated offline (see
    ml/config.py::PRIMARY_LOF_CONTAMINATION and docs/model_selection.md).
    Pass `calibrate=True` to re-run the train/validation calibration grid
    search against the current data and use its result instead -- off by
    default so a normal run is fast and uses the one documented, audited
    value rather than silently re-deriving a new one on every invocation.
    """
    shipments = load_shipments(shipments_path)
    notes = load_notes(notes_path)
    index = TfidfIndex(notes)

    metrics = build_weekly_metrics(shipments)
    feature_rows = build_feature_table(shipments)

    if calibrate:
        eligible = [r for r in feature_rows if r.eligible]
        result = calibrate_lof_contamination(eligible)
        contamination = result.selected_contamination
        logger.info(
            "recalibrated LOF contamination=%.3f (train=%d, validation=%d, test=%d rows; rule: %s)",
            contamination, result.n_train, result.n_validation, result.n_test, result.selection_rule,
        )

    ml_results = run_chronological_evaluation(feature_rows, lof_contamination=contamination)
    ml_result_by_key = {(r.route, r.week_of): r for r in ml_results}

    ledger: List[LedgerEntry] = []
    for metric in metrics:
        ml_result = ml_result_by_key.get((metric.route, metric.week_of))
        signal = evaluate_ml_signal(metric, ml_result, contamination)
        candidates = _gather_candidates(metric.route, metric.week_of, notes, index)
        ledger.append(decide(metric, signal, candidates))

    n_flagged = sum(1 for e in ledger if e.final_verdict == "Yes")
    n_justified = sum(1 for e in ledger if e.final_verdict == "No (justified)")
    n_normal = sum(1 for e in ledger if e.final_verdict == "No")
    logger.info(
        "ML-driven pipeline complete: %d route-weeks -- %d unexplained (Yes), %d justified, %d normal "
        "(contamination=%.3f)",
        len(ledger), n_flagged, n_justified, n_normal, contamination,
    )
    return ledger
