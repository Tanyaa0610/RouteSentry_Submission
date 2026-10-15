"""Documents, deliberately, where the ML-driven pipeline's verdicts diverge
from the brief's own 3 worked examples in sample_output_format_v2.csv.

tests/test_golden_examples.py already proves the RULE-BASED pipeline
(routesentry.pipeline.run, still fully intact and selectable via
`--mode rule_based`) reproduces all 3 rows exactly. This file proves the
ML-driven pipeline reproduces 2 of the 3 exactly and documents the one
divergence precisely -- so the change in behavior is a tracked, asserted
fact, not something a future change could silently alter without a test
failing. See docs/model_selection.md "Divergence from the brief's own
examples" for the full discussion of why this happens and what it means.
"""
from datetime import date
from pathlib import Path

import pytest

from ml.official_pipeline import run as run_ml_driven

ROOT = Path(__file__).resolve().parent.parent
SHIPMENTS = ROOT / "shipment_records.csv"
NOTES = ROOT / "context_notes.csv"


@pytest.fixture(scope="module")
def ml_ledger():
    return run_ml_driven(SHIPMENTS, NOTES)


def test_delhi_jaipur_golden_example_still_matches(ml_ledger):
    e = next(x for x in ml_ledger if x.route == "Delhi-Jaipur" and x.week_of == date(2024, 11, 11))
    assert round(e.cost_per_tonne_km, 2) == 4.17
    assert round(e.vs_own_history_pct, 1) == 35.5
    assert round(e.vs_similar_routes_pct, 1) == 21.0
    assert e.final_verdict == "Yes"
    assert not e.accepted_note_id


def test_ahmedabad_mumbai_golden_example_still_matches(ml_ledger):
    e = next(x for x in ml_ledger if x.route == "Ahmedabad-Mumbai" and x.week_of == date(2025, 1, 20))
    assert round(e.cost_per_tonne_km, 2) == 3.29
    assert round(e.vs_own_history_pct, 1) == 29.5
    assert round(e.vs_similar_routes_pct, 1) == 22.5
    assert e.final_verdict == "No (justified)"
    assert e.accepted_note_id == "N002"


def test_mumbai_pune_golden_example_diverges_under_calibrated_ml_threshold(ml_ledger):
    # KNOWN, DOCUMENTED DIVERGENCE: the brief's sample shows "Yes" here.
    # The numeric fields (cost_per_tonne_km, vs_own_history, vs_similar_routes)
    # still match exactly -- only the ML-calibrated verdict differs. Traced
    # down to the raw shipments and confirmed NOT a bug: this row's LOF score
    # (2.096) is the HIGHEST of its own 56-row test fold, but ranks only at
    # the 95.2nd percentile of the 560-row TRAINING distribution the model's
    # contamination=0.02 boundary (~98th percentile) was fit from -- i.e. the
    # model has already seen enough similarly-elevated weeks (Mumbai-Pune's
    # own documented multi-month drift) that this one no longer looks
    # exceptional to it. Contamination 0.03/0.05 (both within the honest
    # <=5%-false-alert calibration ceiling) and a per-route-type-only LOF fit
    # were both tried empirically and neither flips this row; only
    # contamination >=0.08 would, and that was already excluded by the
    # calibration's own ceiling (6.4% false alerts) for reasons unrelated to
    # this one row. See docs/model_selection.md SS3.1-3.2 for the full trace.
    e = next(x for x in ml_ledger if x.route == "Mumbai-Pune" and x.week_of == date(2025, 9, 15))
    assert round(e.cost_per_tonne_km, 2) == 3.98
    assert round(e.vs_own_history_pct, 1) == 9.2
    assert round(e.vs_similar_routes_pct, 1) == 23.6
    assert e.final_verdict == "No", (
        "if this now flags 'Yes' again, the calibration or threshold logic has changed -- "
        "update docs/model_selection.md's divergence discussion to match, don't just adjust this assertion"
    )
