"""The three rows in sample_output_format_v2.csv are worked examples from
the brief. We validate the RULE-BASED pipeline (routesentry.pipeline.run,
selectable via `run.py --mode rule_based`) reproduces their
cost_per_tonne_km, both percentage comparisons, flagged verdict, and
matched_note_id exactly (wording of `reason` is explicitly allowed to
differ per the brief: "Explanation wording may vary; cited note, verdict,
and numbers must not").

The ML-driven pipeline (now run.py's default) reproduces 2 of these 3
rows exactly and diverges on the third by design -- see
tests/test_ml_golden_divergence.py and docs/model_selection.md."""
from datetime import date
from pathlib import Path

import pytest

from routesentry.pipeline import run

ROOT = Path(__file__).resolve().parent.parent
SHIPMENTS = ROOT / "shipment_records.csv"
NOTES = ROOT / "context_notes.csv"

GOLDEN = [
    # route, week_of, cost_per_tonne_km, vs_own_history_pct, vs_similar_routes_pct, flagged, matched_note_id
    ("Delhi-Jaipur", date(2024, 11, 11), 4.17, 35.5, 21.0, "Yes", ""),
    ("Ahmedabad-Mumbai", date(2025, 1, 20), 3.29, 29.5, 22.5, "No (justified)", "N002"),
    ("Mumbai-Pune", date(2025, 9, 15), 3.98, 9.2, 23.6, "Yes", ""),
]


@pytest.fixture(scope="module")
def ledger():
    return run(SHIPMENTS, NOTES)


@pytest.mark.parametrize("route,week,cpkm,vh,vp,flagged,note_id", GOLDEN)
def test_golden_row_matches_sample_output(ledger, route, week, cpkm, vh, vp, flagged, note_id):
    entry = next(e for e in ledger if e.route == route and e.week_of == week)
    assert round(entry.cost_per_tonne_km, 2) == cpkm
    assert round(entry.vs_own_history_pct, 1) == vh
    assert round(entry.vs_similar_routes_pct, 1) == vp
    assert entry.final_verdict == flagged
    assert (entry.accepted_note_id or "") == note_id
