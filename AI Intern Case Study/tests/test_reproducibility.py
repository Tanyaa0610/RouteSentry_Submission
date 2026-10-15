"""In-process reproducibility of the RULE-BASED pipeline (routesentry.pipeline.run,
selectable via `run.py --mode rule_based`): it is pure (no randomness, no
clock, no network), so re-running it on the same input must yield
byte-identical output. The full 3-process CLI check (with CSV diffing)
lives in scripts/check_reproducibility.py / docs/reproducibility_report.md
-- this test is the fast, always-on pytest guard for the same property.

The ML-driven pipeline (now run.py's default) has its own reproducibility
tests in tests/test_ml_official_pipeline.py and
tests/test_ml_reproducibility.py -- it uses a fixed random seed rather
than being seedless, but is equally deterministic given that seed."""
from pathlib import Path

from routesentry.pipeline import run, write_evidence_ledger_csv, write_output_csv

ROOT = Path(__file__).resolve().parent.parent
SHIPMENTS = ROOT / "shipment_records.csv"
NOTES = ROOT / "context_notes.csv"


def test_two_in_process_runs_produce_identical_ledger_values():
    ledger_a = run(SHIPMENTS, NOTES)
    ledger_b = run(SHIPMENTS, NOTES)
    assert len(ledger_a) == len(ledger_b)
    for a, b in zip(ledger_a, ledger_b):
        assert a.route == b.route
        assert a.week_of == b.week_of
        assert a.cost_per_tonne_km == b.cost_per_tonne_km
        assert a.vs_own_history_pct == b.vs_own_history_pct
        assert a.vs_similar_routes_pct == b.vs_similar_routes_pct
        assert a.final_verdict == b.final_verdict
        assert a.accepted_note_id == b.accepted_note_id
        assert a.reason == b.reason


def test_two_written_csvs_are_byte_identical(tmp_path):
    ledger = run(SHIPMENTS, NOTES)
    out_a = tmp_path / "a.csv"
    out_b = tmp_path / "b.csv"
    write_output_csv(ledger, out_a)
    write_output_csv(run(SHIPMENTS, NOTES), out_b)
    assert out_a.read_bytes() == out_b.read_bytes()


def test_evidence_ledger_csv_is_byte_identical_across_runs(tmp_path):
    out_a = tmp_path / "ledger_a.csv"
    out_b = tmp_path / "ledger_b.csv"
    write_evidence_ledger_csv(run(SHIPMENTS, NOTES), out_a)
    write_evidence_ledger_csv(run(SHIPMENTS, NOTES), out_b)
    assert out_a.read_bytes() == out_b.read_bytes()
