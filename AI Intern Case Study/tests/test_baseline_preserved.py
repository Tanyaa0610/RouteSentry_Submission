"""Regression guard: the routesentry/ rule-based pipeline itself must stay
completely untouched by the ML-driven pivot (ml/official_pipeline.py is
now the default in run.py, but it is a NEW module that only reuses
routesentry/ -- it never modifies it). This re-runs the untouched
routesentry/ pipeline fresh and checks it still matches the preserved
benchmark artifact exactly, and that the brief's 3 golden sample rows
still hold for that rule-based path.

Note: as of the ML-driven pivot, `submission/output.csv` is the ML-driven
result (run.py's new default), NOT this benchmark. The rule-based result
is preserved at `submission/benchmark_rule_based_output.csv` specifically
so this regression check has something stable to compare against -- see
docs/model_selection.md."""
import hashlib
from pathlib import Path

from routesentry.pipeline import run, write_output_csv
from tests.test_golden_examples import GOLDEN

ROOT = Path(__file__).resolve().parent.parent
SHIPMENTS = ROOT / "shipment_records.csv"
NOTES = ROOT / "context_notes.csv"
BENCHMARK_OUTPUT = ROOT / "submission" / "benchmark_rule_based_output.csv"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_fresh_rule_based_run_matches_preserved_benchmark(tmp_path):
    assert BENCHMARK_OUTPUT.exists(), "submission/benchmark_rule_based_output.csv must exist and be committed"
    fresh_path = tmp_path / "output.csv"
    ledger = run(SHIPMENTS, NOTES)
    write_output_csv(ledger, fresh_path)

    fresh_hash = _sha256_bytes(fresh_path.read_bytes())
    benchmark_hash = _sha256_bytes(BENCHMARK_OUTPUT.read_bytes())
    assert fresh_hash == benchmark_hash, (
        "the rule-based benchmark no longer matches a fresh run of the unmodified "
        "routesentry/ pipeline -- routesentry/ must not change as part of the ML-driven pivot"
    )


def test_golden_sample_rows_still_hold_after_ml_additions():
    ledger = run(SHIPMENTS, NOTES)
    for route, week, cpkm, vh, vp, flagged, note_id in GOLDEN:
        entry = next(e for e in ledger if e.route == route and e.week_of == week)
        assert round(entry.cost_per_tonne_km, 2) == cpkm
        assert round(entry.vs_own_history_pct, 1) == vh
        assert round(entry.vs_similar_routes_pct, 1) == vp
        assert entry.final_verdict == flagged
        assert (entry.accepted_note_id or "") == note_id


def test_routesentry_package_has_no_ml_imports():
    # routesentry/ must remain independently runnable without scikit-learn
    # or numpy -- a hard import-boundary check that the ML work stayed out.
    import ast

    routesentry_dir = ROOT / "routesentry"
    banned = {"sklearn", "numpy", "pandas", "ml"}
    for path in routesentry_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module.split(".")[0]] if node.module else []
            else:
                continue
            offending = banned & set(names)
            assert not offending, f"{path.name} imports {offending}, which must stay out of routesentry/"
