"""In-process reproducibility of the full experimental ML comparison pipeline
(fixed-seed repeatability). The 3-process CLI-level check lives in
scripts/check_reproducibility.py --experimental-ml /
docs/experimental_ml_comparison_reproducibility_report.md."""
from pathlib import Path

import pytest

from ml.pipeline import run_full_ml_pipeline

ROOT = Path(__file__).resolve().parent.parent
SHIPMENTS = ROOT / "shipment_records.csv"
NOTES = ROOT / "context_notes.csv"


@pytest.fixture(scope="module")
def two_runs():
    return run_full_ml_pipeline(SHIPMENTS, NOTES), run_full_ml_pipeline(SHIPMENTS, NOTES)


def test_two_full_runs_produce_identical_scores_and_predictions(two_runs):
    out_a, out_b = two_runs
    assert len(out_a.predictions) == len(out_b.predictions)
    for a, b in zip(out_a.predictions, out_b.predictions):
        assert a.route == b.route
        assert a.week_of == b.week_of
        assert a.ml_status == b.ml_status
        assert a.if_score == b.if_score
        assert a.if_prediction == b.if_prediction
        assert a.lof_score == b.lof_score
        assert a.lof_prediction == b.lof_prediction


def test_two_full_runs_produce_identical_synthetic_results(two_runs):
    out_a, out_b = two_runs
    assert out_a.synthetic == out_b.synthetic


def test_two_full_runs_produce_identical_sensitivity_results(two_runs):
    out_a, out_b = two_runs
    assert out_a.sensitivity == out_b.sensitivity
