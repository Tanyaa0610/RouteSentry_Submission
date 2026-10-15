"""Schema and consistency checks for the generated experimental_ml_predictions.csv /
experimental_ml_model_comparison.csv against the real dataset."""
import csv
from pathlib import Path

import pytest

from ml.pipeline import run_full_ml_pipeline
from ml.reporting import PREDICTIONS_COLUMNS, write_comparison_csv, write_predictions_csv

ROOT = Path(__file__).resolve().parent.parent
SHIPMENTS = ROOT / "shipment_records.csv"
NOTES = ROOT / "context_notes.csv"


@pytest.fixture(scope="module")
def pipeline_output():
    return run_full_ml_pipeline(SHIPMENTS, NOTES)


def test_every_route_week_has_exactly_one_prediction_record(pipeline_output):
    keys = [(p.route, p.week_of) for p in pipeline_output.predictions]
    assert len(keys) == len(set(keys))
    assert len(keys) == pipeline_output.n_feature_rows


def test_ml_status_values_are_exactly_the_three_allowed(pipeline_output):
    allowed = {"scored", "insufficient_history", "insufficient_training_data"}
    assert {p.ml_status for p in pipeline_output.predictions} <= allowed


def test_scored_rows_have_both_scores_and_predictions(pipeline_output):
    for p in pipeline_output.predictions:
        if p.ml_status == "scored":
            assert p.if_score is not None and p.if_prediction is not None
            assert p.lof_score is not None and p.lof_prediction is not None
        else:
            assert p.if_score is None and p.lof_score is None


def test_predictions_csv_written_with_exact_contract_columns(tmp_path, pipeline_output):
    out_path = tmp_path / "experimental_ml_predictions.csv"
    write_predictions_csv(pipeline_output.predictions, out_path)
    with out_path.open() as f:
        header = next(csv.reader(f))
    assert header == PREDICTIONS_COLUMNS


def test_predictions_csv_row_count_matches(tmp_path, pipeline_output):
    out_path = tmp_path / "experimental_ml_predictions.csv"
    write_predictions_csv(pipeline_output.predictions, out_path)
    with out_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(pipeline_output.predictions)


def test_comparison_csv_has_all_routes_plus_overall_row(tmp_path, pipeline_output):
    out_path = tmp_path / "experimental_ml_model_comparison.csv"
    write_comparison_csv(pipeline_output.predictions, out_path)
    with out_path.open() as f:
        rows = list(csv.DictReader(f))
    route_names = {r["route"] for r in rows}
    assert "ALL_ROUTES" in route_names
    scored_routes = {p.route for p in pipeline_output.predictions if p.ml_status == "scored"}
    assert scored_routes <= route_names


def test_evidence_reference_present_only_for_baseline_justified_rows(pipeline_output):
    for p in pipeline_output.predictions:
        if p.baseline_verdict == "No (justified)":
            assert p.baseline_matched_note_id
        else:
            assert not p.baseline_matched_note_id
