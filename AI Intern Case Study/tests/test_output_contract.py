"""Validates the submission CSV against the exact schema contract in
sample_output_format_v2.csv: column names, order, verdict vocabulary,
note-id consistency, and deterministic sorting.

Tests routesentry.pipeline.run directly -- the rule-based path, still
fully intact and selectable via `run.py --mode rule_based`. The ML-driven
default path (run.py's new default) is schema-tested separately in
tests/test_ml_official_pipeline.py, against the same contract."""
import csv
from pathlib import Path

import pytest

from routesentry.pipeline import run

ROOT = Path(__file__).resolve().parent.parent
SHIPMENTS = ROOT / "shipment_records.csv"
NOTES = ROOT / "context_notes.csv"
SAMPLE = ROOT / "sample_output_format_v2.csv"


@pytest.fixture(scope="module")
def ledger():
    return run(SHIPMENTS, NOTES)


def test_sample_format_columns_match_config():
    with SAMPLE.open() as f:
        sample_cols = next(csv.reader(f))
    from routesentry import config

    assert sample_cols == config.OUTPUT_COLUMNS


def test_every_route_week_present_exactly_once(ledger):
    import itertools

    keys = [(e.route, e.week_of) for e in ledger]
    assert len(keys) == len(set(keys))


def test_verdict_vocabulary_is_exactly_the_contract_values(ledger):
    allowed = {"Yes", "No", "No (justified)"}
    assert {e.final_verdict for e in ledger} <= allowed


def test_justified_rows_always_have_a_note_id(ledger):
    for e in ledger:
        if e.final_verdict == "No (justified)":
            assert e.accepted_note_id, f"{e.route} {e.week_of} justified but no note id"


def test_unexplained_and_normal_rows_never_have_a_note_id(ledger):
    for e in ledger:
        if e.final_verdict in ("Yes", "No"):
            assert not e.accepted_note_id, f"{e.route} {e.week_of} has note id but verdict is {e.final_verdict}"


def test_accepted_note_id_is_always_a_real_note_id(ledger):
    real_ids = {row["note_id"] for row in csv.DictReader(NOTES.open())}
    for e in ledger:
        if e.accepted_note_id:
            assert e.accepted_note_id in real_ids


def test_output_sorted_by_route_then_week(ledger):
    pairs = [(e.route, e.week_of) for e in ledger]
    assert pairs == sorted(pairs)


def test_row_count_matches_distinct_route_weeks_in_input():
    import csv as _csv
    from datetime import datetime

    from routesentry.dates import monday_of

    route_weeks = set()
    with SHIPMENTS.open() as f:
        for row in _csv.DictReader(f):
            d = datetime.strptime(row["shipment_date"], "%Y-%m-%d").date()
            route = f"{row['origin']}-{row['destination']}"
            route_weeks.add((route, monday_of(d)))

    ledger_obj = run(SHIPMENTS, NOTES)
    assert len(ledger_obj) == len(route_weeks)
