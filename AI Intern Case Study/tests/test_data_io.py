import csv

import pytest

from routesentry.data_io import DataValidationError, load_notes, load_shipments

SHIPMENT_HEADER = "shipment_id,origin,destination,route_type,material,quantity_tonnes,distance_km,freight_cost_inr,shipment_date,transporter"
NOTE_HEADER = "note_id,date,applies_to,note"


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_shipment_file_raises(tmp_path):
    with pytest.raises(DataValidationError):
        load_shipments(tmp_path / "nope.csv")


def test_missing_required_column_raises(tmp_path):
    p = _write(tmp_path / "s.csv", "shipment_id,origin\nSHP1,Mumbai\n")
    with pytest.raises(DataValidationError):
        load_shipments(p)


def test_nonpositive_quantity_row_is_skipped_not_crashed(tmp_path):
    rows = [
        SHIPMENT_HEADER,
        "SHP1,Mumbai,Pune,Short,Steel,0,100,1000,2024-01-01,T1",  # invalid: qty=0
        "SHP2,Mumbai,Pune,Short,Steel,10,100,1000,2024-01-01,T1",  # valid
    ]
    p = _write(tmp_path / "s.csv", "\n".join(rows) + "\n")
    shipments = load_shipments(p)
    assert len(shipments) == 1
    assert shipments[0].shipment_id == "SHP2"


def test_nonpositive_distance_row_is_skipped(tmp_path):
    rows = [
        SHIPMENT_HEADER,
        "SHP1,Mumbai,Pune,Short,Steel,10,-5,1000,2024-01-01,T1",
        "SHP2,Mumbai,Pune,Short,Steel,10,100,1000,2024-01-01,T1",
    ]
    p = _write(tmp_path / "s.csv", "\n".join(rows) + "\n")
    shipments = load_shipments(p)
    assert len(shipments) == 1


def test_nonpositive_cost_row_is_skipped(tmp_path):
    rows = [
        SHIPMENT_HEADER,
        "SHP1,Mumbai,Pune,Short,Steel,10,100,0,2024-01-01,T1",
        "SHP2,Mumbai,Pune,Short,Steel,10,100,1000,2024-01-01,T1",
    ]
    p = _write(tmp_path / "s.csv", "\n".join(rows) + "\n")
    shipments = load_shipments(p)
    assert len(shipments) == 1


def test_invalid_date_row_is_skipped(tmp_path):
    rows = [
        SHIPMENT_HEADER,
        "SHP1,Mumbai,Pune,Short,Steel,10,100,1000,not-a-date,T1",
        "SHP2,Mumbai,Pune,Short,Steel,10,100,1000,2024-01-01,T1",
    ]
    p = _write(tmp_path / "s.csv", "\n".join(rows) + "\n")
    shipments = load_shipments(p)
    assert len(shipments) == 1


def test_unrecognised_route_type_row_is_skipped(tmp_path):
    rows = [
        SHIPMENT_HEADER,
        "SHP1,Mumbai,Pune,Extreme,Steel,10,100,1000,2024-01-01,T1",
        "SHP2,Mumbai,Pune,Short,Steel,10,100,1000,2024-01-01,T1",
    ]
    p = _write(tmp_path / "s.csv", "\n".join(rows) + "\n")
    shipments = load_shipments(p)
    assert len(shipments) == 1


def test_all_rows_invalid_raises(tmp_path):
    rows = [
        SHIPMENT_HEADER,
        "SHP1,Mumbai,Pune,Short,Steel,0,100,1000,2024-01-01,T1",
    ]
    p = _write(tmp_path / "s.csv", "\n".join(rows) + "\n")
    with pytest.raises(DataValidationError):
        load_shipments(p)


def test_duplicate_shipment_ids_are_both_loaded_but_detectable(tmp_path):
    # We do not silently drop duplicates; loading keeps both rows so a
    # caller can detect/report the duplication explicitly.
    rows = [
        SHIPMENT_HEADER,
        "SHP1,Mumbai,Pune,Short,Steel,10,100,1000,2024-01-01,T1",
        "SHP1,Mumbai,Pune,Short,Steel,10,100,1000,2024-01-02,T1",
    ]
    p = _write(tmp_path / "s.csv", "\n".join(rows) + "\n")
    shipments = load_shipments(p)
    ids = [s.shipment_id for s in shipments]
    assert ids.count("SHP1") == 2


def test_load_notes_missing_column_raises(tmp_path):
    p = _write(tmp_path / "n.csv", "note_id,date\nN1,2024-01-01\n")
    with pytest.raises(DataValidationError):
        load_notes(p)


def test_load_notes_blank_text_row_is_skipped(tmp_path):
    rows = [
        NOTE_HEADER,
        'N1,2024-01-01,All Routes,""',
        'N2,2024-01-01,All Routes,"A valid note about a surcharge."',
    ]
    p = _write(tmp_path / "n.csv", "\n".join(rows) + "\n")
    notes = load_notes(p)
    assert len(notes) == 1
    assert notes[0].note_id == "N2"
