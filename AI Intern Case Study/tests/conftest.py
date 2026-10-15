"""Shared fixtures: small synthetic shipment/note sets that exercise one
behaviour at a time, independent of the real 2,940-row dataset."""
from __future__ import annotations

from datetime import date

import pytest

from routesentry.models import Shipment
from routesentry.text_utils import parse_note_window
from routesentry.models import Note


def make_shipment(
    shipment_id="SHP0001",
    origin="Mumbai",
    destination="Pune",
    route_type="Short",
    material="Steel",
    quantity_tonnes=10.0,
    distance_km=100.0,
    freight_cost_inr=1000.0,
    shipment_date=date(2024, 1, 1),
    transporter="Test Transport",
) -> Shipment:
    return Shipment(
        shipment_id=shipment_id,
        origin=origin,
        destination=destination,
        route_type=route_type,
        material=material,
        quantity_tonnes=quantity_tonnes,
        distance_km=distance_km,
        freight_cost_inr=freight_cost_inr,
        shipment_date=shipment_date,
        transporter=transporter,
    )


def make_note(note_id="N001", note_date=date(2024, 1, 1), applies_to="Mumbai-Pune", text="Test note text.") -> Note:
    window_start, window_end, source = parse_note_window(text, note_date)
    return Note(
        note_id=note_id,
        date=note_date,
        applies_to=applies_to,
        text=text,
        window_start=window_start,
        window_end=window_end,
        window_source=source,
    )


@pytest.fixture
def shipment_factory():
    return make_shipment


@pytest.fixture
def note_factory():
    return make_note
