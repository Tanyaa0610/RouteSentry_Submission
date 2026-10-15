"""Loading and validating the two input CSVs.

Data-quality philosophy: fail loudly on structural problems (missing
columns, unparseable file), but on row-level problems skip-and-log rather
than crash, so one bad row doesn't take down a 2,940-row run. Every skip
is logged with the row's shipment_id and the reason.
"""
from __future__ import annotations

import csv
import logging
from datetime import date, datetime
from pathlib import Path
from typing import List

from routesentry.models import Note, Shipment
from routesentry.text_utils import parse_note_window

logger = logging.getLogger("routesentry.data_io")

REQUIRED_SHIPMENT_COLUMNS = {
    "shipment_id", "origin", "destination", "route_type", "material",
    "quantity_tonnes", "distance_km", "freight_cost_inr", "shipment_date",
    "transporter",
}
REQUIRED_NOTE_COLUMNS = {"note_id", "date", "applies_to", "note"}


class DataValidationError(RuntimeError):
    """Raised for structural problems we refuse to silently work around."""


def load_shipments(path: str | Path) -> List[Shipment]:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"shipment file not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_SHIPMENT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(
                f"shipment_records.csv is missing required columns: {sorted(missing)}"
            )

        shipments: List[Shipment] = []
        skipped = 0
        for i, row in enumerate(reader, start=2):  # header is line 1
            try:
                shipment = _parse_shipment_row(row)
            except (ValueError, KeyError) as exc:
                skipped += 1
                logger.warning(
                    "skipping shipment row %d (id=%s): %s",
                    i, row.get("shipment_id", "?"), exc,
                )
                continue
            shipments.append(shipment)

    logger.info("loaded %d shipment rows (%d skipped) from %s", len(shipments), skipped, path)
    if not shipments:
        raise DataValidationError("no valid shipment rows loaded -- cannot proceed")
    return shipments


def _parse_shipment_row(row: dict) -> Shipment:
    quantity = float(row["quantity_tonnes"])
    distance = float(row["distance_km"])
    cost = float(row["freight_cost_inr"])
    if quantity <= 0:
        raise ValueError(f"non-positive quantity_tonnes={quantity}")
    if distance <= 0:
        raise ValueError(f"non-positive distance_km={distance}")
    if cost <= 0:
        raise ValueError(f"non-positive freight_cost_inr={cost}")

    shipment_date = datetime.strptime(row["shipment_date"].strip(), "%Y-%m-%d").date()

    origin = row["origin"].strip()
    destination = row["destination"].strip()
    route_type = row["route_type"].strip()
    if not origin or not destination:
        raise ValueError("blank origin/destination")
    if route_type not in {"Short", "Medium", "Long"}:
        raise ValueError(f"unrecognised route_type={route_type!r}")

    return Shipment(
        shipment_id=row["shipment_id"].strip(),
        origin=origin,
        destination=destination,
        route_type=route_type,
        material=row["material"].strip(),
        quantity_tonnes=quantity,
        distance_km=distance,
        freight_cost_inr=cost,
        shipment_date=shipment_date,
        transporter=row["transporter"].strip(),
    )


def load_notes(path: str | Path) -> List[Note]:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"notes file not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_NOTE_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise DataValidationError(
                f"context_notes.csv is missing required columns: {sorted(missing)}"
            )

        notes: List[Note] = []
        for i, row in enumerate(reader, start=2):
            try:
                notes.append(_parse_note_row(row))
            except (ValueError, KeyError) as exc:
                logger.warning("skipping note row %d (id=%s): %s", i, row.get("note_id", "?"), exc)

    logger.info("loaded %d context notes from %s", len(notes), path)
    if not notes:
        raise DataValidationError("no valid note rows loaded -- cannot proceed")
    return notes


def _parse_note_row(row: dict) -> Note:
    note_date = datetime.strptime(row["date"].strip(), "%Y-%m-%d").date()
    text = row["note"].strip()
    if not text:
        raise ValueError("blank note text")
    window_start, window_end, source = parse_note_window(text, note_date)
    return Note(
        note_id=row["note_id"].strip(),
        date=note_date,
        applies_to=row["applies_to"].strip(),
        text=text,
        window_start=window_start,
        window_end=window_end,
        window_source=source,
    )
