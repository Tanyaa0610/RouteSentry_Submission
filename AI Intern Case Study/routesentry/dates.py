"""Shared week-boundary helpers (Monday-Sunday weeks, per the brief)."""
from __future__ import annotations

from datetime import date, timedelta


def monday_of(d: date) -> date:
    """The Monday that starts the Mon-Sun week containing d."""
    return d - timedelta(days=d.weekday())


def sunday_of(d: date) -> date:
    """The Sunday that ends the Mon-Sun week containing d."""
    return monday_of(d) + timedelta(days=6)


def ranges_overlap(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    return a_start <= b_end and b_start <= a_end
