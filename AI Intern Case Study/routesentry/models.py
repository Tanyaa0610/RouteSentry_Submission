"""Typed records shared across the pipeline stages.

Keeping these as plain dataclasses (rather than dicts) makes the boundary
between computation / retrieval / validation / reporting explicit: each
stage consumes one record type and produces the next.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class Shipment:
    shipment_id: str
    origin: str
    destination: str
    route_type: str
    material: str
    quantity_tonnes: float
    distance_km: float
    freight_cost_inr: float
    shipment_date: date
    transporter: str

    @property
    def route(self) -> str:
        return f"{self.origin}-{self.destination}"


@dataclass(frozen=True)
class Note:
    note_id: str
    date: date
    applies_to: str  # route name, or "All Routes"
    text: str
    window_start: date
    window_end: date
    window_source: str  # "explicit" (parsed "from X to Y") or "default" (note's own week)


@dataclass
class WeeklyMetric:
    route: str
    route_type: str
    week_of: date  # Monday of the ISO week
    cost_per_tonne_km: float
    n_prior_weeks: int  # weeks used for the trailing-history baseline (0..8)
    vs_own_history_pct: Optional[float]  # None iff n_prior_weeks == 0
    n_peer_routes: int
    vs_similar_routes_pct: Optional[float]  # None iff no peer routes reported this week


@dataclass
class StatisticalSignal:
    is_anomaly: bool
    rule_triggered: str  # human-readable rule description for the ledger


@dataclass
class CandidateEvidence:
    note_id: str
    note_date: date
    note_text: str
    similarity: float
    retrieval_source: str  # "semantic" | "deterministic_eligibility" | "both"
    route_match: bool
    date_match: bool
    direction: str  # "cost_increase" | "no_impact" | "improved" | "excluded"
    direction_phrase: str  # the matched phrase driving the direction classification
    relation: str  # "supports" | "contradicts" | "irrelevant"
    rejection_reason: str  # empty string if relation == "supports"


@dataclass
class LedgerEntry:
    route: str
    route_type: str
    week_of: date
    cost_per_tonne_km: float
    n_prior_weeks: int
    vs_own_history_pct: Optional[float]
    n_peer_routes: int
    vs_similar_routes_pct: Optional[float]
    statistical_signal: bool
    signal_rule: str
    candidates: list = field(default_factory=list)  # list[CandidateEvidence]
    accepted_note_id: Optional[str] = None
    final_verdict: str = "No"  # "Yes" | "No" | "No (justified)"
    reason: str = ""
    insufficiency_note: str = ""  # explicit statement when evidence is insufficient, not guessed
