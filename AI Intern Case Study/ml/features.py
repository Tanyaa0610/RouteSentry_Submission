"""Leakage-free feature construction for the ML anomaly detectors.

Reuses `routesentry.weekly.build_weekly_metrics` (unmodified) for
`cost_per_tonne_km`, `vs_own_history_pct`, `vs_similar_routes_pct`,
`n_prior_weeks`, and `n_peer_routes` -- these are exactly the baseline's
own leakage-free definitions (trailing 8 prior weeks only, same-week peer
average excluding self), so the ML features start from the same ground
truth the rule-based system uses.

Two additional features are computed here, independently of
`routesentry/`, from each route's own strictly-prior weekly cost history:

- `wow_change_pct`: week-over-week change vs. the single most recent prior
  observed week (a different signal than the 8-week average -- catches a
  sudden one-week jump that an 8-week average would dilute).
- `prior_volatility_pct`: coefficient of variation (std / mean * 100) of
  the same trailing prior-week window used for `vs_own_history_pct` -- how
  noisy this route's own history has been, so a model can tell a genuinely
  unusual move from this route's normal noise level.

Never computed or used here: `flagged`, `reason`, `matched_note_id`, any
context-note text, or any shipment dated after the row's own week.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from statistics import pstdev
from typing import Dict, List, Optional

from routesentry.dates import monday_of
from routesentry.models import Shipment
from routesentry.weekly import build_weekly_metrics

from ml.config import MIN_PRIOR_WEEKS_FOR_ML

ROUTE_TYPES = ("Short", "Medium", "Long")


@dataclass
class FeatureRow:
    route: str
    route_type: str
    week_of: date
    cost_per_tonne_km: float
    vs_own_history_pct: Optional[float]
    vs_similar_routes_pct: Optional[float]
    wow_change_pct: Optional[float]
    prior_volatility_pct: Optional[float]
    n_prior_weeks: int
    n_peer_routes: int
    eligible: bool
    ineligibility_reason: str

    def feature_vector(self) -> List[float]:
        """Numeric feature vector for the models. Only called on eligible
        rows, where every field used here is guaranteed non-None."""
        one_hot = [1.0 if self.route_type == t else 0.0 for t in ROUTE_TYPES]
        return [
            self.cost_per_tonne_km,
            self.vs_own_history_pct,
            self.vs_similar_routes_pct,
            self.wow_change_pct,
            self.prior_volatility_pct,
            float(self.n_prior_weeks),
            *one_hot,
        ]


FEATURE_NAMES = [
    "cost_per_tonne_km",
    "vs_own_history_pct",
    "vs_similar_routes_pct",
    "wow_change_pct",
    "prior_volatility_pct",
    "n_prior_weeks",
    "route_type_short",
    "route_type_medium",
    "route_type_long",
]


def _route_week_cost_series(shipments: List[Shipment]) -> Dict[str, Dict[date, float]]:
    """Independent re-aggregation of cost_per_tonne_km per (route, week),
    used only to derive wow_change_pct / prior_volatility_pct. Deliberately
    NOT imported from routesentry/weekly.py so that module stays untouched;
    tests/test_ml_features.py asserts this matches WeeklyMetric.cost_per_tonne_km
    exactly, so the two can never silently diverge."""
    agg: Dict[str, Dict[date, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for s in shipments:
        week = monday_of(s.shipment_date)
        cell = agg[s.route][week]
        cell[0] += s.freight_cost_inr
        cell[1] += s.quantity_tonnes * s.distance_km
    return {
        route: {week: cells[0] / cells[1] for week, cells in weeks.items()}
        for route, weeks in agg.items()
    }


def build_feature_table(shipments: List[Shipment]) -> List[FeatureRow]:
    metrics = build_weekly_metrics(shipments)
    cpkm = _route_week_cost_series(shipments)

    rows: List[FeatureRow] = []
    for m in metrics:
        weeks_sorted = sorted(cpkm[m.route].keys())
        idx = weeks_sorted.index(m.week_of)
        prior_weeks = weeks_sorted[:idx][-8:]  # same trailing window as vs_own_history

        if len(prior_weeks) >= 1:
            wow_change_pct = (m.cost_per_tonne_km / cpkm[m.route][prior_weeks[-1]] - 1) * 100.0
        else:
            wow_change_pct = None

        if len(prior_weeks) >= 2:
            prior_values = [cpkm[m.route][w] for w in prior_weeks]
            mean_prior = sum(prior_values) / len(prior_values)
            prior_volatility_pct = (pstdev(prior_values) / mean_prior * 100.0) if mean_prior else None
        else:
            prior_volatility_pct = None

        eligible = (
            m.n_prior_weeks >= MIN_PRIOR_WEEKS_FOR_ML
            and m.vs_own_history_pct is not None
            and m.vs_similar_routes_pct is not None
            and wow_change_pct is not None
            and prior_volatility_pct is not None
        )
        if eligible:
            reason = ""
        elif m.n_prior_weeks < MIN_PRIOR_WEEKS_FOR_ML:
            reason = f"insufficient_history (n_prior_weeks={m.n_prior_weeks} < {MIN_PRIOR_WEEKS_FOR_ML})"
        elif m.vs_similar_routes_pct is None:
            reason = "no_peer_routes_this_week"
        else:
            reason = "insufficient_history_for_derived_features"

        rows.append(
            FeatureRow(
                route=m.route,
                route_type=m.route_type,
                week_of=m.week_of,
                cost_per_tonne_km=m.cost_per_tonne_km,
                vs_own_history_pct=m.vs_own_history_pct,
                vs_similar_routes_pct=m.vs_similar_routes_pct,
                wow_change_pct=wow_change_pct,
                prior_volatility_pct=prior_volatility_pct,
                n_prior_weeks=m.n_prior_weeks,
                n_peer_routes=m.n_peer_routes,
                eligible=eligible,
                ineligibility_reason=reason,
            )
        )

    rows.sort(key=lambda r: (r.week_of, r.route))
    return rows
