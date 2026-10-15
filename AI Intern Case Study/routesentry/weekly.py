"""Core computation: grouping, weekly cost-per-tonne-km, and the two
prescribed baselines. This module implements section "Comparison
baselines, precisely" from the brief EXACTLY as specified -- see
docs/requirements_traceability.md for the line-by-line mapping.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date
from typing import Dict, List, Tuple

from routesentry.config import TRAILING_WEEKS
from routesentry.dates import monday_of
from routesentry.models import Shipment, WeeklyMetric

logger = logging.getLogger("routesentry.weekly")


def build_weekly_metrics(shipments: List[Shipment]) -> List[WeeklyMetric]:
    # route -> week_of -> [cost_sum, tonne_km_sum]
    agg: Dict[str, Dict[date, List[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    route_type_of: Dict[str, str] = {}

    for s in shipments:
        week = monday_of(s.shipment_date)
        cell = agg[s.route][week]
        cell[0] += s.freight_cost_inr
        cell[1] += s.quantity_tonnes * s.distance_km
        route_type_of[s.route] = s.route_type  # dataset: route_type is 1:1 with route

    # route -> week_of -> cost_per_tonne_km (needed for both baselines)
    cpkm: Dict[str, Dict[date, float]] = {
        route: {week: cells[0] / cells[1] for week, cells in weeks.items()}
        for route, weeks in agg.items()
    }

    metrics: List[WeeklyMetric] = []
    for route in sorted(agg.keys()):
        route_type = route_type_of[route]
        weeks_sorted = sorted(agg[route].keys())
        for i, week in enumerate(weeks_sorted):
            current = cpkm[route][week]

            # vs. own history: trailing 8-week rolling average, weeks strictly
            # before the current week. No look-ahead. If fewer than 8 prior
            # weeks exist, use all available (documented in README; also
            # surfaced per-row via n_prior_weeks).
            prior_weeks = weeks_sorted[:i][-TRAILING_WEEKS:]
            if prior_weeks:
                hist_avg = sum(cpkm[route][w] for w in prior_weeks) / len(prior_weeks)
                vs_hist = (current - hist_avg) / hist_avg * 100.0
            else:
                vs_hist = None

            # vs. similar routes: same week, same route_type, excluding this
            # route itself from its own peer average.
            peer_values = [
                cpkm[other][week]
                for other in agg
                if other != route and route_type_of[other] == route_type and week in cpkm[other]
            ]
            if peer_values:
                peer_avg = sum(peer_values) / len(peer_values)
                vs_peer = (current - peer_avg) / peer_avg * 100.0
            else:
                vs_peer = None

            metrics.append(
                WeeklyMetric(
                    route=route,
                    route_type=route_type,
                    week_of=week,
                    cost_per_tonne_km=current,
                    n_prior_weeks=len(prior_weeks),
                    vs_own_history_pct=vs_hist,
                    n_peer_routes=len(peer_values),
                    vs_similar_routes_pct=vs_peer,
                )
            )

    n_insufficient_history = sum(1 for m in metrics if m.n_prior_weeks < TRAILING_WEEKS)
    logger.info(
        "built %d route-week metrics across %d routes; %d rows have < %d prior weeks of history",
        len(metrics), len(agg), n_insufficient_history, TRAILING_WEEKS,
    )
    return metrics
