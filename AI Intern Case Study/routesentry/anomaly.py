"""Statistical signal: is this route-week unusual, on its own numbers,
before any evidence is consulted?

This is deliberately separate from the final verdict (see verdict.py) --
an anomaly is not yet a "justified" or "unexplained" anomaly, it's just a
candidate. See config.py for the full rationale behind the policy below.
"""
from __future__ import annotations

from routesentry.config import RISE_THRESHOLD_PCT
from routesentry.models import StatisticalSignal, WeeklyMetric


def evaluate_signal(metric: WeeklyMetric, threshold: float = RISE_THRESHOLD_PCT) -> StatisticalSignal:
    vh = metric.vs_own_history_pct
    vp = metric.vs_similar_routes_pct

    if vh is None:
        return StatisticalSignal(
            is_anomaly=False,
            rule_triggered=(
                "not evaluated: no prior weeks available for this route yet, "
                "so 'rising vs. own history' cannot be assessed"
            ),
        )

    rising = vh > 0
    if not rising:
        return StatisticalSignal(
            is_anomaly=False,
            rule_triggered=f"not rising: vs_own_history={vh:.1f}% <= 0%",
        )

    if vh > threshold:
        return StatisticalSignal(
            is_anomaly=True,
            rule_triggered=f"vs_own_history={vh:.1f}% > {threshold:.0f}% threshold",
        )

    if vp is not None and vp > threshold:
        return StatisticalSignal(
            is_anomaly=True,
            rule_triggered=(
                f"rising vs. own history ({vh:.1f}%) and vs_similar_routes={vp:.1f}% "
                f"> {threshold:.0f}% threshold"
            ),
        )

    return StatisticalSignal(
        is_anomaly=False,
        rule_triggered=(
            f"rising but within normal range: vs_own_history={vh:.1f}%, "
            f"vs_similar_routes={'n/a' if vp is None else f'{vp:.1f}%'} "
            f"(threshold {threshold:.0f}%)"
        ),
    )
