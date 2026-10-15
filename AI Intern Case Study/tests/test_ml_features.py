"""Feature construction: consistency with the baseline's own cost
calculation, correct eligibility gating, and leakage-free derived
features (wow_change_pct, prior_volatility_pct)."""
from datetime import date

from ml.config import MIN_PRIOR_WEEKS_FOR_ML
from ml.features import FEATURE_NAMES, build_feature_table
from routesentry.weekly import build_weekly_metrics


def test_cost_per_tonne_km_matches_baseline_exactly(shipment_factory):
    ships = [
        shipment_factory(shipment_id=f"S{i}", shipment_date=date(2024, 1, 1 + 7 * (i % 3)),
                          quantity_tonnes=10 + i, distance_km=100 + i, freight_cost_inr=1000 + 50 * i)
        for i in range(9)
    ]
    rows = build_feature_table(ships)
    metrics = build_weekly_metrics(ships)
    metrics_by_key = {(m.route, m.week_of): m for m in metrics}
    for r in rows:
        m = metrics_by_key[(r.route, r.week_of)]
        assert abs(r.cost_per_tonne_km - m.cost_per_tonne_km) < 1e-9
        assert r.vs_own_history_pct == m.vs_own_history_pct
        assert r.vs_similar_routes_pct == m.vs_similar_routes_pct
        assert r.n_prior_weeks == m.n_prior_weeks


def test_feature_vector_length_matches_feature_names(shipment_factory):
    base = date(2024, 1, 1)
    ships = []
    for i in range(9):
        w = date.fromordinal(base.toordinal() + 7 * i)
        ships.append(shipment_factory(shipment_id=f"S{i}", shipment_date=w))
        ships.append(shipment_factory(shipment_id=f"P{i}", origin="Delhi", destination="Jaipur", shipment_date=w))
    rows = build_feature_table(ships)
    eligible = [r for r in rows if r.eligible]
    assert eligible
    for r in eligible:
        assert len(r.feature_vector()) == len(FEATURE_NAMES)


def test_ineligible_below_min_prior_weeks(shipment_factory):
    base = date(2024, 1, 1)
    # Only MIN_PRIOR_WEEKS_FOR_ML weeks total -> the last week has exactly
    # MIN_PRIOR_WEEKS_FOR_ML - 1 prior weeks, which must be ineligible.
    ships = [
        shipment_factory(shipment_id=f"S{i}", shipment_date=date.fromordinal(base.toordinal() + 7 * i))
        for i in range(MIN_PRIOR_WEEKS_FOR_ML)
    ]
    rows = build_feature_table(ships)
    last = max(rows, key=lambda r: r.week_of)
    assert last.eligible is False
    assert "insufficient_history" in last.ineligibility_reason


def test_eligible_once_min_prior_weeks_reached(shipment_factory):
    base = date(2024, 1, 1)
    ships = []
    for i in range(MIN_PRIOR_WEEKS_FOR_ML + 1):
        w = date.fromordinal(base.toordinal() + 7 * i)
        ships.append(shipment_factory(shipment_id=f"S{i}", shipment_date=w))
        ships.append(shipment_factory(shipment_id=f"P{i}", origin="Delhi", destination="Jaipur", shipment_date=w))
    rows = build_feature_table(ships)
    mumbai_rows = [r for r in rows if r.route == "Mumbai-Pune"]
    last = max(mumbai_rows, key=lambda r: r.week_of)
    assert last.eligible is True
    assert last.ineligibility_reason == ""


def test_no_peer_route_makes_row_ineligible(shipment_factory):
    base = date(2024, 1, 1)
    # A single, isolated route with plenty of history but no peers of the
    # same route_type anywhere in the dataset.
    ships = [
        shipment_factory(shipment_id=f"S{i}", route_type="Short", origin="X", destination="Y",
                          shipment_date=date.fromordinal(base.toordinal() + 7 * i))
        for i in range(6)
    ]
    rows = build_feature_table(ships)
    last = max(rows, key=lambda r: r.week_of)
    assert last.eligible is False
    assert "peer" in last.ineligibility_reason


def test_wow_change_pct_is_vs_single_most_recent_prior_week(shipment_factory):
    base = date(2024, 1, 1)
    weeks = [date.fromordinal(base.toordinal() + 7 * i) for i in range(6)]
    # cost doubles only in the final week; wow_change should reflect the
    # jump from the immediately preceding week, not the 8-week average.
    costs = [1000, 1000, 1000, 1000, 1000, 2000]
    ships = [
        shipment_factory(shipment_id=f"S{i}", shipment_date=w, quantity_tonnes=10, distance_km=100, freight_cost_inr=c)
        for i, (w, c) in enumerate(zip(weeks, costs))
    ]
    rows = build_feature_table(ships)
    last = max(rows, key=lambda r: r.week_of)
    assert abs(last.wow_change_pct - 100.0) < 1e-9  # 2000/1000*100 - 100


def test_prior_volatility_is_zero_for_constant_history(shipment_factory):
    base = date(2024, 1, 1)
    weeks = [date.fromordinal(base.toordinal() + 7 * i) for i in range(6)]
    ships = [
        shipment_factory(shipment_id=f"S{i}", shipment_date=w, quantity_tonnes=10, distance_km=100, freight_cost_inr=1000)
        for i, w in enumerate(weeks)
    ]
    rows = build_feature_table(ships)
    last = max(rows, key=lambda r: r.week_of)
    assert abs(last.prior_volatility_pct - 0.0) < 1e-9


def test_features_never_use_future_weeks(shipment_factory):
    # Two parallel histories that are identical up to week 5; the only
    # difference is what happens in week 6 (a huge spike for route A).
    # Every feature computed for week 5 must be identical between A and B,
    # since nothing at week 6 should leak backward.
    base = date(2024, 1, 1)
    weeks = [date.fromordinal(base.toordinal() + 7 * i) for i in range(6)]
    ships = []
    for i, w in enumerate(weeks[:5]):
        ships.append(shipment_factory(shipment_id=f"A{i}", origin="A", destination="B",
                                       shipment_date=w, quantity_tonnes=10, distance_km=100, freight_cost_inr=1000))
        ships.append(shipment_factory(shipment_id=f"C{i}", origin="C", destination="D",
                                       shipment_date=w, quantity_tonnes=10, distance_km=100, freight_cost_inr=1000))
    # Week 6: route A-B spikes, route C-D stays flat.
    ships.append(shipment_factory(shipment_id="A5", origin="A", destination="B",
                                   shipment_date=weeks[5], quantity_tonnes=10, distance_km=100, freight_cost_inr=9000))
    ships.append(shipment_factory(shipment_id="C5", origin="C", destination="D",
                                   shipment_date=weeks[5], quantity_tonnes=10, distance_km=100, freight_cost_inr=1000))

    rows = build_feature_table(ships)
    week5_a = next(r for r in rows if r.route == "A-B" and r.week_of == weeks[4])
    week5_c = next(r for r in rows if r.route == "C-D" and r.week_of == weeks[4])
    assert week5_a.cost_per_tonne_km == week5_c.cost_per_tonne_km
    assert week5_a.vs_own_history_pct == week5_c.vs_own_history_pct
    assert week5_a.wow_change_pct == week5_c.wow_change_pct
    assert week5_a.prior_volatility_pct == week5_c.prior_volatility_pct
