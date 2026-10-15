"""Aggregation and baseline tests -- the core numeric contract of the brief."""
from datetime import date

from routesentry.weekly import build_weekly_metrics


def _m(shipment_factory, week_monday, day_offset, route="A-B", route_type="Short", qty=10.0, dist=100.0, cost=1000.0):
    return shipment_factory(
        shipment_id=f"SHP-{week_monday}-{day_offset}",
        origin=route.split("-")[0],
        destination=route.split("-")[1],
        route_type=route_type,
        shipment_date=date.fromordinal(week_monday.toordinal() + day_offset),
        quantity_tonnes=qty,
        distance_km=dist,
        freight_cost_inr=cost,
    )


def test_ratio_of_sums_not_average_of_ratios(shipment_factory):
    # Two shipments in the same route-week with different individual
    # cost/tonne-km ratios. The aggregate must be sum(cost)/sum(tonne_km),
    # NOT the mean of the two per-shipment ratios.
    week = date(2024, 1, 1)
    s1 = _m(shipment_factory, week, 0, qty=10.0, dist=100.0, cost=1000.0)  # ratio 1.0
    s2 = _m(shipment_factory, week, 1, qty=5.0, dist=100.0, cost=250.0)  # ratio 0.5
    metrics = build_weekly_metrics([s1, s2])
    assert len(metrics) == 1
    expected = (1000.0 + 250.0) / (10.0 * 100.0 + 5.0 * 100.0)
    assert abs(metrics[0].cost_per_tonne_km - expected) < 1e-9
    naive_average = (1.0 + 0.5) / 2
    assert abs(metrics[0].cost_per_tonne_km - naive_average) > 1e-9


def test_monday_to_sunday_grouping(shipment_factory):
    week = date(2024, 1, 1)  # Monday
    monday_ship = _m(shipment_factory, week, 0)
    sunday_ship = _m(shipment_factory, week, 6)
    next_monday_ship = _m(shipment_factory, week, 7)
    metrics = build_weekly_metrics([monday_ship, sunday_ship, next_monday_ship])
    weeks = sorted({m.week_of for m in metrics})
    assert weeks == [date(2024, 1, 1), date(2024, 1, 8)]


def test_no_current_week_leakage_in_history(shipment_factory):
    # Route with two weeks of data; week 2's history baseline must use
    # only week 1's value, never include week 2 itself.
    week1 = date(2024, 1, 1)
    week2 = date(2024, 1, 8)
    s1 = _m(shipment_factory, week1, 0, cost=1000.0, qty=10, dist=100)  # cpkm=1.0
    s2 = _m(shipment_factory, week2, 0, cost=2000.0, qty=10, dist=100)  # cpkm=2.0
    metrics = build_weekly_metrics([s1, s2])
    m2 = next(m for m in metrics if m.week_of == week2)
    assert m2.n_prior_weeks == 1
    assert abs(m2.vs_own_history_pct - 100.0) < 1e-9  # (2.0/1.0 - 1) * 100


def test_no_future_data_leakage(shipment_factory):
    # A later, higher-cost week must not influence an earlier week's baseline.
    week1 = date(2024, 1, 1)
    week2 = date(2024, 1, 8)
    s1 = _m(shipment_factory, week1, 0, cost=1000.0, qty=10, dist=100)
    s2 = _m(shipment_factory, week2, 0, cost=9000.0, qty=10, dist=100)
    metrics = build_weekly_metrics([s1, s2])
    m1 = next(m for m in metrics if m.week_of == week1)
    assert m1.n_prior_weeks == 0
    assert m1.vs_own_history_pct is None


def test_fewer_than_eight_prior_weeks_uses_all_available(shipment_factory):
    weeks = [date(2024, 1, 1 + 7 * i) for i in range(4)]  # 4 weeks total
    ships = [_m(shipment_factory, w, 0, cost=1000.0 * (i + 1), qty=10, dist=100) for i, w in enumerate(weeks)]
    metrics = build_weekly_metrics(ships)
    last = max(metrics, key=lambda m: m.week_of)
    assert last.n_prior_weeks == 3  # all 3 prior weeks, not padded to 8


def test_trailing_window_caps_at_eight_most_recent(shipment_factory):
    base = date(2024, 1, 1)
    weeks = [date.fromordinal(base.toordinal() + 7 * i) for i in range(10)]  # 10 weeks
    # cost rises 1000, 2000, ..., 10000 so each week's cpkm == i+1 (qty*dist=1000 => cpkm = cost/1000)
    ships = [_m(shipment_factory, w, 0, cost=1000.0 * (i + 1), qty=10, dist=100) for i, w in enumerate(weeks)]
    metrics = build_weekly_metrics(ships)
    last = max(metrics, key=lambda m: m.week_of)  # week index 9, cpkm = 10
    assert last.n_prior_weeks == 8
    # trailing 8 prior weeks are indices 1..8 (cpkm 2..9), average = 5.5
    expected_hist_avg = sum(range(2, 10)) / 8
    expected_vh = (10.0 / expected_hist_avg - 1) * 100
    assert abs(last.vs_own_history_pct - expected_vh) < 1e-6


def test_peer_average_excludes_self_and_other_types(shipment_factory):
    week = date(2024, 1, 1)
    a = _m(shipment_factory, week, 0, route="A-B", route_type="Short", cost=1000.0, qty=10, dist=100)  # cpkm=1.0
    b = _m(shipment_factory, week, 0, route="C-D", route_type="Short", cost=3000.0, qty=10, dist=100)  # cpkm=3.0
    c = _m(shipment_factory, week, 0, route="E-F", route_type="Long", cost=9000.0, qty=10, dist=100)  # cpkm=9.0, different type
    metrics = build_weekly_metrics([a, b, c])
    m_a = next(m for m in metrics if m.route == "A-B")
    # peer avg for A-B should be just B's cpkm (3.0): Long-route C must be excluded,
    # and A itself must be excluded from its own peer average.
    assert m_a.n_peer_routes == 1
    assert abs(m_a.vs_similar_routes_pct - (1.0 / 3.0 - 1) * 100) < 1e-9


def test_no_peers_available_leaves_peer_comparison_unset(shipment_factory):
    week = date(2024, 1, 1)
    only_route = _m(shipment_factory, week, 0, route="A-B", route_type="Short")
    metrics = build_weekly_metrics([only_route])
    assert metrics[0].n_peer_routes == 0
    assert metrics[0].vs_similar_routes_pct is None


def test_deterministic_output_order(shipment_factory):
    week = date(2024, 1, 1)
    ships = [
        _m(shipment_factory, week, 0, route="Z-Y", route_type="Short"),
        _m(shipment_factory, week, 0, route="A-B", route_type="Short"),
    ]
    m1 = build_weekly_metrics(ships)
    m2 = build_weekly_metrics(list(reversed(ships)))
    assert [m.route for m in m1] == [m.route for m in m2] == ["A-B", "Z-Y"]
