from datetime import date

from routesentry.anomaly import evaluate_signal
from routesentry.models import WeeklyMetric


def _metric(vh, vp, n_prior=8, n_peer=5):
    return WeeklyMetric(
        route="A-B", route_type="Short", week_of=date(2024, 1, 1),
        cost_per_tonne_km=1.0, n_prior_weeks=n_prior, vs_own_history_pct=vh,
        n_peer_routes=n_peer, vs_similar_routes_pct=vp,
    )


def test_no_history_is_not_an_anomaly():
    m = _metric(vh=None, vp=50.0, n_prior=0)
    sig = evaluate_signal(m)
    assert sig.is_anomaly is False


def test_falling_cost_is_never_an_anomaly_even_if_peer_gap_huge():
    m = _metric(vh=-5.0, vp=90.0)
    sig = evaluate_signal(m)
    assert sig.is_anomaly is False


def test_large_rise_vs_own_history_alone_triggers():
    m = _metric(vh=20.0, vp=None)
    sig = evaluate_signal(m)
    assert sig.is_anomaly is True


def test_small_rise_with_large_peer_gap_triggers():
    m = _metric(vh=5.0, vp=25.0)
    sig = evaluate_signal(m)
    assert sig.is_anomaly is True


def test_small_rise_with_small_peer_gap_does_not_trigger():
    m = _metric(vh=5.0, vp=5.0)
    sig = evaluate_signal(m)
    assert sig.is_anomaly is False


def test_threshold_boundary_is_strictly_greater_than():
    m = _metric(vh=15.0, vp=None)  # exactly at threshold
    sig = evaluate_signal(m, threshold=15.0)
    assert sig.is_anomaly is False  # must be strictly greater, not >=
    m2 = _metric(vh=15.0001, vp=None)
    assert evaluate_signal(m2, threshold=15.0).is_anomaly is True
