"""ml/signal.py: the ML-driven replacement for routesentry/anomaly.py.
Mirrors tests/test_anomaly.py's cases so the two signal sources are held
to the same standard -- most importantly, the "must be rising" gate is
NOT something the ML model is allowed to override."""
from datetime import date

from ml.chronological import MLResult
from ml.signal import evaluate_ml_signal
from routesentry.models import WeeklyMetric


def _metric(vh, vp, n_prior=8, n_peer=5):
    return WeeklyMetric(
        route="A-B", route_type="Short", week_of=date(2024, 1, 1),
        cost_per_tonne_km=1.0, n_prior_weeks=n_prior, vs_own_history_pct=vh,
        n_peer_routes=n_peer, vs_similar_routes_pct=vp,
    )


def _ml_result(eligible=True, excluded=False, train_size=100, lof_score=5.0, lof_prediction=False):
    return MLResult(
        route="A-B", week_of=date(2024, 1, 1), eligible=eligible, ineligibility_reason="",
        fold_index=0, train_size=train_size, excluded_insufficient_training=excluded,
        if_score=None, if_prediction=None, lof_score=lof_score, lof_prediction=lof_prediction,
    )


def test_no_history_is_not_evaluated():
    sig = evaluate_ml_signal(_metric(vh=None, vp=50.0, n_prior=0), _ml_result(), contamination=0.02)
    assert sig.is_anomaly is False


def test_falling_cost_is_never_flagged_regardless_of_lof_score():
    # This is the core guardrail of the ML-driven design: the model is
    # undirected, so the "rising" gate must be enforced structurally here,
    # not left to the model to (potentially) get wrong.
    sig = evaluate_ml_signal(_metric(vh=-5.0, vp=90.0), _ml_result(lof_prediction=True, lof_score=50.0), contamination=0.02)
    assert sig.is_anomaly is False
    assert "not rising" in sig.rule_triggered


def test_rising_with_lof_prediction_true_is_flagged():
    sig = evaluate_ml_signal(_metric(vh=20.0, vp=10.0), _ml_result(lof_prediction=True, lof_score=8.0), contamination=0.02)
    assert sig.is_anomaly is True
    assert "LOF" in sig.rule_triggered


def test_rising_with_lof_prediction_false_is_not_flagged():
    sig = evaluate_ml_signal(_metric(vh=5.0, vp=5.0), _ml_result(lof_prediction=False, lof_score=1.0), contamination=0.02)
    assert sig.is_anomaly is False


def test_ineligible_ml_result_is_not_evaluated():
    sig = evaluate_ml_signal(_metric(vh=20.0, vp=10.0), _ml_result(eligible=False), contamination=0.02)
    assert sig.is_anomaly is False
    assert "insufficient history" in sig.rule_triggered


def test_insufficient_training_data_is_not_evaluated():
    sig = evaluate_ml_signal(_metric(vh=20.0, vp=10.0), _ml_result(excluded=True, lof_score=None, lof_prediction=None), contamination=0.02)
    assert sig.is_anomaly is False
    assert "pooled training rows" in sig.rule_triggered


def test_missing_ml_result_entirely_is_not_evaluated():
    sig = evaluate_ml_signal(_metric(vh=20.0, vp=10.0), None, contamination=0.02)
    assert sig.is_anomaly is False
