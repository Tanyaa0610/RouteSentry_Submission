"""Synthetic controlled evaluation: obviously-extreme labeled anomalies
should be detected at a high rate without flagging most real normal
holdout rows. This is a wiring sanity check, not a real-accuracy claim
(see submission/ml_limitations.md)."""
from pathlib import Path

from ml.features import build_feature_table
from ml.synthetic import run_synthetic_evaluation
from routesentry.data_io import load_shipments

ROOT = Path(__file__).resolve().parent.parent
SHIPMENTS = ROOT / "shipment_records.csv"


def _eligible_rows():
    shipments = load_shipments(SHIPMENTS)
    rows = build_feature_table(shipments)
    return [r for r in rows if r.eligible]


def test_synthetic_detection_rate_is_high_for_both_models():
    eligible = _eligible_rows()
    result = run_synthetic_evaluation(eligible, n_anomalies=20, n_normal_holdout=100, magnitude=80.0, seed=42)
    assert result.if_detection_rate >= 0.8
    assert result.lof_detection_rate >= 0.8


def test_synthetic_false_alert_rate_is_low_on_real_normal_holdout():
    eligible = _eligible_rows()
    result = run_synthetic_evaluation(eligible, n_anomalies=20, n_normal_holdout=100, magnitude=80.0, seed=42)
    # at contamination=0.05 we expect roughly 5% false alerts on genuinely
    # normal held-out data, not a blanket majority
    assert result.if_false_alert_rate < 0.25
    assert result.lof_false_alert_rate < 0.25


def test_synthetic_mean_score_is_higher_for_injected_anomalies():
    eligible = _eligible_rows()
    result = run_synthetic_evaluation(eligible, n_anomalies=20, n_normal_holdout=100, magnitude=80.0, seed=42)
    assert result.if_mean_score_synthetic > result.if_mean_score_normal
    assert result.lof_mean_score_synthetic > result.lof_mean_score_normal


def test_synthetic_evaluation_is_deterministic():
    eligible = _eligible_rows()
    r1 = run_synthetic_evaluation(eligible, n_anomalies=20, n_normal_holdout=100, magnitude=80.0, seed=42)
    r2 = run_synthetic_evaluation(eligible, n_anomalies=20, n_normal_holdout=100, magnitude=80.0, seed=42)
    assert r1 == r2


def test_synthetic_raises_on_too_small_a_pool():
    tiny_pool = _eligible_rows()[:5]
    import pytest
    with pytest.raises(ValueError):
        run_synthetic_evaluation(tiny_pool, n_anomalies=5, n_normal_holdout=100)
