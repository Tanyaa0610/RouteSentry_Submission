"""Score-direction convention, LOF neighbour-count heuristic for small
windows, and basic model-fitting sanity."""
import numpy as np

from ml.models import choose_lof_k, fit_isolation_forest, fit_lof, score_isolation_forest, score_lof


def _synthetic_normal_and_outlier(n_normal=200, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, 3))
    outlier = np.array([[50.0, 50.0, 50.0]])
    return normal, outlier


def test_isolation_forest_scores_outlier_higher_than_normal():
    normal, outlier = _synthetic_normal_and_outlier()
    model = fit_isolation_forest(normal, random_state=42)
    normal_scores, _ = score_isolation_forest(model, normal)
    outlier_scores, outlier_preds = score_isolation_forest(model, outlier)
    assert outlier_scores[0] > normal_scores.mean()
    assert outlier_preds[0] is np.bool_(True) or bool(outlier_preds[0]) is True


def test_lof_scores_outlier_higher_than_normal():
    normal, outlier = _synthetic_normal_and_outlier()
    model = fit_lof(normal)
    normal_scores, _ = score_lof(model, normal)
    outlier_scores, outlier_preds = score_lof(model, outlier)
    assert outlier_scores[0] > normal_scores.mean()
    assert bool(outlier_preds[0]) is True


def test_choose_lof_k_respects_bounds_and_train_size():
    assert choose_lof_k(1000) == 20  # capped at LOF_K_MAX
    assert choose_lof_k(6) <= 5  # choose_lof_k(6) must be < train_size
    assert choose_lof_k(6) < 6
    assert choose_lof_k(50) == 5  # 50 // 10 == 5 == LOF_K_MIN, at the floor


def test_choose_lof_k_never_equals_or_exceeds_train_size():
    for n in range(2, 30):
        assert choose_lof_k(n) < n


def test_fit_lof_on_tiny_window_does_not_crash():
    rng = np.random.default_rng(1)
    tiny_train = rng.normal(size=(3, 2))
    model = fit_lof(tiny_train)
    scores, preds = score_lof(model, rng.normal(size=(2, 2)))
    assert len(scores) == 2


def test_scores_are_deterministic_given_fixed_seed():
    normal, outlier = _synthetic_normal_and_outlier()
    m1 = fit_isolation_forest(normal, random_state=7)
    m2 = fit_isolation_forest(normal, random_state=7)
    s1, _ = score_isolation_forest(m1, outlier)
    s2, _ = score_isolation_forest(m2, outlier)
    assert s1[0] == s2[0]
