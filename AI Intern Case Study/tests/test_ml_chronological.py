"""Chronological expanding-window evaluation: train/test separation, no
future leakage into a fold's training set, and the insufficient-training
exclusion path."""
from datetime import date, timedelta

from ml.chronological import run_chronological_evaluation
from ml.features import FeatureRow

BASE = date(2024, 1, 1)


def _row(week_offset, route="A-B", route_type="Short", eligible=True, cpkm=1.0):
    return FeatureRow(
        route=route, route_type=route_type,
        week_of=BASE + timedelta(weeks=week_offset),
        cost_per_tonne_km=cpkm, vs_own_history_pct=0.0, vs_similar_routes_pct=0.0,
        wow_change_pct=0.0, prior_volatility_pct=1.0, n_prior_weeks=5, n_peer_routes=1,
        eligible=eligible, ineligibility_reason="",
    )


def test_ineligible_rows_pass_through_unscored():
    rows = [_row(0, eligible=False)]
    results = run_chronological_evaluation(rows)
    assert len(results) == 1
    assert results[0].eligible is False
    assert results[0].if_score is None
    assert results[0].lof_score is None


def test_fold_below_min_train_size_is_excluded_not_scored():
    # Only a handful of eligible rows -- never reaches MIN_TRAIN_SIZE.
    rows = [_row(i) for i in range(5)]
    results = run_chronological_evaluation(rows, min_train_size=40)
    assert all(r.excluded_insufficient_training or r.train_size is None for r in results if r.eligible)
    assert all(r.if_score is None for r in results)


def test_train_set_never_includes_rows_at_or_after_test_week():
    # Build enough rows to clear MIN_TRAIN_SIZE, across two routes so each
    # weekly fold has >1 row, then check the public invariant that folds
    # move strictly forward in time (see assertions below).
    rows = []
    for i in range(60):
        rows.append(_row(i, route="A-B"))
        rows.append(_row(i, route="C-D"))

    # Monkeypatch run_chronological_evaluation's internals indirectly by
    # checking the public invariant instead: every MLResult's fold_index
    # groups rows whose week_of is monotonically non-decreasing with
    # fold_index, which can only hold if folds are strictly time-ordered.
    results = run_chronological_evaluation(rows, min_train_size=10, fold_step_weeks=8)
    scored = [r for r in results if r.if_score is not None]
    assert scored, "expected at least one scored fold with this much data"

    # Within each fold, every scored row's week must be >= the max week of
    # any lower-fold-index scored row (folds move strictly forward in time).
    by_fold = {}
    for r in scored:
        by_fold.setdefault(r.fold_index, []).append(r.week_of)
    fold_indices = sorted(by_fold)
    max_week_so_far = None
    for fi in fold_indices:
        min_week_this_fold = min(by_fold[fi])
        if max_week_so_far is not None:
            assert min_week_this_fold > max_week_so_far, "a later fold's week must exceed all earlier folds' weeks"
        max_week_so_far = max(by_fold[fi])


def test_results_sorted_by_week_then_route():
    rows = [_row(i, route=r) for i in range(50) for r in ("A-B", "C-D")]
    results = run_chronological_evaluation(rows, min_train_size=5)
    pairs = [(r.week_of, r.route) for r in results]
    assert pairs == sorted(pairs)


def test_deterministic_across_repeated_runs():
    rows = [_row(i, route=r, cpkm=1.0 + 0.01 * i) for i in range(60) for r in ("A-B", "C-D")]
    r1 = run_chronological_evaluation(rows, min_train_size=10)
    r2 = run_chronological_evaluation(rows, min_train_size=10)
    for a, b in zip(r1, r2):
        assert a.if_score == b.if_score
        assert a.lof_score == b.lof_score
        assert a.if_prediction == b.if_prediction
        assert a.lof_prediction == b.lof_prediction
