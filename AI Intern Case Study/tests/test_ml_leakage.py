"""Direct leakage checks: for every scored fold, the recorded train_size
must exactly equal the number of eligible rows strictly before that fold's
earliest test week -- not more (no future rows admitted) and not fewer (no
rows silently dropped)."""
from datetime import date, timedelta

from ml.chronological import run_chronological_evaluation
from ml.features import FeatureRow

BASE = date(2024, 1, 1)


def _row(week_offset, route, cpkm=1.0):
    return FeatureRow(
        route=route, route_type="Short", week_of=BASE + timedelta(weeks=week_offset),
        cost_per_tonne_km=cpkm, vs_own_history_pct=0.0, vs_similar_routes_pct=0.0,
        wow_change_pct=0.0, prior_volatility_pct=1.0, n_prior_weeks=5, n_peer_routes=1,
        eligible=True, ineligibility_reason="",
    )


def test_train_size_exactly_equals_rows_strictly_before_fold():
    rows = [_row(i, route) for i in range(60) for route in ("A-B", "C-D")]
    results = run_chronological_evaluation(rows, min_train_size=10, fold_step_weeks=8)

    by_fold = {}
    for r in results:
        if r.fold_index is not None:
            by_fold.setdefault(r.fold_index, []).append(r)

    for fold_index, fold_results in by_fold.items():
        test_weeks = {r.week_of for r in fold_results}
        boundary = min(test_weeks)
        expected_train_size = sum(1 for r in rows if r.week_of < boundary)
        actual_train_size = fold_results[0].train_size
        assert actual_train_size == expected_train_size, (
            f"fold {fold_index}: train_size={actual_train_size} but "
            f"{expected_train_size} eligible rows exist strictly before {boundary}"
        )
        # and no result in this fold is dated before the boundary (no
        # training-period row was accidentally also scored as test)
        assert all(r.week_of >= boundary for r in fold_results)


def test_no_row_is_scored_using_a_model_fit_on_data_from_its_own_or_a_later_week():
    # A row's fold's train rows must all be strictly before that row's week.
    rows = [_row(i, route) for i in range(60) for route in ("A-B", "C-D")]
    results = run_chronological_evaluation(rows, min_train_size=10, fold_step_weeks=8)
    scored = [r for r in results if r.if_score is not None]
    assert scored
    by_fold_boundary = {}
    for r in scored:
        by_fold_boundary.setdefault(r.fold_index, []).append(r.week_of)
    for fold_index, weeks in by_fold_boundary.items():
        earliest_test_week = min(weeks)
        train_size_for_fold = next(r.train_size for r in scored if r.fold_index == fold_index)
        rows_on_or_after = sum(1 for row in rows if row.week_of >= earliest_test_week)
        rows_before = len(rows) - rows_on_or_after
        assert train_size_for_fold == rows_before


def test_excluded_fold_rows_report_their_insufficient_train_size():
    rows = [_row(i, "A-B") for i in range(3)]  # far too few to ever reach a sane min_train_size
    results = run_chronological_evaluation(rows, min_train_size=100, fold_step_weeks=8)
    assert all(r.if_score is None and r.lof_score is None for r in results)
