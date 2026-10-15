"""ml/calibration.py: the train/validation/test chronological split used
to pick the production contamination value, and the grid-search selection
rule. The core property under test is that the TEST period is never
touched by calibration -- it exists only for later reporting."""
from datetime import date, timedelta

import pytest

from ml.calibration import calibrate_lof_contamination, chronological_three_way_split
from ml.features import FeatureRow

BASE = date(2024, 1, 1)


def _row(week_offset, route="A-B"):
    return FeatureRow(
        route=route, route_type="Short", week_of=BASE + timedelta(weeks=week_offset),
        cost_per_tonne_km=1.0 + 0.001 * week_offset, vs_own_history_pct=1.0, vs_similar_routes_pct=1.0,
        wow_change_pct=0.5, prior_volatility_pct=2.0, n_prior_weeks=5, n_peer_routes=1,
        eligible=True, ineligibility_reason="",
    )


def test_three_way_split_is_chronologically_ordered_and_non_overlapping():
    rows = [_row(i, route=r) for i in range(100) for r in ("A-B", "C-D")]
    train, val, test = chronological_three_way_split(rows)

    assert train and val and test
    max_train_week = max(r.week_of for r in train)
    min_val_week = min(r.week_of for r in val)
    max_val_week = max(r.week_of for r in val)
    min_test_week = min(r.week_of for r in test)

    assert max_train_week < min_val_week
    assert max_val_week < min_test_week


def test_three_way_split_covers_every_row_exactly_once():
    rows = [_row(i, route=r) for i in range(100) for r in ("A-B", "C-D")]
    train, val, test = chronological_three_way_split(rows)
    assert len(train) + len(val) + len(test) == len(rows)
    all_ids = {(r.route, r.week_of) for r in train + val + test}
    assert len(all_ids) == len(rows)


def test_calibration_raises_on_too_little_data():
    rows = [_row(i) for i in range(5)]
    with pytest.raises(ValueError):
        calibrate_lof_contamination(rows)


def test_calibration_selects_a_value_from_the_grid():
    from ml.calibration import CONTAMINATION_GRID
    rows = [_row(i, route=r) for i in range(150) for r in ("A-B", "C-D", "E-F")]
    result = calibrate_lof_contamination(rows)
    assert result.selected_contamination in CONTAMINATION_GRID
    assert len(result.grid) == len(CONTAMINATION_GRID)
    assert result.n_train > 0 and result.n_validation > 0


def test_calibration_grid_scores_are_between_zero_and_one():
    rows = [_row(i, route=r) for i in range(150) for r in ("A-B", "C-D", "E-F")]
    result = calibrate_lof_contamination(rows)
    for candidate in result.grid:
        assert 0.0 <= candidate.synthetic_detection_rate <= 1.0
        assert 0.0 <= candidate.validation_false_alert_rate <= 1.0


def test_calibration_is_deterministic():
    rows = [_row(i, route=r) for i in range(150) for r in ("A-B", "C-D", "E-F")]
    r1 = calibrate_lof_contamination(rows, seed=42)
    r2 = calibrate_lof_contamination(rows, seed=42)
    assert r1.selected_contamination == r2.selected_contamination
    assert [g.synthetic_detection_rate for g in r1.grid] == [g.synthetic_detection_rate for g in r2.grid]
