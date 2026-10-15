"""qa/engine.py query functions, against a small hand-built QAContext (no
full pipeline run needed -- these test the query logic in isolation)."""
from datetime import date

import pytest

from qa.engine import (
    QAContext,
    evidence_for,
    find_route,
    find_route_week,
    is_unassessed,
    justified_entries,
    normal_entries,
    rank_by_metric,
    summarize,
    unassessed_entries,
    unexplained_entries,
)
from routesentry.models import CandidateEvidence, LedgerEntry


def _entry(route, week_of, cpkm, vh, vp, verdict, signal_rule="", accepted_note_id=None, candidates=None):
    return LedgerEntry(
        route=route, route_type="Short", week_of=week_of, cost_per_tonne_km=cpkm,
        n_prior_weeks=8, vs_own_history_pct=vh, n_peer_routes=1, vs_similar_routes_pct=vp,
        statistical_signal=verdict != "No", signal_rule=signal_rule, candidates=candidates or [],
        accepted_note_id=accepted_note_id, final_verdict=verdict, reason="test reason",
    )


def _candidate(note_id, relation="irrelevant", rejection_reason="", note_text="test note"):
    return CandidateEvidence(
        note_id=note_id, note_date=date(2024, 1, 1), note_text=note_text, similarity=0.5,
        retrieval_source="semantic", route_match=True, date_match=True, direction="cost_increase",
        direction_phrase="", relation=relation, rejection_reason=rejection_reason,
    )


@pytest.fixture
def ctx():
    entries = [
        _entry("A-B", date(2024, 1, 1), 1.0, 5.0, 2.0, "No", signal_rule="rising but within normal range"),
        _entry("A-B", date(2024, 1, 8), 2.0, 30.0, 25.0, "Yes", signal_rule="exceeds threshold"),
        _entry("C-D", date(2024, 1, 8), 3.0, 20.0, 15.0, "No (justified)",
               signal_rule="exceeds threshold", accepted_note_id="N001",
               candidates=[_candidate("N001", relation="supports"), _candidate("N002", relation="irrelevant", rejection_reason="wrong route")]),
        _entry("C-D", date(2024, 1, 15), None, None, None, "No", signal_rule="not evaluated: no prior weeks available"),
    ]
    return QAContext(mode="ml", ledger=entries, routes=["A-B", "C-D"], shipments_path=None, notes_path=None, generated_at="2024-01-01T00:00:00")


def test_find_route_returns_sorted_by_week(ctx):
    entries = find_route(ctx, "A-B")
    assert [e.week_of for e in entries] == [date(2024, 1, 1), date(2024, 1, 8)]


def test_find_route_unknown_returns_empty(ctx):
    assert find_route(ctx, "X-Y") == []


def test_find_route_week_exact_match(ctx):
    e = find_route_week(ctx, "A-B", date(2024, 1, 8))
    assert e is not None and e.final_verdict == "Yes"


def test_find_route_week_no_match_returns_none(ctx):
    assert find_route_week(ctx, "A-B", date(2024, 2, 1)) is None


def test_rank_by_metric_descending(ctx):
    top = rank_by_metric(ctx, "cost_per_tonne_km", n=2, descending=True)
    assert [round(e.cost_per_tonne_km, 1) for e in top] == [3.0, 2.0]


def test_rank_by_metric_ascending(ctx):
    bottom = rank_by_metric(ctx, "cost_per_tonne_km", n=2, descending=False)
    assert [round(e.cost_per_tonne_km, 1) for e in bottom] == [1.0, 2.0]


def test_rank_by_metric_excludes_none_values(ctx):
    # the 4th entry has cost_per_tonne_km=None and must never appear
    ranked = rank_by_metric(ctx, "cost_per_tonne_km", n=10)
    assert None not in [e.cost_per_tonne_km for e in ranked]
    assert len(ranked) == 3


def test_rank_by_metric_rejects_unknown_metric(ctx):
    with pytest.raises(ValueError):
        rank_by_metric(ctx, "not_a_real_metric")


def test_unexplained_entries(ctx):
    entries = unexplained_entries(ctx)
    assert len(entries) == 1
    assert entries[0].route == "A-B" and entries[0].week_of == date(2024, 1, 8)


def test_justified_entries(ctx):
    entries = justified_entries(ctx)
    assert len(entries) == 1
    assert entries[0].accepted_note_id == "N001"


def test_is_unassessed_true_for_not_evaluated_rule():
    e = _entry("A-B", date(2024, 1, 1), None, None, None, "No", signal_rule="not evaluated: no prior weeks")
    assert is_unassessed(e) is True


def test_is_unassessed_false_for_normal_rule():
    e = _entry("A-B", date(2024, 1, 1), 1.0, 1.0, 1.0, "No", signal_rule="rising but within normal range")
    assert is_unassessed(e) is False


def test_unassessed_entries(ctx):
    entries = unassessed_entries(ctx)
    assert len(entries) == 1
    assert entries[0].route == "C-D" and entries[0].week_of == date(2024, 1, 15)


def test_normal_entries_excludes_unassessed(ctx):
    entries = normal_entries(ctx)
    assert len(entries) == 1
    assert entries[0].route == "A-B" and entries[0].week_of == date(2024, 1, 1)
    # the unassessed C-D row must NOT be counted as "normal"
    assert all(e.week_of != date(2024, 1, 15) for e in entries)


def test_evidence_for_specific_route_includes_all_candidates(ctx):
    breakdown = evidence_for(ctx, "C-D")
    key = "C-D 2024-01-08"
    assert key in breakdown
    assert len(breakdown[key].accepted) == 1
    assert breakdown[key].accepted[0].note_id == "N001"
    assert len(breakdown[key].rejected) == 1
    assert breakdown[key].rejected[0].note_id == "N002"


def test_evidence_for_no_route_filter_only_shows_decisive_rows(ctx):
    breakdown = evidence_for(ctx)
    # A-B's "No" row and C-D's "not evaluated" row have no candidates and/or
    # aren't Yes/No(justified) -- only C-D's justified row should appear
    assert list(breakdown.keys()) == ["C-D 2024-01-08"]


def test_summarize_counts(ctx):
    s = summarize(ctx)
    assert s.n_route_weeks == 4
    assert s.n_yes == 1
    assert s.n_justified == 1
    assert s.n_normal == 1
    assert s.n_unassessed == 1
    assert s.notes_cited == ["N001"]


def test_summarize_is_deterministic(ctx):
    s1 = summarize(ctx)
    s2 = summarize(ctx)
    assert s1 == s2
