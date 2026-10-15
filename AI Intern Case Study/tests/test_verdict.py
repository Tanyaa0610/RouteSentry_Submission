from datetime import date

from routesentry.evidence import evaluate_candidate
from routesentry.models import StatisticalSignal, WeeklyMetric
from routesentry.verdict import decide


def _metric(vh=20.0, vp=10.0, n_prior=8, n_peer=5):
    return WeeklyMetric(
        route="A-B", route_type="Short", week_of=date(2024, 6, 10),
        cost_per_tonne_km=1.5, n_prior_weeks=n_prior, vs_own_history_pct=vh,
        n_peer_routes=n_peer, vs_similar_routes_pct=vp,
    )


def test_not_anomalous_yields_plain_no():
    m = _metric(vh=1.0, vp=1.0)
    sig = StatisticalSignal(is_anomaly=False, rule_triggered="not rising")
    entry = decide(m, sig, [])
    assert entry.final_verdict == "No"
    assert entry.accepted_note_id is None


def test_anomalous_with_supporting_note_is_justified(note_factory):
    m = _metric()
    sig = StatisticalSignal(is_anomaly=True, rule_triggered="test")
    note = note_factory(
        note_id="N200", note_date=date(2024, 6, 10), applies_to="A-B",
        text="A surcharge pushed up transportation costs this week.",
    )
    cand = evaluate_candidate(note, 0.8, "semantic", "A-B", date(2024, 6, 10), date(2024, 6, 16))
    entry = decide(m, sig, [cand])
    assert entry.final_verdict == "No (justified)"
    assert entry.accepted_note_id == "N200"
    assert "N200" in entry.reason


def test_anomalous_without_supporting_note_is_unexplained(note_factory):
    m = _metric()
    sig = StatisticalSignal(is_anomaly=True, rule_triggered="test")
    note = note_factory(
        note_id="N201", note_date=date(2024, 6, 10), applies_to="Other-Route",
        text="Irrelevant note.",
    )
    cand = evaluate_candidate(note, 0.2, "semantic", "A-B", date(2024, 6, 10), date(2024, 6, 16))
    entry = decide(m, sig, [cand])
    assert entry.final_verdict == "Yes"
    assert entry.accepted_note_id is None


def test_conflicting_evidence_prefers_higher_similarity_supporting_note(note_factory):
    m = _metric()
    sig = StatisticalSignal(is_anomaly=True, rule_triggered="test")
    weak = note_factory(
        note_id="N300", note_date=date(2024, 6, 10), applies_to="A-B",
        text="A surcharge pushed up transportation costs this week.",
    )
    strong = note_factory(
        note_id="N301", note_date=date(2024, 6, 10), applies_to="A-B",
        text="A surcharge pushed up transportation costs this week.",
    )
    c_weak = evaluate_candidate(weak, 0.3, "semantic", "A-B", date(2024, 6, 10), date(2024, 6, 16))
    c_strong = evaluate_candidate(strong, 0.9, "semantic", "A-B", date(2024, 6, 10), date(2024, 6, 16))
    entry = decide(m, sig, [c_weak, c_strong])
    assert entry.accepted_note_id == "N301"


def test_conflicting_evidence_tie_break_is_lowest_note_id(note_factory):
    m = _metric()
    sig = StatisticalSignal(is_anomaly=True, rule_triggered="test")
    a = note_factory(note_id="N400", note_date=date(2024, 6, 10), applies_to="A-B",
                      text="A surcharge pushed up transportation costs this week.")
    b = note_factory(note_id="N399", note_date=date(2024, 6, 10), applies_to="A-B",
                      text="A surcharge pushed up transportation costs this week.")
    c_a = evaluate_candidate(a, 0.5, "semantic", "A-B", date(2024, 6, 10), date(2024, 6, 16))
    c_b = evaluate_candidate(b, 0.5, "semantic", "A-B", date(2024, 6, 10), date(2024, 6, 16))
    entry = decide(m, sig, [c_a, c_b])
    assert entry.accepted_note_id == "N399"  # equal similarity -> lowest note_id wins deterministically


def test_supporting_and_contradicting_notes_together_still_accepts_only_supporting(note_factory):
    m = _metric()
    sig = StatisticalSignal(is_anomaly=True, rule_triggered="test")
    good = note_factory(note_id="N500", note_date=date(2024, 6, 10), applies_to="A-B",
                         text="A surcharge pushed up transportation costs this week.")
    bad = note_factory(note_id="N501", note_date=date(2024, 6, 10), applies_to="A-B",
                        text="Costs were not significantly affected this week.")
    c_good = evaluate_candidate(good, 0.6, "semantic", "A-B", date(2024, 6, 10), date(2024, 6, 16))
    c_bad = evaluate_candidate(bad, 0.9, "semantic", "A-B", date(2024, 6, 10), date(2024, 6, 16))
    entry = decide(m, sig, [c_good, c_bad])
    assert entry.final_verdict == "No (justified)"
    assert entry.accepted_note_id == "N500"


def test_every_justified_verdict_has_a_note_id_and_vice_versa(note_factory):
    m = _metric()
    sig = StatisticalSignal(is_anomaly=True, rule_triggered="test")
    entry_unexplained = decide(m, sig, [])
    assert entry_unexplained.final_verdict == "Yes"
    assert not entry_unexplained.accepted_note_id

    note = note_factory(note_id="N600", note_date=date(2024, 6, 10), applies_to="A-B",
                         text="A surcharge pushed up transportation costs this week.")
    cand = evaluate_candidate(note, 0.6, "semantic", "A-B", date(2024, 6, 10), date(2024, 6, 16))
    entry_justified = decide(m, sig, [cand])
    assert entry_justified.final_verdict == "No (justified)"
    assert entry_justified.accepted_note_id
