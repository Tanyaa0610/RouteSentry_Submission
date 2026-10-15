"""Evidence validation -- the hallucination-guardrail layer. Each test here
maps to a bullet in the brief's "Hallucination" section or an adversarial
case in docs/adversarial_evaluation.md."""
from datetime import date

from routesentry.evidence import classify_direction, evaluate_candidate


def test_classify_direction_positive_surcharge():
    label, phrase = classify_direction("A festival surcharge was applied due to high demand.")
    assert label == "cost_increase"


def test_classify_direction_negative_not_significantly_affected():
    label, _ = classify_direction("Costs were not significantly affected by the maintenance work.")
    assert label == "no_impact"


def test_classify_direction_improved_road_conditions():
    label, _ = classify_direction("Local authorities announced improved road conditions after resurfacing.")
    assert label == "improved"


def test_classify_direction_excluded_not_part_of_dataset():
    label, _ = classify_direction("The affected routes are not part of this dataset.")
    assert label == "excluded"


def test_classify_direction_unknown_when_no_keyword_matches():
    label, _ = classify_direction("The weather was pleasant this week.")
    assert label == "unknown"


# --- evaluate_candidate: route / date / direction combined -----------------


def test_valid_route_and_window_supports(note_factory):
    note = note_factory(
        note_id="N100", note_date=date(2024, 6, 10), applies_to="Mumbai-Pune",
        text="A toll surcharge pushed up transportation costs from Jun 10 to Jun 16.",
    )
    cand = evaluate_candidate(note, 0.5, "semantic", "Mumbai-Pune", date(2024, 6, 10), date(2024, 6, 16))
    assert cand.relation == "supports"


def test_wrong_event_window_is_rejected(note_factory):
    note = note_factory(
        note_id="N101", note_date=date(2024, 6, 10), applies_to="Mumbai-Pune",
        text="A toll surcharge pushed up transportation costs from Jun 10 to Jun 16.",
    )
    # Query week is a month later -- no overlap.
    cand = evaluate_candidate(note, 0.5, "semantic", "Mumbai-Pune", date(2024, 7, 8), date(2024, 7, 14))
    assert cand.relation != "supports"
    assert cand.date_match is False


def test_correct_date_wrong_route_is_rejected(note_factory):
    note = note_factory(
        note_id="N102", note_date=date(2024, 6, 10), applies_to="Delhi-Jaipur",
        text="A toll surcharge pushed up transportation costs this week.",
    )
    cand = evaluate_candidate(note, 0.9, "semantic", "Mumbai-Pune", date(2024, 6, 10), date(2024, 6, 16))
    assert cand.relation != "supports"
    assert cand.route_match is False


def test_global_note_explicitly_excluding_route_is_rejected(note_factory):
    note = note_factory(
        note_id="N103", note_date=date(2024, 3, 11), applies_to="All Routes",
        text="A new toll plaza was commissioned, but the affected routes are not part of this dataset.",
    )
    cand = evaluate_candidate(note, 0.3, "deterministic_eligibility", "Mumbai-Pune", date(2024, 3, 11), date(2024, 3, 17))
    assert cand.relation != "supports"
    assert cand.route_match is False


def test_note_stating_costs_unaffected_is_rejected(note_factory):
    note = note_factory(
        note_id="N104", note_date=date(2024, 7, 29), applies_to="Mumbai-Delhi",
        text="Scheduled maintenance caused minor delays; costs were not significantly affected.",
    )
    cand = evaluate_candidate(note, 0.4, "semantic", "Mumbai-Delhi", date(2024, 7, 29), date(2024, 8, 4))
    assert cand.relation == "contradicts"


def test_improved_road_conditions_does_not_justify_rise(note_factory):
    note = note_factory(
        note_id="N105", note_date=date(2024, 5, 20), applies_to="Delhi-Jaipur",
        text="Local authorities announced improved road conditions after resurfacing work.",
    )
    cand = evaluate_candidate(note, 0.6, "semantic", "Delhi-Jaipur", date(2024, 5, 20), date(2024, 5, 26))
    assert cand.relation != "supports"


def test_expired_disruption_outside_window_is_rejected(note_factory):
    note = note_factory(
        note_id="N106", note_date=date(2025, 2, 24), applies_to="Chennai-Bangalore",
        text="Heavy flooding disrupted truck movement from Feb 24 to Mar 8. Roads were passable again by Mar 9.",
    )
    # A week well after the flood cleared.
    cand = evaluate_candidate(note, 0.5, "semantic", "Chennai-Bangalore", date(2025, 4, 1), date(2025, 4, 7))
    assert cand.relation != "supports"
    assert cand.date_match is False


def test_no_supporting_note_among_candidates_means_no_accepted(note_factory):
    from routesentry.verdict import _select_supporting_candidate

    note = note_factory(
        note_id="N107", note_date=date(2024, 1, 1), applies_to="Other-Route",
        text="Irrelevant note about a different corridor.",
    )
    cand = evaluate_candidate(note, 0.1, "semantic", "Mumbai-Pune", date(2024, 1, 1), date(2024, 1, 7))
    assert _select_supporting_candidate([cand]) is None


def test_week_boundary_event_starting_monday_is_in_window(note_factory):
    # Event explicitly starts on the Monday of the query week -- must count as overlapping.
    note = note_factory(
        note_id="N108", note_date=date(2024, 6, 10), applies_to="Mumbai-Pune",
        text="A surcharge raised costs from Jun 10 to Jun 10.",
    )
    cand = evaluate_candidate(note, 0.5, "semantic", "Mumbai-Pune", date(2024, 6, 10), date(2024, 6, 16))
    assert cand.date_match is True


def test_mentions_route_but_no_cost_impact_language_does_not_support(note_factory):
    note = note_factory(
        note_id="N109", note_date=date(2024, 6, 10), applies_to="Mumbai-Pune",
        text="A logistics report mentioned the Mumbai-Pune corridor in passing this week.",
    )
    cand = evaluate_candidate(note, 0.4, "semantic", "Mumbai-Pune", date(2024, 6, 10), date(2024, 6, 16))
    assert cand.relation != "supports"
    assert cand.direction == "unknown"


def test_later_event_cannot_explain_an_earlier_anomaly(note_factory):
    # Note describes an event dated well after the anomalous week; its
    # window (lacking an explicit range) defaults to its own week, so it
    # must not overlap an earlier week.
    note = note_factory(
        note_id="N110", note_date=date(2024, 8, 1), applies_to="Mumbai-Pune",
        text="A new surcharge was introduced this week due to fuel costs.",
    )
    cand = evaluate_candidate(note, 0.6, "semantic", "Mumbai-Pune", date(2024, 6, 10), date(2024, 6, 16))
    assert cand.date_match is False
    assert cand.relation != "supports"
