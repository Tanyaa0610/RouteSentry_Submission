"""Deterministic question parsing: route/date extraction and intent
classification. No pipeline run needed -- these test qa/parser.py in
isolation against a fixed list of known routes."""
from datetime import date

from qa.parser import find_route, parse_date, parse_question

ROUTES = [
    "Mumbai-Pune", "Delhi-Jaipur", "Ahmedabad-Mumbai",
    "Chennai-Bangalore", "Kolkata-Bhubaneswar", "Mumbai-Delhi", "Delhi-Chennai",
]


def test_find_route_hyphenated():
    assert find_route("Why was Mumbai-Pune flagged?", ROUTES) == "Mumbai-Pune"


def test_find_route_natural_language():
    assert find_route("What about shipments from Mumbai to Pune?", ROUTES) == "Mumbai-Pune"


def test_find_route_en_dash():
    assert find_route("Mumbai–Pune costs", ROUTES) == "Mumbai-Pune"


def test_find_route_unknown_returns_none():
    assert find_route("Why was Nagpur-Surat flagged?", ROUTES) is None


def test_find_route_partial_city_mention_returns_none():
    # "Mumbai" alone matches 3 different routes' origin -- ambiguous, must not guess.
    assert find_route("How is Mumbai doing?", ROUTES) is None


def test_parse_date_iso():
    assert parse_date("on 2025-09-15 please") == date(2025, 9, 15)


def test_parse_date_day_month_year():
    assert parse_date("flagged on 15 September 2025") == date(2025, 9, 15)


def test_parse_date_month_day_year():
    assert parse_date("flagged on September 15, 2025") == date(2025, 9, 15)


def test_parse_date_none_when_absent():
    assert parse_date("Why was Mumbai-Pune flagged?") is None


def test_parse_question_route_and_date_is_route_week():
    q = parse_question("Why was Mumbai-Pune flagged on 2025-09-15?", ROUTES)
    assert q.intent == "route_week"
    assert q.route == "Mumbai-Pune"
    assert q.week_of == date(2025, 9, 15)  # Monday already


def test_parse_question_route_only():
    q = parse_question("Tell me about Delhi-Jaipur", ROUTES)
    assert q.intent == "route_only"
    assert q.route == "Delhi-Jaipur"


def test_parse_question_ranking_highest():
    q = parse_question("Which routes have the highest cost per tonne-km?", ROUTES)
    assert q.intent == "ranking"
    assert q.descending is True
    assert q.metric == "cost_per_tonne_km"


def test_parse_question_ranking_lowest():
    q = parse_question("Which routes have the lowest cost?", ROUTES)
    assert q.intent == "ranking"
    assert q.descending is False


def test_parse_question_unexplained():
    q = parse_question("Show me unexplained anomalies.", ROUTES)
    assert q.intent == "unexplained"


def test_parse_question_no_supporting_evidence_maps_to_unexplained():
    q = parse_question("Which findings have no supporting evidence?", ROUTES)
    assert q.intent == "unexplained"


def test_parse_question_unassessed_explicit():
    q = parse_question("Which route-weeks were unassessed due to insufficient history?", ROUTES)
    assert q.intent == "unassessed"


def test_parse_question_evidence_breakdown():
    q = parse_question("Which notes were accepted or rejected, and why?", ROUTES)
    assert q.intent == "evidence_breakdown"


def test_parse_question_historical_comparison():
    q = parse_question("Compare this route's cost with its historical baseline.", ROUTES)
    assert q.intent == "historical_comparison"


def test_parse_question_summary():
    q = parse_question("Summarize the most important findings for the current run.", ROUTES)
    assert q.intent == "summary"


def test_parse_question_unsupported_for_unrelated_text():
    q = parse_question("What's the weather like today?", ROUTES)
    assert q.intent == "unsupported"


def test_parse_question_is_deterministic():
    q1 = parse_question("Why was Mumbai-Pune flagged on 2025-09-15?", ROUTES)
    q2 = parse_question("Why was Mumbai-Pune flagged on 2025-09-15?", ROUTES)
    assert q1 == q2
