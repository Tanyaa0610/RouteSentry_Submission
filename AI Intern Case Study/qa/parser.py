"""Deterministic question parsing: route and date extraction, plus intent
classification by keyword. No LLM, no external API, no network call --
this is plain string matching and a small regex set, chosen so the same
question always parses to the same result.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

from routesentry.dates import monday_of

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# "15 September 2025", "15th September 2025"
_DATE_DMY = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b", re.IGNORECASE
)
# "September 15, 2025" / "September 15 2025"
_DATE_MDY = re.compile(
    r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.IGNORECASE
)
# "2025-09-15"
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def parse_date(text: str) -> Optional[date]:
    m = _DATE_ISO.search(text)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        try:
            return date(y, mo, d)
        except ValueError:
            pass

    m = _DATE_DMY.search(text)
    if m:
        d_str, mon_str, y_str = m.groups()
        month = _MONTHS.get(mon_str[:3].lower())
        if month:
            try:
                return date(int(y_str), month, int(d_str))
            except ValueError:
                pass

    m = _DATE_MDY.search(text)
    if m:
        mon_str, d_str, y_str = m.groups()
        month = _MONTHS.get(mon_str[:3].lower())
        if month:
            try:
                return date(int(y_str), month, int(d_str))
            except ValueError:
                pass

    return None


def find_route(text: str, known_routes: List[str]) -> Optional[str]:
    """Matches a route like `Mumbai-Pune` against free text that might say
    "Mumbai-Pune", "Mumbai to Pune", "Mumbai and Pune", "Mumbai–Pune" (en
    dash), case-insensitively, in either city order. Returns the route only
    if exactly one known route's both cities are present -- an ambiguous or
    no-match question returns None rather than guessing."""
    normalized = re.sub(r"[-–—]", " ", text.lower())
    words = set(re.findall(r"[a-z]+", normalized))

    matches = []
    for route in known_routes:
        origin, _, destination = route.partition("-")
        if origin.lower() in words and destination.lower() in words:
            matches.append(route)

    return matches[0] if len(matches) == 1 else None


@dataclass
class ParsedQuestion:
    raw: str
    intent: str  # see INTENTS below
    route: Optional[str] = None
    target_date: Optional[date] = None
    week_of: Optional[date] = None
    metric: Optional[str] = None
    n: int = 5
    descending: bool = True


INTENTS = (
    "summary",
    "unexplained",
    "unassessed",
    "evidence_breakdown",
    "ranking",
    "historical_comparison",
    "route_week",
    "route_only",
    "unsupported",
)

_METRIC_KEYWORDS = {
    "cost_per_tonne_km": ["cost per tonne", "cost/tonne", "cost-per-tonne", "tonne-km", "tonne km"],
    "vs_own_history_pct": ["own history", "historical baseline", "vs own", "history"],
    "vs_similar_routes_pct": ["similar routes", "peer", "similar-length"],
}


def _detect_metric(q: str) -> str:
    for metric, keywords in _METRIC_KEYWORDS.items():
        if any(k in q for k in keywords):
            return metric
    return "cost_per_tonne_km"


def parse_question(question: str, known_routes: List[str]) -> ParsedQuestion:
    q = question.strip()
    ql = q.lower()

    route = find_route(q, known_routes)
    target_date = parse_date(q)
    week_of = monday_of(target_date) if target_date else None

    if any(k in ql for k in ("summarize", "summary", "overview", "most important findings")):
        return ParsedQuestion(raw=q, intent="summary")

    if "unassessed" in ql or "insufficient history" in ql or "not enough history" in ql or "no supporting evidence" in ql or "no evidence" in ql:
        # "no supporting evidence" / "no evidence" is ambiguous between
        # "unexplained" (Yes, evidence was checked and found nothing) and
        # "unassessed" (never statistically evaluated at all) -- we check
        # for explicit "unassessed"/"insufficient history" phrasing first,
        # and otherwise treat "no evidence" as asking about unexplained
        # (evidence-checked-and-absent) findings, which is what the brief's
        # own vocabulary ("unexplained") means.
        if "unassessed" in ql or "insufficient history" in ql or "not enough history" in ql:
            return ParsedQuestion(raw=q, intent="unassessed")
        return ParsedQuestion(raw=q, intent="unexplained")

    if "unexplained" in ql or ("anomal" in ql and "no" not in ql.split()):
        return ParsedQuestion(raw=q, intent="unexplained")

    if ("accepted" in ql or "rejected" in ql) and "note" in ql:
        return ParsedQuestion(raw=q, intent="evidence_breakdown", route=route)

    if any(k in ql for k in ("highest", "lowest", "top", "most expensive", "cheapest", "ranking", "rank")):
        descending = not any(k in ql for k in ("lowest", "cheapest", "least"))
        n_match = re.search(r"\btop\s+(\d+)\b", ql)
        n = int(n_match.group(1)) if n_match else 5
        return ParsedQuestion(raw=q, intent="ranking", metric=_detect_metric(ql), n=n, descending=descending)

    if "compare" in ql and ("histor" in ql or "baseline" in ql):
        return ParsedQuestion(raw=q, intent="historical_comparison", route=route, target_date=target_date, week_of=week_of)

    if route and (target_date or week_of):
        return ParsedQuestion(raw=q, intent="route_week", route=route, target_date=target_date, week_of=week_of)

    if route:
        return ParsedQuestion(raw=q, intent="route_only", route=route)

    return ParsedQuestion(raw=q, intent="unsupported")
