"""Deterministic evidence validation.

This is the ONLY place that decides whether a note counts as supporting
evidence. Retrieval (retrieval.py) finds candidates; this module is the
guardrail layer that prevents a textually-similar-but-wrong note from
ever becoming a "justified" verdict. Every rule here is directly traceable
to a bullet in the brief's "Hallucination" guardrail section.

Direction classification is a small, hand-auditable keyword rule set --
not a learned classifier -- precisely because with only 10 notes a
transparent rule set is more trustworthy (and 100% reproducible) than a
model that might generalise unpredictably. See README for the trade-off
discussion.
"""
from __future__ import annotations

import re
from datetime import date
from typing import List, Tuple

from routesentry.dates import ranges_overlap
from routesentry.models import CandidateEvidence, Note

# Checked in order: EXCLUDED > NEGATIVE (no_impact/improved) > POSITIVE
# (cost_increase). Order matters: a note can mention "cost" while also
# explicitly saying costs were NOT affected -- negation must win.
_EXCLUDED_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"not part of this dataset", re.I), "not part of this dataset"),
]

_NEGATIVE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"not significantly affected", re.I), "not significantly affected"),
    (re.compile(r"no significant disruptions?", re.I), "no significant disruptions"),
    (re.compile(r"no major disruptions?", re.I), "no major disruptions"),
    (re.compile(r"remained (stable|normal)", re.I), "remained {0}"),
    (re.compile(r"without a rate change", re.I), "without a rate change"),
    (re.compile(r"returned to normal", re.I), "returned to normal"),
    (re.compile(r"improved road conditions", re.I), "improved road conditions"),
    (re.compile(r"absorbed", re.I), "costs absorbed without a rate change"),
]

_POSITIVE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"surcharge", re.I), "surcharge"),
    (re.compile(r"higher (trip )?costs?", re.I), "higher trip costs"),
    (re.compile(r"pushing up", re.I), "pushing up transportation costs"),
    (re.compile(r"prices? rose|rose nationwide", re.I), "prices rose"),
    (re.compile(r"disrupt", re.I), "disrupted normal operations"),
    (re.compile(r"detours", re.I), "longer detours"),
]


def classify_direction(text: str) -> Tuple[str, str]:
    """Returns (label, matched_phrase). label is one of:
    "excluded" | "no_impact" | "improved" | "cost_increase" | "unknown"
    """
    for pattern, phrase in _EXCLUDED_PATTERNS:
        if pattern.search(text):
            return "excluded", phrase

    for pattern, phrase in _NEGATIVE_PATTERNS:
        m = pattern.search(text)
        if m:
            label = "improved" if "improved" in phrase or "returned to normal" in phrase else "no_impact"
            return label, phrase.format(*m.groups()) if "{0}" in phrase else phrase

    for pattern, phrase in _POSITIVE_PATTERNS:
        if pattern.search(text):
            return "cost_increase", phrase

    return "unknown", ""


def evaluate_candidate(
    note: Note,
    similarity: float,
    retrieval_source: str,
    route: str,
    week_start: date,
    week_end: date,
) -> CandidateEvidence:
    direction, phrase = classify_direction(note.text)

    applies_field_match = note.applies_to == route or note.applies_to == "All Routes"
    route_match = applies_field_match and direction != "excluded"
    date_match = ranges_overlap(note.window_start, note.window_end, week_start, week_end)

    if route_match and date_match and direction == "cost_increase":
        relation = "supports"
        rejection_reason = ""
    else:
        rejection_reason = _explain_rejection(
            note, direction, phrase, applies_field_match, route_match, date_match, route, week_start, week_end
        )
        relation = (
            "contradicts"
            if applies_field_match and date_match and direction in ("no_impact", "improved")
            else "irrelevant"
        )

    return CandidateEvidence(
        note_id=note.note_id,
        note_date=note.date,
        note_text=note.text,
        similarity=similarity,
        retrieval_source=retrieval_source,
        route_match=route_match,
        date_match=date_match,
        direction=direction,
        direction_phrase=phrase,
        relation=relation,
        rejection_reason=rejection_reason,
    )


def _explain_rejection(
    note: Note,
    direction: str,
    phrase: str,
    applies_field_match: bool,
    route_match: bool,
    date_match: bool,
    route: str,
    week_start: date,
    week_end: date,
) -> str:
    if direction == "excluded":
        return (
            f"note {note.note_id} explicitly states its routes are 'not part of this "
            f"dataset' -- it does not apply despite applies_to='{note.applies_to}'"
        )
    if not applies_field_match:
        return f"note {note.note_id} applies to '{note.applies_to}', not '{route}'"
    if not date_match:
        return (
            f"note {note.note_id}'s applicable window ({note.window_start}..{note.window_end}, "
            f"{'explicit range' if note.window_source == 'explicit' else 'defaulted to its own week'}) "
            f"does not cover the week of {week_start} ({week_start}..{week_end})"
        )
    if direction in ("no_impact", "improved"):
        return f"note {note.note_id} describes '{phrase}' -- not a cost-increasing event"
    return f"note {note.note_id}'s content could not be confirmed as a cost-increasing event"
