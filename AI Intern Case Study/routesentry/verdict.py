"""Combines the statistical signal and the evidence candidates into the
final submission verdict ("Yes" / "No" / "No (justified)").

This is the single source of truth for the verdict -- the CSV writer and
the audit report both read LedgerEntry.final_verdict, there is no second
implementation of this logic anywhere else in the codebase.

Three distinct concepts, kept apart on purpose (brief section E):
  1. statistical signal  (anomaly.py)      -- is the number unusual?
  2. evidence assessment (evidence.py)     -- does a note genuinely explain it?
  3. final verdict       (this module)     -- the exact value the output contract needs.
A justified rise is NOT "flagged=Yes"; an unflagged week is NOT "justified".
"""
from __future__ import annotations

from typing import List, Optional

from routesentry.config import MIN_SIMILARITY_TO_MENTION
from routesentry.dates import sunday_of
from routesentry.models import CandidateEvidence, LedgerEntry, StatisticalSignal, WeeklyMetric


def decide(
    metric: WeeklyMetric,
    signal: StatisticalSignal,
    candidates: List[CandidateEvidence],
) -> LedgerEntry:
    entry = LedgerEntry(
        route=metric.route,
        route_type=metric.route_type,
        week_of=metric.week_of,
        cost_per_tonne_km=metric.cost_per_tonne_km,
        n_prior_weeks=metric.n_prior_weeks,
        vs_own_history_pct=metric.vs_own_history_pct,
        n_peer_routes=metric.n_peer_routes,
        vs_similar_routes_pct=metric.vs_similar_routes_pct,
        statistical_signal=signal.is_anomaly,
        signal_rule=signal.rule_triggered,
        candidates=candidates,
    )

    if not signal.is_anomaly:
        entry.final_verdict = "No"
        entry.accepted_note_id = None
        entry.reason = _not_anomalous_reason(metric)
        return entry

    accepted = _select_supporting_candidate(candidates)
    if accepted is not None:
        entry.final_verdict = "No (justified)"
        entry.accepted_note_id = accepted.note_id
        entry.reason = _justified_reason(accepted)
        return entry

    entry.final_verdict = "Yes"
    entry.accepted_note_id = None
    entry.insufficiency_note = (
        "No note both applies to this route and covers this week with a cost-increasing "
        "event; treated as unexplained rather than guessed, per the no-hallucination guardrail."
    )
    entry.reason = _unexplained_reason(metric, candidates)
    return entry


def _select_supporting_candidate(candidates: List[CandidateEvidence]) -> Optional[CandidateEvidence]:
    supporting = [c for c in candidates if c.relation == "supports"]
    if not supporting:
        return None
    # Deterministic tie-break: highest semantic similarity first, then
    # lowest note_id lexicographically. Documented so a reader can predict
    # the outcome of any future conflicting-evidence case without running it.
    supporting.sort(key=lambda c: (-c.similarity, c.note_id))
    return supporting[0]


def _not_anomalous_reason(metric: WeeklyMetric) -> str:
    if metric.vs_own_history_pct is None:
        return (
            f"Not enough history to assess this route yet ({metric.n_prior_weeks} prior week(s) "
            "available); not flagged."
        )
    return "Cost is within the normal range for this route this week; no anomaly detected."


def _justified_reason(accepted: CandidateEvidence) -> str:
    content = accepted.note_text.strip().rstrip(".")
    return (
        f"Matches note {accepted.note_id} dated {accepted.note_date.isoformat()}: {content}. "
        "The cost rise has a clear explanation."
    )


def _unexplained_reason(metric: WeeklyMetric, candidates: List[CandidateEvidence]) -> str:
    closest = _closest_mentionable_candidate(candidates)
    if closest is None:
        return "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."

    clause = _mention_clause(closest, metric)
    return (
        f"The closest note ({closest.note_id}, {closest.note_date.isoformat()}) {clause}. "
        "No genuine justification found; flagged for review."
    )


def _closest_mentionable_candidate(candidates: List[CandidateEvidence]) -> Optional[CandidateEvidence]:
    """Pick the candidate worth naming in an unexplained-verdict's reason.

    Prefers a note that is at least route-eligible (applies_to matches, and
    not textually excluded) even if it is ultimately rejected for its
    content or timing -- that is what a human reviewer would actually want
    to see next. A note that merely shares vocabulary with a *different*
    route (e.g. two notes that both happen to mention "Mumbai") is a worse
    thing to cite even if its raw text similarity is higher; citing it
    would make a correct verdict read as confused. Falls back to the
    highest-similarity candidate only when nothing route-eligible exists.
    """
    route_eligible = [c for c in candidates if c.route_match]
    if route_eligible:
        # Deterministically scoped to this route (or "All Routes"): mention
        # it even if its raw text similarity is low -- eligibility here
        # comes from the structured applies_to field, not from text match.
        route_eligible.sort(key=lambda c: (-c.similarity, c.note_id))
        return route_eligible[0]

    eligible = [c for c in candidates if c.similarity >= MIN_SIMILARITY_TO_MENTION]
    if not eligible:
        return None
    eligible.sort(key=lambda c: (-c.similarity, c.note_id))
    return eligible[0]


def _mention_clause(candidate: CandidateEvidence, metric: WeeklyMetric) -> str:
    if candidate.direction in ("no_impact", "improved"):
        return f"mentions {candidate.direction_phrase} -- it does not describe a reason for a cost rise on this route"
    if not candidate.route_match:
        return "applies to a different route or is explicitly out of scope for this one"
    if not candidate.date_match:
        week_end = sunday_of(metric.week_of)
        return f"does not cover the week of {metric.week_of.isoformat()} ({metric.week_of.isoformat()}..{week_end.isoformat()})"
    return "does not describe a cost-increasing event relevant here"
