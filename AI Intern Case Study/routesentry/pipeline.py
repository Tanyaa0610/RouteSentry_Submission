"""Orchestrates the full run: load -> compute -> retrieve -> validate ->
decide -> write. Clean separation of stages (brief section G): each stage
below is implemented in its own module and this function only wires them
together.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List

from routesentry import config
from routesentry.anomaly import evaluate_signal
from routesentry.data_io import load_notes, load_shipments
from routesentry.dates import sunday_of
from routesentry.evidence import evaluate_candidate
from routesentry.models import LedgerEntry, Note
from routesentry.retrieval import TfidfIndex, build_query
from routesentry.verdict import decide
from routesentry.weekly import build_weekly_metrics

logger = logging.getLogger("routesentry.pipeline")


def run(
    shipments_path: str | Path,
    notes_path: str | Path,
    threshold: float = config.RISE_THRESHOLD_PCT,
) -> List[LedgerEntry]:
    shipments = load_shipments(shipments_path)
    notes = load_notes(notes_path)
    index = TfidfIndex(notes)

    metrics = build_weekly_metrics(shipments)

    ledger: List[LedgerEntry] = []
    n_llm_calls = 0  # tracked for the cost log: this pipeline makes zero LLM calls
    for metric in metrics:
        signal = evaluate_signal(metric, threshold)
        candidates = _gather_candidates(metric.route, metric.week_of, notes, index)
        ledger.append(decide(metric, signal, candidates))

    n_flagged = sum(1 for e in ledger if e.final_verdict == "Yes")
    n_justified = sum(1 for e in ledger if e.final_verdict == "No (justified)")
    n_normal = sum(1 for e in ledger if e.final_verdict == "No")
    logger.info(
        "pipeline complete: %d route-weeks -- %d unexplained (Yes), %d justified, %d normal; "
        "%d LLM calls made",
        len(ledger), n_flagged, n_justified, n_normal, n_llm_calls,
    )
    return ledger


def _gather_candidates(route: str, week_of, notes: List[Note], index: TfidfIndex):
    week_start, week_end = week_of, sunday_of(week_of)

    # Deterministic eligibility prefilter: every note whose applies_to field
    # names this route or "All Routes". At this corpus size (10 notes) this
    # is cheap and exhaustive, guaranteeing we never miss a truly-applicable
    # note due to a weak semantic match -- semantic retrieval is a ranking
    # and explanation aid here, not a recall gate (see retrieval.py).
    deterministic_ids = {
        n.note_id for n in notes if n.applies_to == route or n.applies_to == "All Routes"
    }

    query_text = build_query(route, week_of)
    semantic_hits = index.query(query_text, config.SEMANTIC_TOP_K)
    semantic_ids = {n.note_id for n, _sim in semantic_hits}
    similarity_by_id = {n.note_id: sim for n, sim in semantic_hits}

    candidate_ids = deterministic_ids | semantic_ids
    candidates = []
    for note in notes:
        if note.note_id not in candidate_ids:
            continue
        sim = similarity_by_id.get(note.note_id, 0.0)
        if note.note_id in deterministic_ids and note.note_id in semantic_ids:
            source = "both"
        elif note.note_id in deterministic_ids:
            source = "deterministic_eligibility"
        else:
            source = "semantic"
        candidates.append(evaluate_candidate(note, sim, source, route, week_start, week_end))

    candidates.sort(key=lambda c: (-c.similarity, c.note_id))
    return candidates


# --- Output writers ---------------------------------------------------------

def write_output_csv(ledger: List[LedgerEntry], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(config.OUTPUT_COLUMNS)
        for e in ledger:
            writer.writerow(
                [
                    e.route,
                    e.week_of.isoformat(),
                    f"{e.cost_per_tonne_km:.2f}",
                    _format_pct(e.vs_own_history_pct, "vs this route's past average", e.n_prior_weeks),
                    _format_pct(e.vs_similar_routes_pct, "vs similar-length routes this week", None),
                    e.final_verdict,
                    e.accepted_note_id or "",
                    e.reason,
                ]
            )
    logger.info("wrote %d output rows to %s", len(ledger), path)


def _format_pct(value, suffix: str, n_prior) -> str:
    if value is None:
        if n_prior is not None:
            return f"insufficient history ({n_prior} prior week(s) available)"
        return f"no peer routes reported this week"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}% {suffix}"


def write_evidence_ledger_csv(ledger: List[LedgerEntry], path: str | Path) -> None:
    """Full inspectable audit trail: one row per (route-week, candidate
    note) pair, so any output row can be traced to exactly which notes
    were considered and why each was accepted or rejected."""
    path = Path(path)
    fieldnames = [
        "route", "route_type", "week_of", "cost_per_tonne_km",
        "n_prior_weeks", "vs_own_history_pct", "n_peer_routes", "vs_similar_routes_pct",
        "statistical_signal", "signal_rule", "final_verdict", "accepted_note_id", "reason",
        "insufficiency_note",
        "candidate_note_id", "candidate_note_date", "candidate_similarity", "candidate_source",
        "candidate_route_match", "candidate_date_match", "candidate_direction",
        "candidate_direction_phrase", "candidate_relation", "candidate_rejection_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for e in ledger:
            base = {
                "route": e.route,
                "route_type": e.route_type,
                "week_of": e.week_of.isoformat(),
                "cost_per_tonne_km": f"{e.cost_per_tonne_km:.4f}",
                "n_prior_weeks": e.n_prior_weeks,
                "vs_own_history_pct": "" if e.vs_own_history_pct is None else f"{e.vs_own_history_pct:.2f}",
                "n_peer_routes": e.n_peer_routes,
                "vs_similar_routes_pct": "" if e.vs_similar_routes_pct is None else f"{e.vs_similar_routes_pct:.2f}",
                "statistical_signal": e.statistical_signal,
                "signal_rule": e.signal_rule,
                "final_verdict": e.final_verdict,
                "accepted_note_id": e.accepted_note_id or "",
                "reason": e.reason,
                "insufficiency_note": e.insufficiency_note,
            }
            if not e.candidates:
                writer.writerow({**base, **{k: "" for k in fieldnames if k.startswith("candidate_")}})
                continue
            for c in e.candidates:
                writer.writerow(
                    {
                        **base,
                        "candidate_note_id": c.note_id,
                        "candidate_note_date": c.note_date.isoformat(),
                        "candidate_similarity": f"{c.similarity:.4f}",
                        "candidate_source": c.retrieval_source,
                        "candidate_route_match": c.route_match,
                        "candidate_date_match": c.date_match,
                        "candidate_direction": c.direction,
                        "candidate_direction_phrase": c.direction_phrase,
                        "candidate_relation": c.relation,
                        "candidate_rejection_reason": c.rejection_reason,
                    }
                )
    logger.info("wrote evidence ledger (%d route-weeks) to %s", len(ledger), path)
