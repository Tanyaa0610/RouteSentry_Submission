"""Loads a run's findings and answers structured queries against them.

Everything here is read-only: it builds a `QAContext` once (by running the
chosen pipeline exactly as `run.py` would, in-memory -- never by writing or
mutating any file), then every query function below only reads from the
resulting `List[LedgerEntry]`. No function here can change a verdict,
accept/reject a note, or alter a threshold -- see tests/test_qa_guardrails.py
for the tests that prove this.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from routesentry.models import CandidateEvidence, LedgerEntry

VALID_METRICS = {
    "cost_per_tonne_km": "cost per tonne-km",
    "vs_own_history_pct": "vs. own history",
    "vs_similar_routes_pct": "vs. similar routes",
}


@dataclass
class QAContext:
    mode: str  # "ml" or "rule_based" -- which pipeline's findings are loaded
    ledger: List[LedgerEntry]
    routes: List[str]
    shipments_path: Path
    notes_path: Path
    generated_at: str  # ISO timestamp -- when THIS context was built, so an
    # answer can always say which run it's grounded in


def build_context(
    shipments_path: str | Path,
    notes_path: str | Path,
    mode: str = "ml",
) -> QAContext:
    """Runs the chosen pipeline fresh, in-memory, and returns a read-only
    context to answer questions against. Never writes any file."""
    shipments_path = Path(shipments_path)
    notes_path = Path(notes_path)

    if mode == "rule_based":
        from routesentry.pipeline import run as run_pipeline
    elif mode == "ml":
        from ml.official_pipeline import run as run_pipeline
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'ml' or 'rule_based'")

    ledger = run_pipeline(shipments_path, notes_path)
    routes = sorted({e.route for e in ledger})
    return QAContext(
        mode=mode,
        ledger=ledger,
        routes=routes,
        shipments_path=shipments_path,
        notes_path=notes_path,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )


def is_unassessed(entry: LedgerEntry) -> bool:
    """True iff this row's statistical signal was never actually evaluated
    (insufficient history / insufficient pooled training data), as opposed
    to being evaluated and found normal. Both routesentry/anomaly.py and
    ml/signal.py consistently write "not evaluated" into `signal_rule` for
    this case -- this reads that real convention, it does not invent a new
    field."""
    return "not evaluated" in entry.signal_rule


def find_route(ctx: QAContext, route: str) -> List[LedgerEntry]:
    return sorted([e for e in ctx.ledger if e.route == route], key=lambda e: e.week_of)


def find_route_week(ctx: QAContext, route: str, week_of: date) -> Optional[LedgerEntry]:
    return next((e for e in ctx.ledger if e.route == route and e.week_of == week_of), None)


def find_week(ctx: QAContext, week_of: date) -> List[LedgerEntry]:
    return sorted([e for e in ctx.ledger if e.week_of == week_of], key=lambda e: e.route)


def rank_by_metric(
    ctx: QAContext,
    metric: str,
    n: int = 5,
    descending: bool = True,
    only_assessed: bool = False,
) -> List[LedgerEntry]:
    if metric not in VALID_METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(VALID_METRICS)}")

    def value(e: LedgerEntry) -> Optional[float]:
        return getattr(e, metric)

    candidates = [e for e in ctx.ledger if value(e) is not None]
    if only_assessed:
        candidates = [e for e in candidates if not is_unassessed(e)]
    candidates.sort(key=lambda e: value(e), reverse=descending)
    return candidates[:n]


def unexplained_entries(ctx: QAContext) -> List[LedgerEntry]:
    return sorted([e for e in ctx.ledger if e.final_verdict == "Yes"], key=lambda e: (e.week_of, e.route))


def justified_entries(ctx: QAContext) -> List[LedgerEntry]:
    return sorted([e for e in ctx.ledger if e.final_verdict == "No (justified)"], key=lambda e: (e.week_of, e.route))


def unassessed_entries(ctx: QAContext) -> List[LedgerEntry]:
    return sorted([e for e in ctx.ledger if is_unassessed(e)], key=lambda e: (e.week_of, e.route))


def normal_entries(ctx: QAContext) -> List[LedgerEntry]:
    return sorted(
        [e for e in ctx.ledger if e.final_verdict == "No" and not is_unassessed(e)],
        key=lambda e: (e.week_of, e.route),
    )


@dataclass
class EvidenceBreakdown:
    accepted: List[CandidateEvidence]
    rejected: List[CandidateEvidence]


def evidence_for(ctx: QAContext, route: Optional[str] = None) -> Dict[str, EvidenceBreakdown]:
    """Per-route-week accepted vs. rejected candidate notes, straight from
    the ledger entries' own `candidates` lists -- nothing here is re-derived
    or guessed; every CandidateEvidence already carries its own
    relation/rejection_reason from routesentry/evidence.py.

    When a specific route is given, every route-week with any retrieved
    candidate is included (the user asked about that route specifically).
    With no route filter, only route-weeks where evidence was actually
    decisive -- a statistical anomaly that got justified or stayed
    unexplained -- are included, since routine weeks each still retrieve a
    handful of semantically-similar-but-irrelevant candidates via the
    top-K TF-IDF search and would otherwise swamp a general-purpose answer
    with noise that was never close to being accepted."""
    if route:
        entries = find_route(ctx, route)
    else:
        entries = [e for e in ctx.ledger if e.final_verdict in ("Yes", "No (justified)")]

    result: Dict[str, EvidenceBreakdown] = {}
    for e in entries:
        if not e.candidates:
            continue
        accepted = [c for c in e.candidates if c.relation == "supports" and c.note_id == e.accepted_note_id]
        rejected = [c for c in e.candidates if c.note_id != e.accepted_note_id]
        key = f"{e.route} {e.week_of.isoformat()}"
        result[key] = EvidenceBreakdown(accepted=accepted, rejected=rejected)
    return result


@dataclass
class RunSummary:
    mode: str
    n_route_weeks: int
    n_routes: int
    n_yes: int
    n_justified: int
    n_normal: int
    n_unassessed: int
    notes_cited: List[str]
    generated_at: str


def summarize(ctx: QAContext) -> RunSummary:
    yes = unexplained_entries(ctx)
    justified = justified_entries(ctx)
    unassessed = unassessed_entries(ctx)
    normal = normal_entries(ctx)
    notes_cited = sorted({e.accepted_note_id for e in justified if e.accepted_note_id})
    return RunSummary(
        mode=ctx.mode,
        n_route_weeks=len(ctx.ledger),
        n_routes=len(ctx.routes),
        n_yes=len(yes),
        n_justified=len(justified),
        n_normal=len(normal),
        n_unassessed=len(unassessed),
        notes_cited=notes_cited,
        generated_at=ctx.generated_at,
    )
