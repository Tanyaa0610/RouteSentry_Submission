"""Ties qa/parser.py's question parsing to qa/engine.py's queries and
formats a grounded, readable answer. Every sentence in every answer is
built from a real field on a `LedgerEntry` or `CandidateEvidence` object
already produced by the pipeline -- there is no free-text generation step
here that could invent a route, a date, a note id, or a verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from routesentry.models import LedgerEntry

from qa.engine import (
    QAContext,
    evidence_for,
    find_route,
    find_route_week,
    is_unassessed,
    justified_entries,
    rank_by_metric,
    summarize,
    unassessed_entries,
    unexplained_entries,
)
from qa.parser import ParsedQuestion, parse_question

EXAMPLE_QUESTIONS = [
    "Why was Mumbai-Pune flagged on 2025-09-15?",
    "Which routes have the highest cost per tonne-km?",
    "Show me unexplained anomalies.",
    "Which notes were accepted or rejected, and why?",
    "Compare Ahmedabad-Mumbai's cost with its historical baseline.",
    "Which findings have no supporting evidence?",
    "Summarize the most important findings for the current run.",
]


@dataclass
class Answer:
    text: str
    intent: str
    grounded: bool  # False only for "unsupported" / no-match cases
    entries: List[LedgerEntry] = field(default_factory=list)


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "unavailable"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _fmt_entry_line(e: LedgerEntry) -> str:
    return (
        f"{e.route}, week of {e.week_of.isoformat()}: cost_per_tonne_km={e.cost_per_tonne_km:.2f}, "
        f"vs_own_history={_fmt_pct(e.vs_own_history_pct)}, vs_similar_routes={_fmt_pct(e.vs_similar_routes_pct)}, "
        f"verdict={e.final_verdict}"
        + (f" (note {e.accepted_note_id})" if e.accepted_note_id else "")
    )


def _fmt_entry_detail(e: LedgerEntry, ctx: QAContext) -> str:
    lines = [
        f"Route: {e.route} ({e.route_type})",
        f"Week: {e.week_of.isoformat()} (Monday-start)",
        f"Observed data: cost_per_tonne_km={e.cost_per_tonne_km:.2f}, "
        f"vs_own_history={_fmt_pct(e.vs_own_history_pct)} ({e.n_prior_weeks} prior week(s) used), "
        f"vs_similar_routes={_fmt_pct(e.vs_similar_routes_pct)} ({e.n_peer_routes} peer route(s) that week)",
        f"Model-generated anomaly signal ({ctx.mode} pipeline): "
        f"{'statistical anomaly' if e.statistical_signal else 'not a statistical anomaly'} -- {e.signal_rule}",
        f"Final verdict: {e.final_verdict}",
    ]
    if e.accepted_note_id:
        accepted = next((c for c in e.candidates if c.note_id == e.accepted_note_id), None)
        lines.append(f"Evidence-based justification: note {e.accepted_note_id}" + (f" -- \"{accepted.note_text}\"" if accepted else ""))
    elif e.final_verdict == "Yes":
        lines.append("Evidence-based justification: none found (checked, not fabricated) -- " + (e.insufficiency_note or e.reason))
    rejected = [c for c in e.candidates if c.note_id != e.accepted_note_id]
    if rejected:
        lines.append(f"Other candidate notes considered and rejected: {len(rejected)}")
        for c in rejected[:3]:
            lines.append(f"  - {c.note_id}: {c.rejection_reason or ('relation=' + c.relation)}")
    lines.append(f"Inference/explanation (plain English): {e.reason}")
    return "\n".join(lines)


def _no_entry_for_route_week(route: str, week_of) -> Answer:
    return Answer(
        text=(
            f"No route-week record exists for {route} in the week of {week_of.isoformat()} "
            "in the current run's data -- there is no shipment activity for that route that week, "
            "so there is nothing to report rather than a guess."
        ),
        intent="route_week",
        grounded=False,
    )


def answer_question(question: str, ctx: QAContext) -> Answer:
    parsed = parse_question(question, ctx.routes)

    if parsed.intent == "summary":
        s = summarize(ctx)
        text = (
            f"Run summary ({s.mode}-driven pipeline, generated {s.generated_at}):\n"
            f"- {s.n_route_weeks} route-weeks across {s.n_routes} routes\n"
            f"- {s.n_yes} unexplained (Yes)\n"
            f"- {s.n_justified} justified (No (justified)), citing note(s): {', '.join(s.notes_cited) or 'none'}\n"
            f"- {s.n_normal} not anomalous (No)\n"
            f"- {s.n_unassessed} not statistically assessed (insufficient history/training data)"
        )
        return Answer(text=text, intent="summary", grounded=True, entries=ctx.ledger)

    if parsed.intent == "unexplained":
        entries = unexplained_entries(ctx)
        if not entries:
            return Answer(text="No unexplained (\"Yes\") findings in this run.", intent="unexplained", grounded=True)
        text = f"{len(entries)} unexplained finding(s) -- flagged as anomalous with no validated supporting note:\n" + \
            "\n".join(f"- {_fmt_entry_line(e)}" for e in entries)
        return Answer(text=text, intent="unexplained", grounded=True, entries=entries)

    if parsed.intent == "unassessed":
        entries = unassessed_entries(ctx)
        if not entries:
            return Answer(text="Every route-week in this run was statistically assessed; none were skipped for insufficient history.", intent="unassessed", grounded=True)
        text = f"{len(entries)} route-week(s) were never statistically assessed (insufficient history or pooled training data), not \"found normal\":\n" + \
            "\n".join(f"- {e.route}, week of {e.week_of.isoformat()}: {e.signal_rule}" for e in entries[:15])
        if len(entries) > 15:
            text += f"\n  ... and {len(entries) - 15} more."
        return Answer(text=text, intent="unassessed", grounded=True, entries=entries)

    if parsed.intent == "evidence_breakdown":
        breakdown = evidence_for(ctx, parsed.route)
        if not breakdown:
            scope = f" for {parsed.route}" if parsed.route else ""
            return Answer(text=f"No candidate notes were retrieved{scope} in this run.", intent="evidence_breakdown", grounded=True)
        lines = []
        for key, b in list(breakdown.items())[:10]:
            lines.append(f"{key}:")
            for c in b.accepted:
                lines.append(f"  accepted: {c.note_id} -- \"{c.note_text}\"")
            for c in b.rejected:
                lines.append(f"  rejected: {c.note_id} -- {c.rejection_reason or c.relation}")
        if len(breakdown) > 10:
            lines.append(f"... and {len(breakdown) - 10} more route-week(s) with candidate notes.")
        return Answer(text="\n".join(lines), intent="evidence_breakdown", grounded=True)

    if parsed.intent == "ranking":
        entries = rank_by_metric(ctx, parsed.metric, n=parsed.n, descending=parsed.descending)
        direction = "highest" if parsed.descending else "lowest"
        metric_label = parsed.metric
        text = f"Top {len(entries)} route-weeks by {direction} {metric_label}:\n" + \
            "\n".join(f"- {_fmt_entry_line(e)}" for e in entries)
        return Answer(text=text, intent="ranking", grounded=True, entries=entries)

    if parsed.intent == "historical_comparison":
        if not parsed.route:
            return Answer(
                text="I can compare a route's current cost with its historical baseline, but I need a route name "
                     "(e.g. \"Mumbai-Pune\") to do it.",
                intent="historical_comparison", grounded=False,
            )
        route_entries = find_route(ctx, parsed.route)
        if not route_entries:
            return Answer(text=f"No data found for route '{parsed.route}'.", intent="historical_comparison", grounded=False)
        target = find_route_week(ctx, parsed.route, parsed.week_of) if parsed.week_of else route_entries[-1]
        if target is None:
            return _no_entry_for_route_week(parsed.route, parsed.week_of)
        text = _fmt_entry_detail(target, ctx)
        return Answer(text=text, intent="historical_comparison", grounded=True, entries=[target])

    if parsed.intent == "route_week":
        entry = find_route_week(ctx, parsed.route, parsed.week_of)
        if entry is None:
            return _no_entry_for_route_week(parsed.route, parsed.week_of)
        return Answer(text=_fmt_entry_detail(entry, ctx), intent="route_week", grounded=True, entries=[entry])

    if parsed.intent == "route_only":
        entries = find_route(ctx, parsed.route)
        if not entries:
            return Answer(text=f"No data found for route '{parsed.route}'.", intent="route_only", grounded=False)
        latest = entries[-1]
        n_yes = sum(1 for e in entries if e.final_verdict == "Yes")
        n_justified = sum(1 for e in entries if e.final_verdict == "No (justified)")
        text = (
            f"{parsed.route}: {len(entries)} route-week(s) on record, {n_yes} unexplained, {n_justified} justified.\n"
            f"Most recent week on record:\n{_fmt_entry_detail(latest, ctx)}"
        )
        return Answer(text=text, intent="route_only", grounded=True, entries=entries)

    examples = "\n".join(f"  - {q}" for q in EXAMPLE_QUESTIONS)
    text = (
        "I couldn't match that to a route, date, or supported question type in the current data. "
        f"I can answer questions like:\n{examples}\n"
        f"Known routes in this run: {', '.join(ctx.routes)}"
    )
    return Answer(text=text, intent="unsupported", grounded=False)
