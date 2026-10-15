#!/usr/bin/env python3
"""RouteSentry Q&A -- ask plain-English questions about the current run's
findings. Entirely local: no external LLM, no API key, no network call --
every answer is read from the in-memory results of the same pipeline
`run.py` uses (ml-driven by default, or --mode rule_based), via
deterministic keyword/entity matching. It never changes a verdict, a
threshold, or any file the pipelines produce -- see qa/ and
tests/test_qa_guardrails.py.

Usage:
    python3 ask.py                      # interactive session, ML-driven findings (default)
    python3 ask.py --mode rule_based    # interactive session, rule-based findings
    python3 ask.py --ask "question"     # answer one question and exit (scripting/testing)

Example questions:
    Why was Mumbai-Pune flagged on 2025-09-15?
    Which routes have the highest cost per tonne-km?
    Show me unexplained anomalies.
    Which notes were accepted or rejected, and why?
    Compare Ahmedabad-Mumbai's cost with its historical baseline.
    Which findings have no supporting evidence?
    Summarize the most important findings for the current run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qa.answer import EXAMPLE_QUESTIONS, answer_question
from qa.engine import build_context

DEFAULT_DIR = Path(__file__).resolve().parent
EXIT_WORDS = {"exit", "quit", "q", ":q"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask plain-English questions about RouteSentry's findings")
    parser.add_argument("--shipments", default=str(DEFAULT_DIR / "shipment_records.csv"))
    parser.add_argument("--notes", default=str(DEFAULT_DIR / "context_notes.csv"))
    parser.add_argument("--mode", choices=["ml", "rule_based"], default="ml",
                         help="which pipeline's findings to explore (default: ml, matching run.py's default)")
    parser.add_argument("--ask", metavar="QUESTION", help="answer one question non-interactively and exit")
    args = parser.parse_args()

    print(f"Loading {args.mode}-driven findings from {args.shipments} / {args.notes} ...")
    try:
        ctx = build_context(args.shipments, args.notes, mode=args.mode)
    except Exception as exc:
        print(f"Could not build findings: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(ctx.ledger)} route-weeks across {len(ctx.routes)} routes "
          f"({args.mode}-driven pipeline, generated {ctx.generated_at}).")

    if args.ask:
        print(answer_question(args.ask, ctx).text)
        return 0

    print()
    print("RouteSentry Q&A -- ask a question, or type 'exit' to quit.")
    print("Example questions:")
    for q in EXAMPLE_QUESTIONS:
        print(f"  - {q}")
    print()

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            break
        print(answer_question(question, ctx).text)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
