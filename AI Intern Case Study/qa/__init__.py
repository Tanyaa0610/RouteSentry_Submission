"""Read-only natural-language Q&A layer over RouteSentry's findings.

This package never computes a verdict, never fits a model, never accepts
or rejects a note, and never writes to any file the pipelines produce. It
only reads the in-memory `LedgerEntry` list that `routesentry.pipeline.run`
or `ml.official_pipeline.run` already produced, and answers questions
about it with deterministic keyword/entity matching -- no external LLM, no
network call, no API key.

See qa/engine.py for the query layer, qa/parser.py for question parsing,
and qa/answer.py for the function that ties them together. `ask.py` at the
project root is the CLI entrypoint.
"""
