# Token / Cost Report — one full run over `shipment_records.csv`

| Metric | Value |
|---|---|
| LLM provider | none |
| LLM model | none |
| Total input tokens | 0 |
| Total output tokens | 0 |
| Number of LLM calls | 0 |
| LLM API cost | ₹0 / $0.00 |

## Why zero

Neither pipeline — the default ML-driven one or the preserved `--mode rule_based` one — calls an
LLM, an embeddings API, or any network service at any stage. The statistical-signal step differs
between them (a trained, locally-run scikit-learn LOF model for the default; a flat threshold rule
for `rule_based`), but "trained ML model" and "LLM" are different things: LOF has no tokens, no
prompt, and no API — it is a classical nearest-neighbor density estimator fit on numeric features,
running entirely on this machine. Note retrieval (both pipelines), evidence validation, and
verdict/reason generation are implemented with the Python standard library plus a from-scratch
TF-IDF cosine-similarity index — no embeddings API, no vector database, no LLM call of any kind, at
any stage. This is a deliberate choice, not a shortcut forced by the brief:

- With only 10 context notes, a from-scratch TF-IDF index is exhaustive
  and exact (every note is scored against every query) — a vector
  database or an LLM-based retriever would add infrastructure without
  improving recall.
- Every `reason` string in `submission/output.csv` is template-generated
  from the specific `Note` object that was deterministically accepted or
  from the specific rejected candidate being cited — there is no free-text
  generation step that could hallucinate a cause, a date, or a note ID.
  This directly serves the brief's hallucination guardrail: a template
  cannot invent content that is not already a field on the `Note` or
  `CandidateEvidence` dataclass it is built from.

This is an honest measurement of an LLM-free design, not an estimate: the
figure is 0 because neither `routesentry/` nor `ml/` imports an LLM client
or makes any network call (grep-verifiable — `ml/`'s only third-party
imports are `numpy` and `sklearn`, both purely local; see
`routesentry/pipeline.py`, which logs `n_llm_calls = 0` once per run as a
self-check).

## What this does NOT cover

This is the LLM **API** cost only, per the brief's "rough cost at your
model's published rate" framing. It does not include:

- Engineering/development time.
- The compute cost of running this process itself: sub-second for the rule-based pipeline
  (`docs/reproducibility_report.md`), a few seconds for the ML-driven default — it fits one LOF model
  per chronological fold on every run (`docs/ml_driven_reproducibility_report.md`). Still local,
  still free either way.

## If an LLM were added later

The natural extension point (see README "Future improvements") is an
optional LLM pass purely for *prose smoothing* of an already-accepted
`reason` string — never for deciding the verdict, the note ID, or the
numeric fields, all of which must remain deterministic per the brief's
guardrails. If that is ever implemented, this report must be regenerated
from the provider's actual reported usage for a real run — not estimated —
before it is included in a submission.
