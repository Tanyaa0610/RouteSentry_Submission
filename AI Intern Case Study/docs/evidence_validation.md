# Evidence Retrieval and Validation — Full Detail

This is the detailed version of README §5: how a candidate note becomes (or doesn't become) a cited
justification.

## Retrieval (`routesentry/retrieval.py`)

A from-scratch TF-IDF cosine-similarity index over the 10 notes, plus a deterministic eligibility
prefilter — every note whose `applies_to` names the route being evaluated or `"All Routes"`. At 10
notes, this prefilter is exhaustive: the system never relies on semantic similarity alone to decide
whether a note *could* apply. Similarity only ranks candidates and helps choose what's worth
mentioning in an unexplained row's `reason` (the closest near-miss, named and explained, rather than
a silent blank).

Why not a vector database? With 10 notes, exact brute-force TF-IDF cosine similarity over every note
is already exhaustive and exact — a vector database would add infrastructure with no recall benefit
at this scale.

## Validation (`routesentry/evidence.py`)

This is the only place that decides whether a candidate note counts as support. Every candidate is
checked, in order:

1. **Excluded?** Does the note's own text explicitly say the affected routes aren't part of this
   dataset (e.g. N004)? If so, reject regardless of its `applies_to` field.
2. **Route match?** Does `applies_to` actually name this route, or `"All Routes"`?
3. **Date match?** Does the note's applicable window overlap the query week? An explicit "from X to
   Y" range in the text is used when present; otherwise the note is scoped to its own ISO week only
   (see below).
4. **Direction?** Does the note's text describe a cost-*increasing* event, as opposed to "not
   significantly affected," "improved road conditions," "absorbed without a rate change," and
   similar phrasing that denies or reverses an effect?

A note must pass all four to ever become a `"No (justified)"` verdict. `reason` strings are built
only from the fields of the note that was actually accepted (or actually closest-and-rejected) —
there's no free-text generation step that could invent a cause, a date, or a note id.

## Handling unsupported and contradictory evidence

- **No eligible note at all** → `"Yes"` (unexplained). The pipeline never guesses; `reason` states
  that explicitly, optionally naming the closest *route-eligible* candidate and exactly why it
  didn't qualify (wrong window, wrong direction), so a reviewer can audit the near-miss instead of
  just seeing a blank.
- **A note explicitly contradicts a rise** (e.g. "costs were not significantly affected") →
  classified `"contradicts"`, never accepted, and still recorded in the evidence ledger even though
  it isn't the cited reason.
- **Multiple candidates, only one genuinely supports** → the genuinely supporting one wins; a
  textually-similar-but-wrong-route or wrong-window note is excluded by the deterministic checks,
  not by similarity score.
- **Multiple genuinely supporting candidates** (not observed in the real data, but handled) →
  deterministic tie-break: highest similarity, then lowest `note_id` — documented so the outcome is
  predictable without running the code.
- **A note with no explicit date range** (e.g. N003's nationwide diesel rise, "starting this week")
  is deliberately **not** treated as applying indefinitely — it's scoped to its own ISO week only.
  This trades recall for avoiding a confidently-wrong justification months later, matching the
  brief's own stated preference ("a confident wrong reason scores worse than couldn't find one").

## How this was checked before trusting it

1. **Unit tests for the math, not just the output.** `tests/test_weekly.py` hand-computes expected
   `cost_per_tonne_km` and baseline values for small synthetic fixtures and asserts against them,
   rather than only checking that the real run "looks plausible."
2. **The brief's own 3 examples reproduce exactly under the rule-based pipeline** (`run.py --mode
   rule_based`) on every numeric field, the verdict, and the cited note id
   (`tests/test_golden_examples.py`) — confirmed only *after* the threshold and evidence rules were
   designed independently (see `docs/threshold_analysis.md`). Under the ML-driven default, 2 of the
   3 still reproduce exactly; the third diverges on verdict only, by design — see
   `docs/model_selection.md` §3 and `tests/test_ml_golden_divergence.py`.
3. **14 adversarial hallucination cases**, one per failure mode the brief names explicitly (wrong
   route, wrong window, "not significantly affected", an excluded global note, a later event that
   can't explain an earlier rise, etc.) — all pass; see `submission/adversarial_evaluation.md`.
4. **Manually re-read** the 2 real notes (N001, N002) behind the 3 real `"No (justified)"` rows
   against their cited route and week, by hand, per the brief's own spot-check promise.
5. **Reproducibility actually run**, not assumed: 3 full CLI invocations in 3 separate processes,
   SHA-256-compared (`docs/reproducibility_report.md`).
