# Likely Interview Questions

**Why trailing 8 weeks of *available* data rather than 8 calendar weeks?**
The brief says "trailing 8-week rolling average... using only weeks
strictly before the current week... if fewer than 8 prior weeks exist, use
all prior weeks available." We read "weeks" as weeks the route actually
has an observation for (since a route might legitimately skip a week with
no shipments), not calendar weeks with implied zero-filling — zero-filling
would distort the average with fabricated zero-cost weeks, which is worse.

**Why exclude the current week from the baseline?**
Including it would mean the baseline partly measures itself — a spike would
inflate its own comparison average and understate how anomalous it is. The
brief is also explicit on this point ("no look-ahead").

**How does the peer baseline work, concretely?**
For route R in week W: take every other route with the same `route_type`
that also shipped in week W, average their `cost_per_tonne_km`, and compare
R's cost to that average. R never appears in its own peer average. If no
peer shipped that week, the comparison is reported as unavailable, not
computed from an empty set.

**Why isn't high retrieval similarity treated as proof?**
Because text similarity measures *topical overlap*, not *causal relevance*.
A note can mention the right route and similar vocabulary while explicitly
stating costs were **not** affected, or describing an *improved* condition,
or applying to a *different, specifically excluded* set of routes despite a
broad `applies_to` field. All of these can score high on cosine similarity.
`evidence.py` treats similarity purely as a ranking signal for what to
surface — the actual accept/reject decision runs through four independent,
deterministic checks (exclusion, route match, date overlap, direction of
impact).

**How is hallucination prevented, specifically?**
Three layers: (1) retrieval only ever proposes real `Note` objects loaded
from `context_notes.csv` — there is no free-text generation step that could
invent a note id, a date, or an event; (2) every `reason` string is built
by string-templating fields of the one note that was actually accepted (or
actually closest-and-rejected) — never generated independently of that
object; (3) `_select_supporting_candidate` only accepts a note whose
`relation == "supports"`, which requires passing all four validation
checks — a merely-plausible-sounding note is never enough.

**Why do some anomalies remain unexplained?**
Because no note in the 10-row corpus both names the affected route (or "All
Routes," without excluding it) and describes a cost-increasing event
overlapping that specific week. The brief is explicit that this should
produce "unexplained," not a best-guess — "a confident wrong reason scores
worse than couldn't find one." Under the ML-driven default (`submission/output.csv`),
7 of 728 route-weeks land here; under the preserved rule-based pipeline
(`submission/benchmark_rule_based_output.csv`), it's 23 — different counts
because the two pipelines flag different candidate weeks in the first place,
not because evidence validation behaves differently (it's identical code in
both). See `submission/adversarial_evaluation.md` for the spot-check
methodology, run against both sets.

**How is deterministic output ensured?**
No randomness anywhere: no LLM calls, no sampling, no `random` module use,
no dependence on wall-clock time or dict/set iteration order affecting the
*result* (sorting is always explicit before output). Verified by actually
running the CLI three times in three separate OS processes and
SHA-256-comparing the outputs (`scripts/check_reproducibility.py`), not by
code inspection alone.

**How does evaluation differ from unit testing here?**
Unit tests (`tests/test_*.py`) check individual functions against
hand-computed expected values in isolation (e.g., "does `build_weekly_metrics`
correctly exclude the current week from its own baseline?"). The evaluation
in `submission/evaluation_report.md` and `submission/adversarial_evaluation.md`
additionally (a) runs the *whole* pipeline end-to-end on the real dataset
and reports the actual verdict distribution, and (b) manually re-reads the
real notes behind every real "justified" row against their cited
route/week — closer to an integration + manual audit check than a unit
test.

**Why these dependencies (standard library + pytest only)?**
At this data scale (2,940 shipments, 10 notes, 7 routes), pandas/numpy
would add a dependency for operations that `csv` + plain dicts handle
clearly and just as fast (sub-second full run), and a vector database or
embeddings API would add infrastructure with no recall benefit over
exhaustive TF-IDF at 10 documents. Fewer dependencies also means the
reproducibility and "runs without a paid API key" guarantees are easier to
verify by inspection.

**Why is the production pipeline ML-driven now, when the earlier comparison found neither model
added clear value?**
That comparison (`submission/ml_evaluation_report.md`) was evaluating whether IF/LOF added anything
*beyond* the rule-based rule's own signal and found little evidence either way — it never argued the
rule-based rule was *better*, just that the comparison alone didn't justify a switch. The production
pivot is a separate, later decision (made explicit, not implied) to make the ML signal primary
regardless, with the engineering done properly: a real model-selection criterion, a train/validation
calibration that never touches the held-out test period, and the literal "rising" requirement kept
as a hard gate the model can't override. I was explicit in the docs about what this costs (fewer
alerts, one brief-example divergence) rather than presenting it as a strict improvement.

**Why LOF over Isolation Forest for the production model?**
A real head-to-head, not a coin flip: LOF's standardized separation between normal and synthetic
labeled-anomaly scores (Cohen's d = 24.35) was about 5x larger than Isolation Forest's (4.80) on the
only labeled evaluation available. Isolation Forest was marginally more stable fold-to-fold (CV 1.08
vs. 1.18), but both models' instability tracked the same real weeks, which looks like genuine
temporal clustering of anomalies rather than model-specific noise — so I weighted the much larger,
unambiguous separation margin over the small, likely-shared stability difference. Full numbers in
`docs/model_selection.md`.

**How was the anomaly threshold calibrated, and what's its weakest point?**
LOF's `contamination` was swept over {0.02, 0.03, 0.05, 0.08, 0.10} on a chronological train (420
rows) / validation (140 rows) split, picking the lowest value that kept real false alerts on
validation under 5%; the final 140-row test period was never touched. The weak point, stated plainly
in `docs/model_selection.md`: every candidate hit 100% detection on the injected synthetic
anomalies, so the search didn't get to trade off detection against false alarms — it really just
picked the most conservative option under the false-alert ceiling. A tighter synthetic magnitude
would fix that in a follow-up.

**What did switching to ML actually cost you?**
Concretely: one of the brief's own 3 example rows (Mumbai-Pune, 2025-09-15) no longer gets flagged —
its LOF score falls just under the calibrated boundary. The numeric fields are unchanged; only the
verdict differs, and it's the direct, traceable result of calibrating for fewer false alerts on real
data. I pinned this down with a test (`tests/test_ml_golden_divergence.py`) specifically so it can't
silently change again without someone having to update the documentation that explains it.

**What would make you reconsider whether LOF (vs. Isolation Forest, vs. the rule-based rule) is the
right choice?**
A labeled set of real, independently-confirmed cost anomalies — large enough to measure actual
precision/recall per model, not just agreement with the rule-based baseline or separation on a
synthetic proxy. Without that, "the model flags more things" isn't evidence of anything.

**What should change before a production deployment?**
(1) A real labeled evaluation set, per above — everything else here is a reasoned judgment call made
in its absence. (2) Persist an actual trained model artifact and re-fit it on a schedule, rather than
re-deriving the whole chronological walk-forward on every batch run — fine for a one-shot historical
score, not for a live service scoring new weeks as they arrive (`docs/ml_driven_pipeline.md`). (3) A
trend-aware flagging rule, so a genuinely new multi-week cost shift is flagged once, not every week it
persists. (4) Move the keyword direction classifier toward a small, still-auditable rule set validated
against a much larger, labeled notes corpus — the current rules were built by reading only 10 real
notes. (5) Add the stretch-goal NL Q&A layer, answering strictly from the computed
`evidence_ledger.csv`/`LedgerEntry` data, never from an unconstrained LLM response. (6) Add structured
logging and a data-quality dashboard if this runs on a live, continuously-updated feed rather than a
static CSV snapshot.
