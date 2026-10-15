# RouteSentry

A freight-cost anomaly detector built for FreightTiger's Software Engineering Intern (AI) case
study. **The production pipeline is ML-driven**: a calibrated Local Outlier Factor model, trained
chronologically with no look-ahead, decides whether a route-week's cost is statistically unusual.
The original deterministic rule (a flat threshold on two prescribed comparisons) is fully preserved
and selectable — it's what the brief's own grading examples were validated against — and is kept as
a benchmark artifact rather than deleted.

## 1. Overview

RouteSentry looks at a shipment history, tracks what each route costs per tonne-km week by week,
and flags weeks where a route's cost has risen in a way that doesn't look normal — either against
that route's own recent past, or against other routes of similar length running the same week.
Before calling a rise "unexplained," it checks a small set of free-text operational notes (fuel
price changes, tolls, weather, festival surcharges) to see whether a genuine, verifiable reason
already accounts for it.

"Doesn't look normal" is now decided by a trained anomaly-detection model rather than a fixed
percentage cutoff (see §3 and `docs/model_selection.md`) — but the model is never allowed to decide
that a *falling* cost is a problem, and it never sees note text, so it can't influence which
explanations get accepted any more than the old rule-based signal could.

The output is a single CSV, one row per route-week, with a verdict (`Yes` / `No` / `No
(justified)`), the two numeric comparisons behind that verdict, and — when a note explains the
rise — the exact note id and a plain-English reason built from that note's own text. The column
schema is identical regardless of which detection mechanism produced the verdict.

## 2. The problem, and why it's harder than it looks

Transporters bill by distance and weight, so a route's cost per tonne-km should be fairly stable
week to week. In practice it drifts: a toll goes up, a stretch of highway floods, a festival week
pushes up demand for trucks, or a route just quietly gets more expensive for no traceable reason.
Someone reviewing freight spend wants to know which of those weeks are worth a second look and
which already have a known explanation — without reading through every shipment by hand, and
without trusting a tool that makes up a reason just to clear an alert.

That last part is the actual difficulty. A naive version of this tool is a threshold on a
percentage change. Getting it right requires two more things: a baseline definition precise enough
that two people computing it by hand would get the same number, and a way to accept or reject a
candidate explanation without ever inventing one.

## 3. Approach

The core design principle is simple to state and easy to get wrong in the details: **a cost
increase is worth investigating when the data says it's unusual, and an explanation is only
accepted when the evidence actually supports it** — not when it merely sounds plausible or
mentions the right city.

Concretely, that means keeping three things strictly separate, in three different modules, with no
shortcuts between them:

1. **Is this week's cost unusual?** — answered only from shipment data, by a calibrated LOF model
   (`ml/signal.py`), never from note text.
2. **Does a note genuinely explain it?** — a separate question, answered only from
   `context_notes.csv`, with its own validation rules (`routesentry/evidence.py`, unchanged).
3. **What's the final verdict?** — combines the two (`routesentry/verdict.py`, unchanged), and is
   the only place that decision gets made.

A matching note never overrides the model's score, and a textually-similar note never counts as
evidence just because it mentions the right route. The point of this split isn't architectural
purity for its own sake — it's what makes it possible to prove (with tests, not just a claim) that
the system never flags a falling cost as a rise, and never cites a note that doesn't actually apply
— properties that hold regardless of which detection mechanism sits in step 1.

**Why move step 1 from a fixed rule to a trained model at all?** A flat +15pp threshold (still
available via `--mode rule_based`, see §10) treats every qualifying week the same regardless of how
noisy that specific route normally is, and it was chosen by reading a percentile table once, not by
validating against held-out data. An anomaly-detection model fit on each route-week's own recent
volatility, trend, and peer context can in principle tell "a route that's always jumpy" apart from
"a route that's never moved like this before" — the trade-off, made explicit in §10 and
`docs/model_selection.md`, is that model's threshold has to be calibrated from data too, and on this
dataset the calibrated version turned out meaningfully more conservative, not more sensitive.

One thing did not change: the brief's literal words — "flag any route where the cost is **rising**"
— are enforced as a hard, non-negotiable gate before the model's score is even consulted
(`ml/signal.py`). Isolation Forest and LOF are undirected outlier detectors; left alone they flag
unusual cost in *either* direction, and on this data most of what they'd add beyond the gate turns
out to be unusually low-cost weeks (see `submission/ml_evaluation_report.md`). That is out of scope
for this brief, so it isn't left to the model to get right.

## 4. Why two baselines, not one

A route can look expensive for two different reasons, and they call for different responses:

- **Against its own history** — if a route that normally runs ₹3/tonne-km jumps to ₹4, that's
  worth flagging regardless of what anything else costs.
- **Against similar routes right now** — if every Short route got pricier this week (say, a
  nationwide fuel hike), a route tracking its own history alone would look normal, because its
  *own* baseline hasn't caught up yet. Comparing it to peers of the same `route_type` in the same
  week catches a shared, route-type-wide shift that a pure self-history check would miss for
  several weeks.

Neither comparison alone is enough — that's why the brief asks for both, and why RouteSentry treats
a rise as a candidate anomaly if it clears either bar (details and the exact threshold rationale in
`docs/threshold_analysis.md`).

## 5. Why contextual evidence has to be validated, not just matched

The easy way to build the "AI" part of this is to search `context_notes.csv` for a route name and
treat a hit as an explanation. That fails in several very normal cases that the brief explicitly
calls out: a note might name the right route but describe a *different* time period; a note might
say costs were explicitly **not** affected; a globally-worded note ("All Routes") might exclude the
exact routes in this dataset; a note might describe an improvement, not a disruption. A keyword or
similarity match doesn't distinguish any of these.

So retrieval (finding candidate notes) and validation (deciding whether one actually counts) are
two separate steps. Retrieval ranks what's worth looking at; validation runs every candidate
through checks for route applicability, date overlap, and whether the note's content actually
describes a cost increase, in that order (full detail in `docs/evidence_validation.md`). A note
only becomes a cited justification if it survives every check — a wrong route, a wrong window, or
a note that denies an effect, leads straight to "unexplained," which the brief treats as the
correct output when no real support exists.

## 6. Architecture and data flow

```
shipment_records.csv ──► routesentry/data_io.py (load + validate)
                              │
                              ▼
                      routesentry/weekly.py — group by route + week, compute
                                   cost_per_tonne_km, vs_own_history, vs_similar_routes
                              │
                              ▼
                      ml/features.py — build the LOF feature vector (same
                                   numbers above, plus wow_change, volatility,
                                   route_type) for every eligible route-week
                              │
                              ▼
                      ml/chronological.py — walk-forward: fit a scaler + LOF
                                   per fold on strictly-prior pooled rows, score
                                   the next 8-week block, advance
                              │
                              ▼
                      ml/signal.py — "rising" gate (not ML-decided) + the
                                   calibrated LOF score -> statistical signal
                              │
context_notes.csv ──► routesentry/retrieval.py — TF-IDF candidates for this route-week
                              │
                              ▼
                      routesentry/evidence.py — deterministic validation: route
                                     match, date overlap, direction of impact
                              │
                              ▼
                      routesentry/verdict.py — combine signal + validated evidence
                                    → Yes / No / No (justified) + reason
                              │
                              ▼
                      ml/official_pipeline.py — orchestrates all of the above,
                                   writes the CSVs via routesentry.pipeline's writers
                              │
                ┌─────────────┴──────────────┐
                ▼                             ▼
   submission/output.csv          submission/evidence_ledger.csv
   (the grading contract)         (full per-candidate audit trail)
```

`run.py --mode rule_based` runs the same diagram with `ml/features.py` through `ml/signal.py`
replaced by `routesentry/anomaly.py`'s flat-threshold rule — everything from `retrieval.py` onward
is the identical code either way. Each stage lives in its own module and does one job: nothing
upstream of the signal step reads note text, nothing in `evidence.py` computes a baseline, and
`verdict.py` is the only place a signal and evidence are combined — so there's no second, divergent
implementation of the verdict logic anywhere else in the codebase, in either mode.

## 7. Input data and output contract

**`shipment_records.csv`** (2,940 rows): `shipment_id, origin, destination, route_type
(Short/Medium/Long, given — never re-derived), material, quantity_tonnes, distance_km,
freight_cost_inr, shipment_date, transporter`.

**`context_notes.csv`** (10 rows): `note_id, date, applies_to` (a specific `Origin-Destination`
route, or `"All Routes"`), `note` (free text).

**`submission/output.csv`** — the graded deliverable, column order fixed by
`sample_output_format_v2.csv`:

`route, week_of, cost_per_tonne_km, vs_own_history, vs_similar_routes, flagged, matched_note_id,
reason`

- `cost_per_tonne_km` = Σfreight_cost_inr / Σ(quantity_tonnes × distance_km) for that route-week —
  a ratio of sums, not an average of per-shipment ratios.
- `vs_own_history` / `vs_similar_routes` are formatted as in the brief's own sample (e.g. `+29.5%
  vs this route's past average`), or an explicit unavailability message when there isn't enough
  history or no peer route shipped that week — never a number invented to fill the gap.
- `flagged` is one of `Yes` (anomalous, unexplained), `No (justified)` (anomalous, explained by a
  validated note), or `No` (not anomalous).
- `matched_note_id` carries the exact note id from `context_notes.csv`, blank unless the verdict is
  `No (justified)`.

728 rows — one per (route, week) that had at least one shipment. The 3-row sample file is
illustrative of the column contract, not the full row set; see
`docs/requirements_traceability.md` for why the full 728 are included. Full baseline formulas
(historical and peer) are in `docs/baseline_definitions.md`.

**Under the default ML-driven mode**, 84 of the 728 rows (11.5%) get no statistical model
assessment at all — not enough pooled training history yet — and resolve to `No` with an explicit
"not evaluated" reason rather than a guess (vs. 7 rows under the rule-based mode, which needs much
less history per route). See `docs/ml_driven_pipeline.md` for exactly which rows and why.

## 8. Installation and execution

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.9+. `scikit-learn` and `numpy` are now required for the default (ML-driven)
pipeline, not just the optional comparison — `pytest` remains the only *test* dependency.

Run the pipeline:

```bash
python3 run.py                        # ML-driven (default): calibrated LOF decides the flag
python3 run.py --mode rule_based      # original deterministic +15pp-threshold rule, preserved
python3 run.py --recalibrate          # ML mode: re-run the contamination grid search first
```

Writes `submission/output.csv` and `submission/evidence_ledger.csv` by default (override with
`--shipments`, `--notes`, `--output`, `--ledger`; `--threshold` applies only to `--mode rule_based`).
The ML-driven default takes a few seconds (fitting one LOF model per chronological fold); the
rule-based mode runs in well under a second. No network access, no API key, no database, either way.

## 9. Testing and reproducibility

```bash
python3 -m pytest tests/ -v                               # 136 tests
python3 scripts/check_reproducibility.py                  # ML-driven, 3 separate processes
python3 scripts/check_reproducibility.py --mode rule_based # rule-based, 3 separate processes
```

All currently pass: 136/136 tests, and both reproducibility checks report byte-identical output
across 3 untouched processes (`docs/ml_driven_reproducibility_report.md`,
`docs/reproducibility_report.md`). The rule-based mode has no randomness at all; the ML-driven mode
fixes a random seed (`ml.config.RANDOM_SEED = 42`) for every scaler/model fit, so exact equality is
the expected, verified result in both cases — not an assumption.

Test coverage spans the aggregation math (ratio-of-sums, week boundaries), both baselines (no
look-ahead, correct peer exclusion), 12 hallucination-guardrail cases in evidence validation
(unchanged, shared by both modes), the output schema under both modes, and the brief's own 3 worked
examples — reproduced exactly under `--mode rule_based`
(`tests/test_golden_examples.py`), 2 of 3 exactly under the ML-driven default with the third
divergence explicitly pinned down (`tests/test_ml_golden_divergence.py`, explained in
`docs/model_selection.md`). ML-specific tests cover the "must be rising" gate overriding the model
(`tests/test_ml_signal.py`), leakage-free chronological fitting and calibration
(`tests/test_ml_leakage.py`, `tests/test_ml_calibration.py`), and that the ML detector is actually
invoked rather than silently bypassed (`tests/test_ml_official_pipeline.py`). A separate 14-case
adversarial suite for the evidence layer (wrong route, wrong window, a note that denies an effect,
etc.) is documented in `submission/adversarial_evaluation.md`. Full numbers in
`submission/evaluation_report.md`.

LLM usage: none, in either mode. `submission/token_cost_report.md` reports 0 tokens, 0 calls, $0 —
the ML-driven pipeline is a classical scikit-learn model trained locally, not an LLM call.

## 10. The ML-driven pipeline: selection, calibration, and what it costs

RouteSentry started as a rule-based detector with a separate, non-integrated experimental comparison
against Isolation Forest and LOF (that comparison still exists, unchanged, see below). It was later
rebuilt so that comparison's winner drives the actual production verdict. This section explains how,
and is honest about what changed in the process.

**Model selection.** Both models were already evaluated side by side in the original comparison.
Picking a production model took two additional, actually-computed criteria: a standardized
separation margin (Cohen's d) between real-normal and synthetic-anomaly LOF/IF scores — **24.35 for
LOF vs. 4.80 for Isolation Forest**, a 5x difference — and fold-to-fold alert-rate stability on real
data, where Isolation Forest was only marginally steadier (CV 1.08 vs. 1.18) in a way that looks more
like shared temporal clustering of real anomalies than a genuine stability difference. **LOF was
selected.** Full numbers in `docs/model_selection.md`.

**Calibration.** LOF's `contamination` parameter (which sets its decision boundary) was calibrated
using a chronological train (420 rows) / validation (140 rows) split of real data plus synthetic
anomalies injected into the validation period — the final 140-row test period was never touched
during calibration. The selected value, **0.02**, is more conservative than the 0.05 used in the
earlier exploratory comparison; the grid search and its one real limitation (every candidate hit
100% synthetic detection, so the result was effectively driven by minimizing real false alerts, not
a genuine detection/false-alarm trade-off) are both in `docs/model_selection.md`.

**Training and scoring.** No single persisted model file exists, deliberately: this is a one-shot
batch score over historical data, and the leakage-free way to do that is the same expanding
chronological window already built for the experimental comparison (`ml/chronological.py`) — a fresh
scaler + LOF fit per 8-week fold, using only strictly-prior pooled rows. Full design in
`docs/ml_driven_pipeline.md`.

**Results on real data vs. synthetic.** The ML-driven pipeline raises far fewer statistical signals
than the rule-based rule — 10 route-weeks (7 unexplained + 3 justified) vs. 26 — because the
calibration explicitly optimized for fewer false alerts on held-out real data. The synthetic suite
(extreme, labeled, injected cases) is detected reliably by both models at every tested contamination;
that is a pipeline-wiring sanity check, not evidence about real-world recall, and is never conflated
with the real-data numbers above (see `submission/ml_limitations.md`).

**Known limitations and false-positive risk.** The hard "rising" gate means the model can't create a
new *kind* of false positive the rule-based rule couldn't also produce (a falling week can never be
flagged, in either mode); what the calibration changes is the false-*negative* rate relative to the
rule-based rule — one of the brief's own 3 worked examples (Mumbai-Pune, 2025-09-15) is no longer
flagged under the calibrated threshold, documented and test-pinned in
`tests/test_ml_golden_divergence.py`. A more aggressive contamination (0.05–0.08) would recover more
rule-based-style sensitivity at a correspondingly higher, but still bounded and known, real
false-alert rate (2.1%–6.4% on the validation period) — a genuine, disclosed trade-off, not something
hidden behind a single "best" number. 84 route-weeks (11.5%) get no model assessment at all due to
insufficient pooled training history early in the dataset (§7) — more than the rule-based rule's 7.

**The original rule-based pipeline is not a historical artifact — it's a live, selectable mode** and
the one the brief's own sample rows were validated against:

```bash
python3 run.py --mode rule_based                          # the original pipeline, unchanged
```

**The earlier, separate experimental comparison** (not the production calibration above) still
exists and still runs independently:

```bash
python3 scripts/run_ml_evaluation.py                       # writes submission/ml_*
python3 scripts/check_reproducibility.py --experimental-ml # 3 separate-process runs
```

Full methodology, real-data results, the synthetic suite, and a sensitivity analysis for that
earlier comparison are in `submission/ml_evaluation_report.md` — it predates and is distinct from
the calibrated production pipeline described above, which supersedes its "neither model is
integrated" conclusion.

## 11. Limitations and future improvements

- **Calibration was limited by an easy synthetic test.** Every contamination candidate hit 100%
  synthetic detection, so the grid search couldn't actually trade off detection vs. false alarms —
  it reduced to minimizing real false alerts. A tighter synthetic magnitude would make the next
  calibration run more discriminating (`docs/model_selection.md`).
- **The ML-driven pipeline covers fewer rows than the rule-based rule** (84 vs. 7 "not evaluated"
  route-weeks) because it needs a pooled, cross-route training set, not just one route's own history.
- **The ML-driven pipeline is measurably more conservative** than the rule-based rule on this data —
  10 statistical signals vs. 26 — which is a disclosed consequence of calibrating against false
  alerts on real data, not a universal property of "ML vs. rules."
- **No real ground-truth labels exist** for any route-week in either mode, so no precision/recall/F1
  claim is made anywhere for real data — only the labeled synthetic sanity check, kept clearly
  separate (`submission/ml_limitations.md`).
- The note-matching rules are a small, hand-built keyword list, built by reading the 10 real notes —
  shared unchanged by both modes. It's fully auditable and reproducible, but a much larger or
  differently-worded notes corpus would need that list extended by hand.
- The brief's stretch goal — asking the assistant plain-English questions about its findings — is
  not implemented.
- No persisted model artifact exists (§10, `docs/ml_driven_pipeline.md`) because this is a batch
  score over static historical data, not a live service — a real production deployment scoring new
  incoming weeks against a model trained on everything seen so far would need one.
- A labeled set of real, independently-confirmed cost anomalies would let both the model-selection
  comparison and the calibration grid search become real accuracy measurements instead of
  synthetic-proxy ones.

## 12. Repository structure and deliverables

```
routesentry/        rule-based pipeline (standard library only) -- preserved, `--mode rule_based`
ml/                  ML-driven production pipeline (default) + the separate experimental comparison
run.py               CLI entrypoint -- both pipelines, via --mode
tests/               136 tests (pytest)
scripts/             check_reproducibility.py, run_ml_evaluation.py
docs/                methodology detail: thresholds, baselines, evidence rules, model selection, traceability
submission/          the actual deliverables (below)
```

`submission/` — required by the case study:

| File | Contents |
|---|---|
| `output.csv` | the graded CSV — **ML-driven** (default pipeline) |
| `benchmark_rule_based_output.csv` | the original rule-based pipeline's output, preserved as a benchmark |
| `evidence_ledger.csv` | full per-candidate audit trail behind every ML-driven verdict |
| `benchmark_rule_based_evidence_ledger.csv` | the same, for the preserved rule-based benchmark |
| `evaluation_report.md` | test and run results |
| `adversarial_evaluation.md` | the 14-case hallucination guardrail check (evidence layer, shared by both modes) |
| `token_cost_report.md` | the $0 LLM cost log |
| `walkthrough.md` | notes for the 10-minute Round 2 walkthrough |
| `interview_questions.md` | prepared answers on design decisions and trade-offs |

Plus the separate, earlier experimental comparison (`experimental_ml_predictions.csv`,
`experimental_ml_model_comparison.csv`, `ml_evaluation_report.md`, `ml_feature_documentation.md`,
`ml_limitations.md`) that the production calibration in §10 was built from.

`docs/` holds the detail this README intentionally keeps out: `threshold_analysis.md`,
`baseline_definitions.md`, `evidence_validation.md`, `model_selection.md`, `ml_driven_pipeline.md`,
`requirements_traceability.md`, `reproducibility_report.md`, `ml_driven_reproducibility_report.md`,
and `experimental_ml_comparison_reproducibility_report.md`.
