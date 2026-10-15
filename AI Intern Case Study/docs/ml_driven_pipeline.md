# ML-Driven Pipeline — Architecture

How `run.py`'s default path (`ml/official_pipeline.py`) produces `submission/output.csv`. See
`docs/model_selection.md` for why LOF and why contamination=0.02, and
`submission/ml_limitations.md` for what this design does not and cannot prove.

## What changed from the rule-based pipeline, and what didn't

| Stage | Rule-based (`routesentry/`) | ML-driven (`ml/official_pipeline.py`) |
|---|---|---|
| Weekly cost aggregation | `routesentry.weekly.build_weekly_metrics` | **same, unchanged** |
| `vs_own_history` / `vs_similar_routes` | exact brief-mandated formulas | **same, unchanged** |
| Statistical anomaly signal | flat +15pp threshold (`routesentry.anomaly`) | **calibrated LOF score** (`ml.signal`) |
| "Must be rising" gate | `vs_own_history > 0` | **same rule, same gate — not ML-decided** |
| Note retrieval | TF-IDF + eligibility prefilter | **same, unchanged** |
| Evidence validation | route/date/direction checks | **same, unchanged** |
| Verdict assembly | `routesentry.verdict.decide` | **same function, reused as-is** |
| Output CSV schema | `routesentry.pipeline.write_output_csv` | **same function, reused as-is** |

The only new logic is the statistical-signal step. Everything downstream of "is this route-week a
candidate anomaly" — evidence retrieval, evidence validation, verdict wording, the CSV writers — is
literally the same code, imported and called unchanged. This is deliberate: it means every existing
guarantee about evidence-grounding and hallucination-avoidance (see `docs/evidence_validation.md`)
carries over to the ML-driven pipeline without re-proving it from scratch, and it's why
`routesentry/` needed zero modifications for this pivot.

## Training, validation, and scoring

There is no single, persisted "the model" file. This is a deliberate choice, not an oversight:

- The task is a one-shot **batch** scoring of a static historical dataset (728 route-weeks), not a
  live service that scores one new week against a previously-trained model.
- A single global model fit once on all 728 rows and then used to score all of them would leak every
  row's own future into its own training set — exactly the kind of look-ahead the brief's own
  baseline definitions explicitly forbid for `vs_own_history`.
- Instead, `ml/chronological.py`'s existing expanding-window walk-forward evaluation is reused
  unchanged: at each 8-week fold boundary, a fresh `StandardScaler` + LOF model is fit **only** on
  eligible rows strictly before that boundary (at least `MIN_TRAIN_SIZE=40` pooled rows across all
  routes), then used to score the next 8-week block. The window then advances. This is the same
  leakage-free procedure already tested in `tests/test_ml_leakage.py`, just now feeding the
  **production** verdict instead of an experimental comparison.

If a true live-deployment version of this were built (scoring next week's shipments against a model
trained on everything seen so far), a persisted artifact (e.g. via `joblib`) would make sense then —
see "Future improvements" in the README. It isn't needed for this batch task, and persisting one now
would invite it being misapplied in a way that reintroduces leakage.

## Features

Unchanged from the earlier experimental comparison (`ml/features.py`,
`submission/ml_feature_documentation.md`): `cost_per_tonne_km`, `vs_own_history_pct`,
`vs_similar_routes_pct`, `wow_change_pct`, `prior_volatility_pct`, `n_prior_weeks`, and a
`route_type` one-hot (Short/Medium/Long — never ordinal-encoded, since route types have no
meaningful order for a distance-based detector to exploit). None of `flagged`, `reason`,
`matched_note_id`, or any note text is ever used as a feature. Eligibility requires at least 4
strictly-prior observed weeks and an available same-week peer average — see `ml/config.py`.

## What rows get no ML-driven assessment

A route-week can reach the final verdict stage with no model score at all, for two distinct reasons,
both explicit in the evidence ledger's `signal_rule` text:

1. **Insufficient per-route history** (`eligible=False`, 28 rows): fewer than 4 prior weeks for that
   specific route, or no peer route shipped that week.
2. **Insufficient pooled training data for the fold** (`excluded_insufficient_training=True`, 56
   rows): the route itself has enough history, but the fold's *pooled, cross-route* training set
   (needed to fit LOF at all) hadn't yet reached `MIN_TRAIN_SIZE=40` rows.

Both cases resolve to `final_verdict = "No"` with an explicit, non-guessed reason — never an
assumed "normal" silently standing in for "not actually assessed." This is a real trade-off against
the rule-based pipeline, which could assess `vs_own_history` with as little as 1 prior week per
route: **84 of 728 rows (11.5%) get no ML-driven statistical assessment**, versus only 7 rows (1%)
under the rule-based rule (the first week of each of the 7 routes). See
`submission/ml_limitations.md` for the full discussion.

## Commands

```bash
python3 run.py                              # ML-driven (default) -> submission/output.csv
python3 run.py --mode rule_based            # original deterministic rule, preserved
python3 run.py --recalibrate                # re-run the calibration grid search first (ml mode only)
python3 scripts/check_reproducibility.py                 # 3-process check, ML-driven path
python3 scripts/check_reproducibility.py --mode rule_based  # 3-process check, rule-based path
python3 scripts/check_reproducibility.py --experimental-ml  # 3-process check, the separate IF/LOF/baseline comparison
```
