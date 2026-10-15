# Anomaly threshold analysis

The brief gives two exact comparison *definitions* (`vs_own_history`,
`vs_similar_routes`) but leaves the "how big a rise counts as out of the
ordinary" threshold unspecified. This document records the exact rule-based
flagging rule, the actual distribution of both percentages across the real
2,940-row dataset, and the reasoning behind the chosen cutoff —
`RISE_THRESHOLD_PCT = 15.0` in `routesentry/config.py`. (For the ML-driven
default pipeline's threshold, calibrated differently, see
`docs/model_selection.md` instead — this document is specific to the
preserved rule-based pipeline, `run.py --mode rule_based`.)

## The flagging rule

A route-week is flagged as a statistical anomaly under the rule-based pipeline only if:

1. It is actually **rising**: `vs_own_history > 0%` (a route that is structurally pricier than one
   peer, but flat against its own past, is not "rising" and is never flagged — see
   `routesentry/anomaly.py:27`), **and**
2. The rise clears the **+15 percentage point** bar on at least one of the two prescribed comparisons
   — either `vs_own_history` itself, or (given it's already rising) `vs_similar_routes`.

```
vs_own_history = (current_week_cost / trailing_avg - 1) × 100
vs_similar_routes = (current_week_cost / peer_avg - 1) × 100
```

(Full formulas for `trailing_avg` and `peer_avg` are in `docs/baseline_definitions.md`.)

## Method

Computed once, deterministically, over all 728 route-week rows produced by
`build_weekly_metrics` on `shipment_records.csv` (no notes/evidence
involved at this stage — pure numeric distribution):

```
n(vs_own_history, defined) = 721   (7 rows have 0 prior weeks and are excluded)
n(vs_similar_routes, defined) = 728
```

| percentile | vs_own_history % | vs_similar_routes % |
|---|---|---|
| p50 | 0.22 | -0.38 |
| p75 | 2.61 | 5.87 |
| p90 | 4.48 | 10.60 |
| p95 | 5.75 | 13.48 |
| p99 | 11.62 | 31.98 |

Rows exceeding +15%:

- `vs_own_history > 15%`: 6 / 721 rows (0.83%, ≈ 99th percentile)
- `vs_similar_routes > 15%`: 27 / 728 rows (3.71%, ≈ 96th percentile)

## Why 15 percentage points

- It sits above the ~99th percentile of the route's own week-to-week
  noise and above the ~96th percentile of cross-route variation at this
  route_type granularity — i.e. it flags genuinely unusual weeks, not
  ordinary variance.
- It is a single round, easy-to-explain number rather than a
  per-route-type or statistically-fitted cutoff, which keeps the rule
  auditable by a non-technical reviewer.
- It is **not** derived from `sample_output_format_v2.csv`. A wide range
  of thresholds (roughly 12%–23%, see `tests/test_golden_examples.py` for
  the exact rows this constrains) reproduces the same three verdicts in
  that file, so the sample does not pin down 15% specifically — the
  percentile analysis above does.

## Limitation

A flat threshold does not distinguish a persistent trend (e.g. the
Mumbai-Pune route, which drifts upward against its peers for most of
2024–2025 under this rule — see `submission/benchmark_rule_based_output.csv`,
the preserved output of the pipeline this document describes) from a sharp
one-off spike. This rule flags every qualifying week independently; it does
not deduplicate or summarize a multi-week run into a single incident. This
is called out again in the README's "Known limitations" section.

Note: `submission/output.csv` (no longer this rule-based pipeline's output —
see `docs/model_selection.md`) does not show this same repeated pattern,
because the ML-driven default's calibrated threshold is considerably more
conservative; see `submission/ml_limitations.md` for that pipeline's own
limitations instead.
