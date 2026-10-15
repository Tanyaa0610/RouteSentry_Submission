# Baseline Definitions — Full Detail

This is the detailed version of README §4 and §7: the exact formulas behind `cost_per_tonne_km`,
`vs_own_history`, and `vs_similar_routes` — all three fixed exactly by the brief, computed
identically by both the rule-based and ML-driven pipelines. For the flagging rule built on top of
these numbers (different in each pipeline — a flat threshold in one, a calibrated model in the
other), see `docs/threshold_analysis.md` (rule-based) and `docs/model_selection.md` (ML-driven).

## Weekly aggregation

```
cost_per_tonne_km(route, week) = Σ freight_cost_inr
                                  ─────────────────────────────────
                                  Σ (quantity_tonnes × distance_km)
```

summed over every shipment with that `route` whose `shipment_date` falls in that Monday–Sunday
week. Weeks are assigned by `shipment_date` before any cost math happens, and weeks run
Monday-to-Sunday with `week_of` set to that week's Monday — both per the brief. This is a ratio of
sums, not an average of individual shipments' own cost/tonne-km ratios; two shipments with very
different per-shipment ratios still aggregate correctly because the numerator and denominator are
each summed independently before dividing (`routesentry/weekly.py`, tested in
`tests/test_weekly.py::test_ratio_of_sums_not_average_of_ratios`).

## Historical baseline (`vs_own_history`)

Trailing average of the route's own `cost_per_tonne_km` over its 8 most recent **prior** weeks —
weeks strictly before the current one, never including it. If fewer than 8 prior weeks of data
exist for that route, all available prior weeks are used as-is (no padding, no extrapolation); if
there are zero prior weeks, the comparison is reported as unavailable rather than computed. This is
a simple arithmetic mean of the weekly `cost_per_tonne_km` values (not a weighted average, not a
median) — the brief's "rolling average" wording is read literally.

```
vs_own_history = (current_week_cost / trailing_avg - 1) × 100
```

## Peer baseline (`vs_similar_routes`)

Simple average of `cost_per_tonne_km` across every *other* route sharing the same `route_type`
(Short/Medium/Long, as given in the data — never re-derived from distance) in the *same* week. The
route being evaluated is always excluded from its own peer average. If no peer route shipped that
week, the comparison is reported as unavailable.

```
vs_similar_routes = (current_week_cost / peer_avg - 1) × 100
```

The brief fixes these two comparison *definitions* exactly but leaves "how unusual is unusual" open
— that's a flagging-rule policy choice, not a baseline definition, and lives in
`docs/threshold_analysis.md` (rule-based: a flat +15pp rule, configurable via
`python3 run.py --mode rule_based --threshold 20`) and `docs/model_selection.md` (ML-driven: a
calibrated LOF score).
