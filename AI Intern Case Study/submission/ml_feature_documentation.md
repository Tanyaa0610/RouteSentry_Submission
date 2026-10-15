# ML Feature Documentation

All 9 features used by both Isolation Forest and LOF, built in `ml/features.py`.
None of `flagged`, `reason`, `matched_note_id`, or any context-note text is ever used as a feature.

| Feature | Definition | Leakage-free? |
|---|---|---|
| `cost_per_tonne_km` | This route-week's own aggregate cost per tonne-km (`routesentry.weekly`'s ratio-of-sums) | Yes — only this week's own shipments |
| `vs_own_history_pct` | Same definition as the rule-based baseline: % vs. the trailing average of up to 8 strictly-prior weeks | Yes — no current/future week ever enters the average |
| `vs_similar_routes_pct` | Same definition as the baseline: % vs. the same-week average across other routes of the same `route_type`, excluding self | Yes — same week only, never a future or adjacent week |
| `wow_change_pct` | % change vs. the single most recent prior observed week for this route | Yes — strictly prior week only |
| `prior_volatility_pct` | Coefficient of variation (population stdev / mean x 100) of the same trailing prior-week window used for `vs_own_history_pct` | Yes — prior weeks only |
| `n_prior_weeks` | Count of prior weeks actually used (capped at 8) | Yes — a count of past data, not a future-looking value |
| `route_type_short` / `_medium` / `_long` | One-hot encoding of the given (not derived) `route_type` | Yes — static attribute of the route |

## Eligibility gate

A route-week is only included in model training/scoring if:

- `n_prior_weeks >= 4` (stricter than the baseline's own minimum of 1, since
  `prior_volatility_pct` and `wow_change_pct` need more than a point or two of history to be stable), and
- a same-week peer average exists (`vs_similar_routes_pct is not None`).

Rows failing this gate are reported in `submission/experimental_ml_predictions.csv` with
`ml_status=insufficient_history` and blank scores — never imputed or defaulted to zero.

## Preprocessing

A single `sklearn.preprocessing.StandardScaler` is fit per chronological fold, on that fold's TRAINING
rows only, then applied to both that fold's training and test rows. No scaler is ever fit on the full
dataset or on any row at or after the week being scored.
