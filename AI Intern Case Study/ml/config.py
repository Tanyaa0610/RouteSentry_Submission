"""All tunable ML-evaluation policy constants, in one place, documented --
same philosophy as routesentry/config.py.
"""

# --- Reproducibility --------------------------------------------------------
RANDOM_SEED = 42

# --- Feature eligibility -----------------------------------------------------
# A route-week needs this many strictly-prior observed weeks for its history-
# based features (prior_volatility_pct, wow_change_pct, vs_own_history_pct)
# to be considered stable enough to feed a model. This is stricter than the
# rule-based baseline's own minimum of 1 prior week (baseline reports
# "insufficient history" only at 0) -- for ML we want the rolling statistics
# to reflect more than one or two points before trusting them as features.
MIN_PRIOR_WEEKS_FOR_ML = 4

# --- Chronological (expanding-window) evaluation ----------------------------
# Width of each scored (test) window, in weeks. Matches the baseline's own
# 8-week trailing-history concept for consistency/interpretability.
FOLD_STEP_WEEKS = 8

# Minimum number of ELIGIBLE training rows (across all routes, pooled)
# required before a fold is scored at all. Below this, a model fit would be
# on too little data to be meaningful; those rows are reported as
# "excluded: insufficient training data" rather than scored with a
# barely-fit model.
MIN_TRAIN_SIZE = 40

# --- Isolation Forest --------------------------------------------------------
# Expected anomaly fraction. Not fit to match the rule-based baseline's own
# alert rate (which runs ~1-4% depending on which comparison trips it, see
# docs/threshold_analysis.md) -- deliberately a separate, independently
# documented policy choice for this separate detector.
ISOLATION_FOREST_CONTAMINATION = 0.05
ISOLATION_FOREST_N_ESTIMATORS = 200

# --- Local Outlier Factor ----------------------------------------------------
LOF_CONTAMINATION = 0.05
# n_neighbors is chosen per training fold from its size: max(LOF_K_MIN,
# min(LOF_K_MAX, train_size // LOF_K_DIVISOR)), then capped below
# train_size (sklearn requires n_neighbors < n_samples_fit).
LOF_K_MIN = 5
LOF_K_MAX = 20
LOF_K_DIVISOR = 10

# --- Synthetic controlled evaluation -----------------------------------------
# Number of synthetic anomalous rows injected per synthetic run, and how
# extreme their comparison percentages are set relative to a typical real
# route-week. This suite exists ONLY to sanity-check that each model can
# detect an obviously extreme, labeled case -- it is not a measurement of
# real-world precision/recall (see submission/ml_evaluation_report.md).
SYNTHETIC_N_ANOMALIES = 30
SYNTHETIC_N_NORMAL_HOLDOUT = 120
SYNTHETIC_PCT_MAGNITUDE = 80.0  # injected vs_own_history/vs_similar_routes/wow %, chosen to be far outside the real data's observed range (see docs/threshold_analysis.md percentiles)

# --- Sensitivity analysis ----------------------------------------------------
# Alternate contamination values re-run to report how alert counts move
# under "reasonable" parameter changes (requirement: do not pick a winner
# based on raw alert counts, but DO show how sensitive each model is).
SENSITIVITY_CONTAMINATIONS = [0.03, 0.05, 0.10]

# --- Production (ML-driven primary pipeline) ---------------------------------
# LOF was selected as the primary production detector over Isolation Forest
# based on the real, computed head-to-head comparison in
# docs/model_selection.md: a standardized separation margin (Cohen's d)
# between real-normal and synthetic-anomaly scores of 24.35 for LOF vs. 4.80
# for Isolation Forest -- a ~5x larger margin -- outweighs Isolation
# Forest's marginally lower fold-to-fold alert-rate variance (CV 1.08 vs
# 1.18, a ~9% relative difference that both models share the same pattern
# of, suggesting it reflects real temporal clustering of anomalies rather
# than model instability).
#
# PRIMARY_LOF_CONTAMINATION is NOT the same value as the experimental
# LOF_CONTAMINATION above (0.05, used only by the separate, already-reported
# parallel evaluation in submission/ml_evaluation_report.md). This value was
# calibrated by ml/calibration.py using ONLY a chronological train
# (2024-01-29..2025-03-17, 420 rows) / validation (2025-03-24..2025-08-04,
# 140 rows) split of real data plus synthetic anomalies injected into the
# validation period -- the held-out test period (2025-08-11..2025-12-22,
# 140 rows) was never touched during calibration. See
# docs/model_selection.md for the full grid search table and its stated
# limitation (the synthetic magnitude used is easily separable, so the
# grid could not discriminate on detection rate -- selection was driven by
# minimizing real-data false alerts on the validation period instead).
PRIMARY_LOF_CONTAMINATION = 0.02
