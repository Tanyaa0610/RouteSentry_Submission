# Experimental ML Comparison Reproducibility Report

Checks scripts/run_ml_evaluation.py -- the separate, exploratory side-by-side comparison of Isolation Forest / LOF / the rule-based baseline (submission/ml_evaluation_report.md). This is NOT the production ML-driven pipeline (see docs/ml_driven_reproducibility_report.md for that).

Command: `python3 scripts/run_ml_evaluation.py --out-dir <tmp>` run 3 times in separate processes, same untouched inputs, same RANDOM_SEED.

## experimental_ml_predictions.csv SHA-256 per run

- run 1: `9444072f99f02cd187d9b921202f2611445366e8c631720880a89836788c4392`
- run 2: `9444072f99f02cd187d9b921202f2611445366e8c631720880a89836788c4392`
- run 3: `9444072f99f02cd187d9b921202f2611445366e8c631720880a89836788c4392`

**Identical across all 3 runs: YES**

## experimental_ml_model_comparison.csv SHA-256 per run

- run 1: `50dd363877def6a5db31c297675c08b9b4fccbe2a1f128b246a8379bd83dbae6`
- run 2: `50dd363877def6a5db31c297675c08b9b4fccbe2a1f128b246a8379bd83dbae6`
- run 3: `50dd363877def6a5db31c297675c08b9b4fccbe2a1f128b246a8379bd83dbae6`

**Identical across all 3 runs: YES**

## Overall result

**PASS**: 3 untouched runs on the same input and RANDOM_SEED produced identical experimental_ml_predictions.csv and experimental_ml_model_comparison.csv (byte-for-byte, via SHA-256).

`ml_evaluation_report.md`'s prose is intentionally excluded from this byte-identity check: it contains a measured wall-clock runtime line that varies run to run by construction -- every *number* in that report is re-derived from the same underlying `experimental_ml_predictions.csv` checked here, so this check extends to its content transitively.