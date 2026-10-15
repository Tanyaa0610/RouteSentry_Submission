# Reproducibility Report (ML-driven, production default)

Command: `python3 run.py --mode ml` run 3 times in separate processes, same untouched inputs.

## output.csv SHA-256 per run

- run 1: `0ca4bd624ba735266a1991aaee4e49210c8e047f58c1a71245e292b075977ef4`
- run 2: `0ca4bd624ba735266a1991aaee4e49210c8e047f58c1a71245e292b075977ef4`
- run 3: `0ca4bd624ba735266a1991aaee4e49210c8e047f58c1a71245e292b075977ef4`

**Identical across all 3 runs: YES**

## evidence_ledger.csv SHA-256 per run

- run 1: `4e2502ea59b77490258f36ea1927a649642909aeb222d13eb6f2cfc2ced11c56`
- run 2: `4e2502ea59b77490258f36ea1927a649642909aeb222d13eb6f2cfc2ced11c56`
- run 3: `4e2502ea59b77490258f36ea1927a649642909aeb222d13eb6f2cfc2ced11c56`

**Identical across all 3 runs: YES**

## Overall result

**PASS**: 3 untouched runs on the same input produced identical output.csv and evidence_ledger.csv (byte-for-byte, via SHA-256).

Both the rule-based and ML-driven pipelines are fully deterministic: the rule-based path uses no randomness at all, and the ML-driven path fixes RANDOM_SEED=42 (ml/config.py) for every StandardScaler/LOF fit, with feature rows always explicitly sorted before any split -- no LLM calls, no unseeded sampling, in either path.