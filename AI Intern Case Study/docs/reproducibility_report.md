# Reproducibility Report (rule-based)

Command: `python3 run.py --mode rule_based` run 3 times in separate processes, same untouched inputs.

## output.csv SHA-256 per run

- run 1: `3eb0f7cc771acd0cc301c5d21af035a77a80ae7585082afdfcedb44ac2df6172`
- run 2: `3eb0f7cc771acd0cc301c5d21af035a77a80ae7585082afdfcedb44ac2df6172`
- run 3: `3eb0f7cc771acd0cc301c5d21af035a77a80ae7585082afdfcedb44ac2df6172`

**Identical across all 3 runs: YES**

## evidence_ledger.csv SHA-256 per run

- run 1: `2c34674b8080c4dab6f187301ebe21fc524bb20c13a3677bd45d02f376d96c78`
- run 2: `2c34674b8080c4dab6f187301ebe21fc524bb20c13a3677bd45d02f376d96c78`
- run 3: `2c34674b8080c4dab6f187301ebe21fc524bb20c13a3677bd45d02f376d96c78`

**Identical across all 3 runs: YES**

## Overall result

**PASS**: 3 untouched runs on the same input produced identical output.csv and evidence_ledger.csv (byte-for-byte, via SHA-256).

No randomness is used anywhere in the rule-based pipeline (no LLM calls, no sampling, no wall-clock-dependent values in the computed fields) so exact equality is expected and enforced by this check.