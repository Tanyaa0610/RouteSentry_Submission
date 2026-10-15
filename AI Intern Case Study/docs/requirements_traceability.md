# Requirements Traceability Matrix

Source: `FreightTiger_Intern_CaseStudy.pdf` ("Build a Smart Assistant That
Watches Shipping Costs"). Every row below was checked against the actual
code and a passing test, not assumed from a report. Re-verified on
2026-10-15 by re-running `pytest tests/ -q` (136 passed) and all three
reproducibility checks (all PASS) against the code as it exists in this
repository right now.

Status vocabulary: **PASS** (implemented and verified by a specific test or
direct inspection), **PARTIAL** (implemented but with a caveat below),
**FAIL** (not implemented), **NOT VERIFIABLE** (cannot be confirmed from
this repository alone, e.g. depends on live Round-2 delivery).

**IMPORTANT — this matrix was written against the original rule-based
pipeline, which is why every row below cites `routesentry/anomaly.py`'s
flat-threshold rule as "the" statistical-signal mechanism.** A later,
separate, explicit user request changed the project's core requirement: the
*default* pipeline (`run.py`, no flags) is now ML-driven
(`ml/official_pipeline.py`, a calibrated LOF model — see
`docs/model_selection.md`), and it is what currently produces
`submission/output.csv`. The rule-based pipeline this matrix describes is
fully preserved and still exactly matches every row below — it is just no
longer the default; run it explicitly with `run.py --mode rule_based`, and
its output is preserved at `submission/benchmark_rule_based_output.csv`.
Rows 8, 16, 17, 20, 21, and 24 below are annotated with how the ML-driven
default compares. This matrix is not rewritten against the ML-driven
pipeline wholesale because the brief's own grading examples
(`sample_output_format_v2.csv`) were written against, and are fully
satisfied only by, the rule-based mechanism (see row 21's annotation and
`docs/model_selection.md` §3 for the one documented divergence).

| # | Requirement | Evidence in the code | Test or validation | Status | Remaining gap |
|---|---|---|---|---|---|
| 1 | Group shipment records by route (same origin and destination) | `routesentry/models.py:28` — `Shipment.route` property returns `f"{origin}-{destination}"` | `tests/test_weekly.py::test_deterministic_output_order` | PASS | none |
| 2 | Group by the given `route_type` (Short/Medium/Long); do not derive it | `routesentry/data_io.py` loads `route_type` verbatim from the CSV and rejects any value outside `{Short, Medium, Long}`; never recomputed from distance anywhere in the codebase | `tests/test_data_io.py::test_unrecognised_route_type_row_is_skipped` | PASS | none |
| 3 | `cost_per_tonne_km` = total freight cost / (Σ quantity_tonnes × distance_km) per route-week | `routesentry/weekly.py:28-34` — ratio of sums, not a mean of per-shipment ratios | `tests/test_weekly.py::test_ratio_of_sums_not_average_of_ratios` | PASS | none |
| 4 | Weeks run Monday–Sunday; `week_of` = that week's Monday; group by `shipment_date` before computing | `routesentry/dates.py:7` (`monday_of`), used in `weekly.py:26` before any aggregation | `tests/test_dates.py` (5 tests), `tests/test_weekly.py::test_monday_to_sunday_grouping` | PASS | none |
| 5 | `vs_own_history`: trailing 8-week rolling average, weeks strictly before current, no look-ahead | `routesentry/weekly.py:49` — `weeks_sorted[:i][-TRAILING_WEEKS:]` excludes index `i` (current week) by construction | `tests/test_weekly.py::test_no_current_week_leakage_in_history`, `::test_no_future_data_leakage` | PASS | none |
| 6 | If fewer than 8 prior weeks exist, use all available, don't pad/extrapolate, document in README | `routesentry/weekly.py:49-54`; `n_prior_weeks` surfaced per row | `tests/test_weekly.py::test_fewer_than_eight_prior_weeks_uses_all_available`; documented in `README.md` §"Historical baseline" | PASS | none |
| 7 | `vs_similar_routes`: same-week average across other routes sharing `route_type`, excluding self | `routesentry/weekly.py:58-62` | `tests/test_weekly.py::test_peer_average_excludes_self_and_other_types`, `::test_no_peers_available_leaves_peer_comparison_unset` | PASS | none |
| 8 | Flag a route where cost is rising **and** out of the ordinary vs. history or peers | `routesentry/anomaly.py:27` — `rising = vh > 0` is a hard gate; a falling cost is never flagged regardless of peer gap | `tests/test_anomaly.py::test_falling_cost_is_never_an_anomaly_even_if_peer_gap_huge`, `::test_threshold_boundary_is_strictly_greater_than` | PASS | The exact "how unusual" threshold (`RISE_THRESHOLD_PCT = 15.0`, `routesentry/config.py:35`) is not specified by the brief — chosen from the real data's percentile distribution, documented in `docs/threshold_analysis.md`, not fit to the 3 sample rows. **ML-driven default:** the same "rising" gate is enforced structurally in `ml/signal.py` (not ML-decided), but "out of the ordinary" is now a calibrated LOF score, not this flat threshold — see `docs/model_selection.md`; `tests/test_ml_signal.py` covers the same gate on the ML path |
| 9 | Check a small set of notes for a real reason before flagging (retrieval/RAG) | `routesentry/retrieval.py` (from-scratch TF-IDF) + `routesentry/pipeline.py::_gather_candidates` (deterministic eligibility prefilter) | Exercised end-to-end in `tests/test_golden_examples.py`; real-run spot check in `submission/adversarial_evaluation.md` | PASS | No vector database used — deliberate choice, documented in `submission/token_cost_report.md` (justified at 10-note scale) |
| 10 | Plain-English explanation backed by what was actually found, not made up | `routesentry/verdict.py::_justified_reason` / `_unexplained_reason` build every `reason` string from fields of the one `Note`/`CandidateEvidence` object actually accepted or rejected — no free-text generation step | `tests/test_verdict.py` (7 tests) | PASS | none |
| 11 | Stretch goal: let someone ask the assistant questions in plain English about its findings | — | — | FAIL | Not implemented — explicitly an optional stretch goal in the brief; out of scope given the time spent verifying the mandatory core and guardrails. See README "Future improvements" |
| 12 | Every "justified" verdict must cite the exact note, and that note must apply to that route and that time | `routesentry/evidence.py::evaluate_candidate` checks route applicability (`applies_to` field) and temporal overlap (`ranges_overlap`) before a note can ever reach `relation == "supports"` | `tests/test_evidence.py` (12 tests covering wrong route, wrong window, excluded routes); `tests/test_output_contract.py::test_justified_rows_always_have_a_note_id`, `::test_accepted_note_id_is_always_a_real_note_id` | PASS | none |
| 13 | A note that says costs weren't affected does not count as justification | `routesentry/evidence.py::classify_direction` classifies "not significantly affected" / "no major disruptions" / etc. as `no_impact`, which can never become `relation == "supports"` | `tests/test_evidence.py::test_note_stating_costs_unaffected_is_rejected` | PASS | none |
| 14 | No supporting note → unexplained; do not guess | `routesentry/verdict.py::decide` — `final_verdict = "Yes"` with `accepted_note_id = None` whenever `_select_supporting_candidate` returns `None` | `tests/test_verdict.py::test_anomalous_without_supporting_note_is_unexplained` | PASS | none |
| 15 | Spot-check justified rows against the given notes | Manual re-read of both real notes (N001, N002) behind the 3 real `"No (justified)"` rows against their cited route and week | `submission/adversarial_evaluation.md` §"Real-dataset spot check" | PASS | This is a one-time manual audit by the author, not an automated test — recorded in writing with the reasoning shown |
| 16 | Show a lightweight eval harness or labeled set, and why you trusted the result | 71 rule-based tests + 14 hand-built adversarial hallucination cases + 25 ML-driven-pipeline tests | `pytest tests/` (136 passed); `submission/adversarial_evaluation.md` (14/14 pass) | PASS | Adversarial cases are synthetic and hand-constructed — they check policy behavior, not real-world predictive accuracy. **ML-driven default:** model selection and calibration are their own documented, evidence-based procedures (`docs/model_selection.md`), separate from this adversarial suite |
| 17 | Three untouched runs on the same input: identical flags and numbers (wording may vary) | `scripts/check_reproducibility.py --mode rule_based` runs `run.py --mode rule_based` 3× in 3 separate OS processes and SHA-256-compares `output.csv` / `evidence_ledger.csv` | Re-run on 2026-10-15: **PASS**, byte-identical — `docs/reproducibility_report.md` | PASS | **ML-driven default:** `scripts/check_reproducibility.py` (no flag) checks `run.py`'s new default the same way — also re-run on 2026-10-15, **PASS**, byte-identical — `docs/ml_driven_reproducibility_report.md`. The ML path is seeded (`RANDOM_SEED=42`) rather than seedless, so exact equality had to be verified, not assumed, by this 3-process check |
| 18 | If anything is random, document the setting and show it doesn't move the verdict | No randomness anywhere: no LLM, no sampling, `random` module never imported in `routesentry/` | `tests/test_baseline_preserved.py::test_routesentry_package_has_no_ml_imports` (confirms no numpy/sklearn either); reproducibility PASS is itself evidence of zero randomness | PASS | Not applicable by construction — there is nothing random to show doesn't move the verdict |
| 19 | Report total input tokens, output tokens, LLM calls, and rough cost for one full run | `submission/token_cost_report.md` — 0 calls, 0 tokens, $0, with the reasoning for why no LLM is used anywhere in `routesentry/` | Confirmed by code inspection: no LLM client import anywhere in `routesentry/` | PASS | none |
| 20 | Output CSV must match `sample_output_format_v2.csv` exactly: column names, order, types, including `matched_note_id` | `routesentry/config.py::OUTPUT_COLUMNS`; `routesentry/pipeline.py::write_output_csv` (uses `lineterminator="\n"` to match the sample's LF line endings exactly) — used unchanged by **both** pipelines | `tests/test_output_contract.py::test_sample_format_columns_match_config` (rule-based); `tests/test_ml_official_pipeline.py::test_ml_output_schema_matches_contract` (ML-driven); byte-level header diff confirmed identical for both | PASS | none |
| 21 | `reason` may be in own words, but must state the verdict; if justified, name the note and reflect its content/dates; note id goes in `matched_note_id`, not only in prose | `routesentry/verdict.py::_justified_reason` includes the note id and its literal text in the `reason` string **and** `LedgerEntry.accepted_note_id` is written to the dedicated `matched_note_id` column — same function, both pipelines | `tests/test_golden_examples.py` (rule-based: all 3 rows match the brief's examples exactly, including `matched_note_id`) | PASS | **ML-driven default diverges on verdict for 1 of the 3 brief examples** (Mumbai-Pune, 2025-09-15: `Yes`→`No`) due to the calibrated threshold being more conservative — numeric fields and the other 2 rows still match exactly. Documented and test-pinned: `tests/test_ml_golden_divergence.py`, `docs/model_selection.md` §3. This is the one row in this entire matrix where the current default behavior differs from what the brief's own sample output shows |
| 22 | Working code | `routesentry/` (11 modules, ~1,060 lines, unchanged) + `ml/` (11 modules, the production ML-driven pipeline plus the earlier experimental comparison), `run.py` | `pytest tests/` — 136/136 pass | PASS | none |
| 23 | README explaining how it works, how it was built, and how correctness was verified before submitting | `README.md` | Manually reviewed for completeness against this exact list during this audit | PASS | none |
| 24 | Reproducibility check across 3 runs (diff or confirming note) | `docs/reproducibility_report.md` (rule-based) and `docs/ml_driven_reproducibility_report.md` (ML-driven default), both generated by `scripts/check_reproducibility.py`, re-run on 2026-10-15 | Re-run during this audit: PASS, both | PASS | none |
| 25 | Token/cost log for one full run | `submission/token_cost_report.md` | Code-inspection confirmed ($0, 0 calls) | PASS | none |
| 26 | 10-minute walkthrough at the start of Round 2 | `submission/walkthrough.md` (prepared speaker notes and real examples) | NOT VERIFIABLE from this repository — the walkthrough itself is delivered live, not as a file | NOT VERIFIABLE | Actual delivery and discussion happen in person; only the prepared script can be reviewed here |

## Summary counts

PASS: 23 · PARTIAL: 0 · FAIL: 1 (stretch-goal NL Q&A, explicitly optional and undone) · NOT VERIFIABLE: 1 (live walkthrough delivery)

These statuses are against the rule-based pipeline this matrix was originally written for, which
still passes every one of them unchanged. Against the **current default** (ML-driven) pipeline, every
status is identical except row 21, where the brief's own sample output and the current default
disagree on 1 of 3 example verdicts (see row 21's annotation and `docs/model_selection.md`).

## Ambiguities and the choices made

These are explicit, documented judgment calls on points the brief leaves
open — not silent assumptions.

1. **Anomaly threshold.** The brief fixes the two comparison *definitions*
   but not the "how big is too big" cutoff (row 8 above). Chose a flat +15
   percentage points on `vs_own_history` (alone) or on `vs_similar_routes`
   (only once already rising vs. own history), justified against the real
   data's actual percentile distribution — see `docs/threshold_analysis.md`.
   Configurable via `--threshold` on `run.py` / `config.RISE_THRESHOLD_PCT`.
2. **Note applicability window when no explicit date range is given.** A
   note like N003 ("Diesel prices rose nationwide starting this week...")
   has no explicit "from X to Y" range. Rather than assume it applies
   indefinitely (which could retroactively "explain" cost rises months
   later), it is scoped to its own ISO week only — a conservative choice
   that trades recall for avoiding a confidently-wrong justification, per
   the brief's own stated preference ("a confident wrong reason scores
   worse than couldn't find one").
3. **Row scope of `output.csv`.** `sample_output_format_v2.csv` shows only
   3 illustrative rows, all anomalous. The actual output contains all 728
   route-week rows (including the 702 `"No"` / not-anomalous ones), because
   the brief asks to "track cost per tonne per km, week by week" for each
   group — not only for flagged weeks — and because this lets a reviewer
   audit the anomaly rule against the full distribution, not just the
   positives. This does not change the verdict vocabulary or column
   contract.
4. **Directionality.** `origin-destination` is directional (`Mumbai-Pune`
   is never merged with a hypothetical `Pune-Mumbai`); the real dataset
   only contains one direction per route pair, so this choice was never
   exercised against actual data but is documented for completeness.
