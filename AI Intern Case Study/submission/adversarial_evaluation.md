# Adversarial Evaluation

Purpose: catch false justifications and misleading retrieval matches in the
evidence-validation layer (`routesentry/evidence.py`, `routesentry/verdict.py`)
before trusting it on the real dataset. These are synthetic, hand-constructed
cases, each targeting one specific way a note could be wrongly accepted or
wrongly rejected — they are a policy check, not a measure of real-world
predictive accuracy (see "Limitations" below).

Run with: `python3 -m pytest tests/test_evidence.py tests/test_verdict.py -v`

## Cases and results

| # | Case | Test | Expected outcome | Result |
|---|---|---|---|---|
| 1 | Correct route, correct event window | `test_valid_route_and_window_supports` | accepted (`relation == "supports"`) | PASS |
| 2 | Correct route, wrong event window | `test_wrong_event_window_is_rejected` | rejected, `date_match == False` | PASS |
| 3 | Correct date, wrong route | `test_correct_date_wrong_route_is_rejected` | rejected, `route_match == False` | PASS |
| 4 | Global note excluding the affected routes | `test_global_note_explicitly_excluding_route_is_rejected` | rejected despite `applies_to == "All Routes"` | PASS |
| 5 | Note stating costs were unaffected | `test_note_stating_costs_unaffected_is_rejected` | rejected, `relation == "contradicts"` | PASS |
| 6 | Note describing improved road conditions | `test_improved_road_conditions_does_not_justify_rise` | rejected (`direction == "improved"`) | PASS |
| 7 | Note about a completed disruption | `test_expired_disruption_outside_window_is_rejected` | rejected, window ended before query week | PASS |
| 8 | No supporting note among candidates | `test_no_supporting_note_among_candidates_means_no_accepted` | `_select_supporting_candidate` returns `None` | PASS |
| 9 | Multiple notes, one supports / one contradicts | `test_supporting_and_contradicting_notes_together_still_accepts_only_supporting` | only the genuinely supporting note is accepted | PASS |
| 9b | Multiple supporting candidates, differing similarity | `test_conflicting_evidence_prefers_higher_similarity_supporting_note` | higher-similarity note wins, deterministically | PASS |
| 10 | Anomaly with no eligible note at all | `test_anomalous_without_supporting_note_is_unexplained` | verdict `"Yes"`, no note id — never guesses | PASS |
| 11 | Two equally-similar valid candidates | `test_conflicting_evidence_tie_break_is_lowest_note_id` | deterministic tie-break (lowest note_id) | PASS |
| 12 | Event beginning exactly on the query week's Monday | `test_week_boundary_event_starting_monday_is_in_window` | counted as overlapping (`date_match == True`) | PASS |
| 13 | Note mentions the route but no cost-impact language | `test_mentions_route_but_no_cost_impact_language_does_not_support` | rejected, `direction == "unknown"` | PASS |
| 14 | Later event cannot explain an earlier anomaly | `test_later_event_cannot_explain_an_earlier_anomaly` | rejected, `date_match == False` | PASS |

**14 / 14 cases pass. 0 false justifications. 0 missed valid-evidence cases** (case 1's valid note is correctly accepted; no case where genuinely valid evidence was wrongly rejected).

## Real-dataset spot check

On the real `shipment_records.csv` + `context_notes.csv`, the pipeline
produces exactly 3 `"No (justified)"` rows, citing only `N001` (twice,
Chennai-Bangalore) and `N002` (once, Ahmedabad-Mumbai) — see
`submission/output.csv` and `submission/evidence_ledger.csv`. Both notes were
manually re-read against their cited route-weeks:

- `N002` (2025-01-20, Ahmedabad-Mumbai festival surcharge) justifies the
  Ahmedabad-Mumbai week of 2025-01-20 — exact route and exact week match.
- `N001` (2025-02-24 to 2025-03-08 flooding, Chennai-Bangalore) justifies
  the Chennai-Bangalore weeks of 2025-02-24 and 2025-03-03, both inside the
  note's explicit date range.

Neither acceptance required extrapolating beyond the note's stated route or
dates. This holds under **both** pipelines, since they share the identical,
unmodified evidence-validation code (`routesentry/evidence.py`) — only the
statistical-signal step that decides *which* rows reach evidence validation
differs (see `docs/model_selection.md`).

The rule-based pipeline's 23 `"Yes"` (unexplained) rows (preserved in
`submission/benchmark_rule_based_output.csv`) were spot-checked by eye
against `context_notes.csv`'s 10 notes; none has a note that both names the
affected route and states a cost-increasing event in that week.

The ML-driven default (`submission/output.csv`) produces a different,
smaller set of 7 `"Yes"` rows (Ahmedabad-Mumbai 2025-03-10 & 2025-03-17,
Chennai-Bangalore 2025-03-10, 2025-03-17 & 2025-05-12, Delhi-Jaipur
2024-11-11 & 2024-11-18) — re-checked the same way: the closest-dated notes
(N002 for Ahmedabad-Mumbai, N001/N009 for Chennai-Bangalore, N007 for
Delhi-Jaipur) each fail on date coverage, direction, or both, exactly as the
deterministic evidence layer's own candidate-rejection text in
`submission/evidence_ledger.csv` records. In both cases, the pipeline is not
under-crediting available evidence.

## Limitations

- These are synthetic unit-level tests of the *validation policy*, not an
  estimate of real-world precision/recall against unseen notes or routes —
  with only 10 notes in the real corpus there is not enough labeled data for
  a statistically meaningful accuracy number.
- The keyword-based `classify_direction` rule set (`evidence.py`) was
  designed by reading all 10 real notes plus the above synthetic phrasings;
  it has not been tested against note wordings outside this vocabulary and
  would need extending (not retraining — it is a transparent rule list) if
  the real notes corpus grows with materially different phrasing.
