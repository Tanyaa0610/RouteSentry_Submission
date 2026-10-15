# RouteSentry — Smart Shipping Cost Anomaly Detection

## Overview

RouteSentry is a shipment cost monitoring system developed for the FreightTiger Software Engineering Intern Case Study. It identifies unusual increases in shipping costs, compares routes against their historical performance and similar routes, and checks contextual notes to determine whether a cost increase has a documented explanation.

The system prioritizes deterministic calculations, evidence-backed verdicts, reproducibility, and transparent reporting.

## Problem Statement

Shipping costs can increase because of fuel price changes, toll updates, weather disruptions, and festival surcharges. However, not every increase is justified.

RouteSentry analyzes shipment records to identify unusual cost patterns and classifies flagged route-weeks as:

- **JUSTIFIED:** A relevant contextual note supports the cost increase.
- **UNEXPLAINED:** An anomaly is detected, but no sufficiently relevant supporting note is found.
- **NORMAL:** No qualifying anomaly is detected.

A contextual note is not treated as evidence merely because it mentions the same city or a similar event. Its route, applicable dates, and content must support the explanation.

## Features

- Route-level shipment aggregation using origin, destination, and `route_type`.
- Weekly cost-per-tonne-kilometre calculations.
- Historical and same-week peer comparisons.
- Anomaly detection using statistical and machine-learning approaches.
- TF-IDF-based contextual note retrieval.
- Evidence validation before assigning a justified verdict.
- CSV output with the required column structure.
- Automated tests, reproducibility checks, and execution cost reporting.

## Methodology

### 1. Data Processing

The system loads shipment records, validates the required fields, and groups shipments by route and route type.

Shipment dates are assigned to Monday-to-Sunday weeks, with `week_of` representing the Monday of each week.

### 2. Cost Normalization

Cost efficiency is measured using:

`Cost per tonne-km = Total freight cost / (Quantity in tonnes × Distance in km)`

This normalization makes it possible to compare routes while accounting for shipment quantity and distance.

### 3. Comparison Baselines

**Historical baseline**

The current week's cost is compared with the average cost per tonne-km over the route's previous eight available weeks. Only weeks strictly before the current week are considered. When fewer than eight prior weeks exist, all available prior weeks are used without padding or extrapolation.

**Peer baseline**

The current week's cost is compared with the average cost per tonne-km of other routes sharing the same `route_type` in that week. The route being evaluated is excluded from its own peer average.

### 4. Anomaly Detection

RouteSentry supports rule-based and machine-learning anomaly detection modes.

- **Rule-based mode:** Uses explicit threshold and baseline comparisons to identify unusual cost increases.
- **ML mode:** Uses a machine-learning detector to identify unusual patterns in the processed route-week data.

The two approaches can produce different flags because their detection criteria differ. The selected mode should be stated when reporting results, and the output CSV should be generated using the intended submission mode.

### 5. Contextual Evidence and Verdicts

The system retrieves potentially relevant contextual notes using TF-IDF similarity and validates whether the retrieved evidence supports the route and time period being evaluated.

A justified verdict must have a supporting note, and its identifier must be recorded in `matched_note_id`. When no qualifying note is available, the anomaly remains unexplained rather than being assigned a speculative cause.

TF-IDF retrieval is a lightweight information-retrieval approach; it is not an LLM-based generation pipeline or a vector-database RAG implementation.

## Output

The generated CSV contains one record per evaluated route-week and follows the included output-format reference.

The output includes:

- Route and week information.
- Current cost and comparison metrics.
- Anomaly verdict and explanation.
- `matched_note_id` for the supporting contextual note, when applicable.

The column names, order, and types must match the supplied case-study schema exactly. The final submission CSV should be generated from the same pipeline mode documented for the submission.

## Project Structure

The project includes the following artifacts:

- **Source code:** Data processing, baseline calculations, anomaly detection, retrieval, and evidence validation.
- **Input CSV files:** Shipment records and contextual notes.
- **Output CSV:** Generated route-week analysis.
- **Tests:** Checks for calculations, dates, evidence validation, verdicts, and output formatting.
- **Evaluation artifacts:** Reproducibility results and token/cost documentation.

Refer to the repository files for the exact module names and execution commands.

## Validation and Testing

The project includes targeted tests covering:

- Monday-based weekly aggregation.
- Cost-per-tonne-kilometre calculations.
- Historical and peer baselines.
- Input validation and edge cases.
- Contextual evidence matching.
- Verdict assignment and output schema.
- Case-study worked examples.

The repository currently documents 175 collected tests. This is the number of collected tests, not a claim that all 175 passed. The full test suite should be executed before submission, and its actual results should be reported.

The case-study worked examples should also be checked against the selected pipeline mode.

## Reproducibility

The case study requires three runs on the same input to produce identical flags and numeric values.

The reproducibility check should compare the generated outputs from three untouched runs, including verdicts, numeric fields, and matched note identifiers. Any differences should be investigated before submission.

Deterministic processing and evidence validation help ensure that explanations do not change the underlying verdict or numeric calculations.

## Token and Cost Report

The submitted execution report records:

- Input tokens.
- Output tokens.
- Number of LLM calls.
- Estimated execution cost.

The documented pipeline uses no LLM calls for its standard processing and TF-IDF retrieval path. Consequently, its LLM token usage and LLM API cost are zero for that run. This does not imply that all computation has zero resource cost.

## Limitations and Trade-offs

- Rule-based and ML-based detectors can produce different anomaly flags.
- TF-IDF retrieval depends on the wording of contextual notes and may not identify every relevant note.
- A retrieved note is not automatically accepted as justification; evidence must be validated.
- Routes with limited historical observations use fewer than eight prior weeks when necessary.
- The system reports patterns in the supplied data and does not independently establish the real-world cause of a cost increase.

## Running the Project

Use the commands and dependencies documented in the repository's source files and existing execution instructions.

For a final submission run:

1. Use the supplied shipment records and contextual notes.
2. Select and document the intended anomaly detection mode.
3. Generate the output CSV.
4. Verify the output schema against the supplied reference.
5. Run the test suite.
6. Compare three runs for reproducibility.
7. Record the execution metrics and token/cost report.

## Conclusion

RouteSentry combines shipment cost analysis, route-level comparisons, anomaly detection, and contextual evidence retrieval in a reproducible pipeline. Its primary design principle is to distinguish an unusual cost increase from a documented explanation, avoiding unsupported claims when the available evidence is insufficient.
