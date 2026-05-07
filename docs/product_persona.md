# Product Persona and Workflow

## Primary Persona

Primary user: mortgage valuation reviewer / valuation analyst.

This persona is a strong fit for the current DC-REIF framework because the workflow is conservative, interval-based, and designed to support human review rather than automate a pricing decision.

## Core Job To Be Done

Review realized sale transactions and identify records where the observed sale price appears meaningfully outside a model-supported fair-value range.

The system should help the reviewer answer:

- Which transactions require review first?
- Is the observed sale above, below, or within the model-supported interval?
- How strong is the evidence behind the model estimate?
- Which features most influenced the model's fair-value estimate?
- Where are the blind spots where the model abstains or has weak local support?

## Workflow

1. Run the pipeline on the approved dataset.
2. Open the dashboard or exported review queue.
3. Filter by anomaly label, zipcode, segment, price band, and evidence strength.
4. Inspect property-level details for flagged transactions.
5. Route high-priority cases to manual comparable-sale review.
6. Track abstention clusters as product blind spots.

## Product Policy

The product should not make autonomous valuation decisions. It should provide a triage queue and structured evidence for human review.

Recommended label interpretation:

- `within_expected_range`: no immediate anomaly signal
- `potentially_over_valued`: observed sale price is above model-supported range
- `potentially_under_valued`: observed sale price is below model-supported range
- Low-support rows: remain scored in the current table and should be interpreted through `evidence_strength`, `slice_risk_level`, and confidence notes.

## Success Metrics

Product metrics:

- anomaly review queue size
- low-support rate overall and by segment
- share of high-value homes in abstention
- reviewer acceptance rate for flagged cases
- time saved per review batch

Model/product trust metrics:

- holdout MAE / RMSE
- interval coverage
- average interval width
- slice-level coverage by price band, zipcode, and segment
- false positive / false negative proxy rates after manual review labels exist
