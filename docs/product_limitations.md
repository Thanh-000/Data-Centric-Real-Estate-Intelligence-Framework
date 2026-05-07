# DA Product Limitations and Roadmap

This document evaluates the current DC-REIF framework from a data analytics product perspective. The current repository is a reproducible modeling framework, not yet a complete stakeholder-facing analytics product.

## 1. Early Consumption Layer Only

Severity: severe

Current state:

- The repository now includes a Streamlit dashboard MVP in `app/streamlit_app.py`.
- The dashboard can load `outputs/tables/property_intelligence_table.csv`, filter the review queue, and show anomaly locations when latitude/longitude are available.
- The dashboard is still a prototype, not a complete production consumption layer.

Product impact:

Non-technical users now have a first review interface, but the workflow still needs stronger stakeholder polish: clearer property drill-down, better spatial context, saved review states, and persona-specific triage logic.

Recommended next step:

Extend the dashboard MVP after the core pipeline is stable. The next dashboard iteration should prioritize:

- richer anomaly map by latitude / longitude
- saved filters for anomaly label, zipcode, segment, price band, and evidence strength
- property detail view with observed price, fair-value estimate, interval, and top drivers
- exportable review list for flagged transactions

Suggested implementation:

- Streamlit for the existing internal prototype
- Folium, pydeck, or Plotly Mapbox for spatial review
- CSV/Parquet outputs from the existing pipeline as the dashboard input

## 2. Low-Support Coverage Risk

Severity: severe

Previous reference result:

- `3061` of approximately `21597` transactions are labeled `insufficient_history`.
- This is roughly `14%` of the dataset.

Product impact:

An abstention mechanism is methodologically responsible, but a 14% abstention rate is large for a data product. In real estate, low-support cases may include sparse neighborhoods, new builds, unusual high-value homes, waterfront homes, or other edge cases that stakeholders care about most.

Current implementation change:

- The runtime property table now uses fallback scoring from the selected final model so low-support rows remain inspectable.
- Evidence fields and slice-risk fields carry the caution signal instead of removing the row from review.
- The pipeline writes `outputs/tables/anomaly_threshold_sensitivity.csv` and `outputs/tables/interval_width_*.csv` so threshold and interval-width behavior can be audited.

Recommended next step:

Treat abstention as a product metric, not only a modeling safeguard. The next iteration should report:

- abstention rate by zipcode
- abstention rate by price band
- abstention rate by segment
- abstention rate by property age / renovation status
- overlap between abstention and high-value properties

Current implementation:

- `scripts/analyze_abstention.py` generates abstention tables by zipcode, segment, predicted price band, observed price band, grade band, house-age band, evidence strength, and slice risk level.
- `scripts/evaluate_slices.py` generates slice metrics including abstention rate, anomaly rate, MAE, RMSE, MAPE, interval coverage, and average interval width.

Remaining remediation paths:

- introduce a fallback global interval when local support is weak
- lower confidence level for explicit exploratory review modes
- use hierarchical support: price band + segment -> segment -> price band -> global
- add an explicit `needs_manual_review` queue rather than returning no usable estimate

Success criterion:

Reduce blind-spot concentration in high-value or sparse-market slices while preserving honest uncertainty communication.

## 3. Missing User Persona and Use Case Definition

Severity: medium

Current state:

The framework describes Pricing Anomaly Detection, but it does not choose a primary user:

- broker / agent
- investor
- mortgage valuation reviewer
- assessor / public-sector analyst
- internal data science reviewer

Product impact:

Each persona has different tolerance for false positives, false negatives, interval width, and abstention. Without a persona, it is hard to decide thresholds, dashboard UX, escalation rules, and evaluation metrics.

Recommended next step:

Define one primary user and one secondary user before hardening the consumption layer.

Candidate primary persona:

Mortgage valuation reviewer or analyst reviewing realized sale transactions for possible valuation gaps.

Current implementation:

- `docs/product_persona.md` defines the primary reviewer persona, core review workflow, label interpretation, and success metrics.

Why this persona fits the current framework:

- values conservative uncertainty
- can tolerate a review queue rather than automated decisions
- benefits from interval-based anomaly labels
- needs transparent explanations and evidence strength

Product decisions that depend on persona:

- anomaly threshold
- acceptable abstention rate
- required explanation detail
- whether the system optimizes for recall, precision, or coverage
- escalation workflow for insufficient-history cases

## 4. Static Historical Dataset

Severity: medium

Current state:

The framework uses the static King County House Sales dataset. It does not implement a production ingest loop, data freshness checks, scheduled retraining, model drift monitoring, or data quality monitoring over new batches.

Product impact:

This limits real-world deployment. A stakeholder-facing product would need to know whether the model is still calibrated against current market behavior.

Recommended next step:

Add a production-readiness plan rather than immediately adding complex infrastructure.

Current implementation:

- `docs/monitoring_plan.md` defines the minimum monitoring and retraining plan for data freshness, schema quality, distribution drift, performance drift, abstention monitoring, and promotion gates.

Minimum viable monitoring plan:

- data freshness timestamp
- row count and schema validation for each new batch
- missingness and invalid-value drift
- distribution drift for key variables such as price, sqft_living, grade, zipcode, and sale month
- metric drift on recent labeled periods
- interval coverage monitoring when new observed prices become available

Retraining plan:

- scheduled retraining after enough new transactions arrive
- validation gate before replacing the current model artifact
- compare new model against previous model on holdout and slice metrics
- keep model card metadata for each trained artifact

## 5. Roadmap Needs Priority and Impact

Severity: medium

Current state:

Future work is currently broad. It names useful directions, but it does not prioritize them or explain product impact.

Recommended product roadmap:

### Phase 1: Consumption Layer

Goal: make current outputs usable by non-technical reviewers.

Deliverables:

- anomaly dashboard prototype
- geographic anomaly map
- property detail panel
- filterable review queue

Impact:

Turns the modeling framework into a usable analytics product.

### Phase 2: Abstention Diagnostics

Goal: understand and reduce blind spots.

Deliverables:

- abstention analysis by zipcode, segment, and price band
- fallback interval policy proposal
- separate manual-review queue

Impact:

Improves stakeholder trust and coverage in difficult market slices.

### Phase 3: Persona-Specific Thresholds

Goal: align anomaly labels with a chosen business workflow.

Deliverables:

- primary persona definition
- threshold policy
- precision / recall / coverage tradeoff report
- review workflow notes

Impact:

Connects model behavior to business value.

### Phase 4: Freshness and Monitoring

Goal: prepare the framework for repeated use beyond the static dataset.

Deliverables:

- batch ingest contract
- data drift checks
- calibration monitoring
- retraining trigger policy

Impact:

Moves the project from a reproducible case study toward a maintainable data product.

## Current Product Maturity

Current maturity: strong analytical framework, incomplete analytics product.

The modeling foundation is credible, but the project needs a consumption layer, persona definition, abstention strategy, and monitoring plan before it should be presented as a full DA product.
