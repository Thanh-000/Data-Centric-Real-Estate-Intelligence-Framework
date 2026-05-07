# Methodology

This document describes the runnable DC-REIF framework and the methodological controls that keep the workflow reproducible, leakage-aware, and appropriately scoped.

## 1. Problem Scope

The project uses the King County House Sales dataset and models realized sale prices. The downstream decision product is Pricing Anomaly Detection / Valuation Gap Analysis for observed transactions.

The system does not estimate asking-price policy, investment advice, or causal price effects. It produces model-supported fair-value estimates, uncertainty intervals, and anomaly labels that should be reviewed with domain context.

## 2. Data Governance

The dataset is downloaded at runtime from the configured public URL. The default source is pinned by SHA-256 checksum in `configs/data_contracts/data_manifest.json`.

Runtime data is deliberately excluded from version control:

- `data/raw/`: downloaded raw CSV files
- `data/processed/`: cleaned and feature-engineered CSV files
- `data/artifacts/`: runtime manifests and fitted model artifacts

Static data contracts live outside `data/` in `configs/data_contracts/` so the `data/` tree remains a runtime workspace rather than a second source of truth.

## 3. Schema Validation

The schema contract defines the required columns for the King County dataset. The pipeline validates raw data before cleaning or modeling. Missing required columns stop the workflow early rather than allowing silent downstream failure.

The target column is `price`. Target-derived variables such as `price_per_sqft`, fair-value estimates, valuation gaps, and anomaly scores are forbidden from the predictive feature path.

## 4. Cleaning Policy

Cleaning is deterministic and conservative:

- parse sale dates into a stable datetime representation
- preserve valid market extremes instead of dropping them as outliers
- remove or flag structurally invalid records
- add quality flags before downstream modeling

The goal is not to create a perfect real estate dataset. The goal is to make the transformations explicit, repeatable, and auditable.

## 5. Feature Policy

Predictive features are derived from observed property, temporal, renovation, historical-spatial, and geospatial-context fields available in the source dataset.

The feature layer separates descriptive-only variables from predictive variables. For example, `price_per_sqft` may be useful for human-readable analysis, but it is not allowed in the model branch because it is directly target-derived.

Renovation is represented as `renovated_flag`, `years_since_renovation`, and `renovation_recency`; the raw `yr_renovated = 0` field is not used directly as a predictive feature. Historical spatial features use strictly earlier sale dates only, such as prior zipcode median price and nearby prior-sale median price. Same-day transactions and future holdout rows are not used as historical evidence for a row.

## 6. Split Design

The workflow uses time-based train, validation, and test splits. By default, the King County run trains through 2014-12-31, validates through 2015-03-31, and tests on later 2015 transactions when those periods are available. If a small custom dataset cannot support those cutoffs, the splitter falls back to chronological fractions.

All learned preprocessing objects, encoders, scalers, imputers, clustering mappings, and model parameters are fit on training data or training folds only.

## 7. Market Representation

Zipcode-based submarket grouping is used as pragmatic real-estate market context. This replaces the previous KMeans segment layer because low-silhouette KMeans clusters were not meaningful enough to defend as market boundaries.

The segment mapping is fit on training data, then assigned to validation/test rows by zipcode. Rare or unseen zipcodes fall back to an explicit `segment_zipcode_other` label.

## 8. Valuation Core

The official valuation core is XGBoost. Model selection is validation-driven and CPU-friendly, with log-target candidates included to reduce price-level heteroscedasticity. The framework also writes a baseline comparison against median, linear regression, and random forest models so users can judge whether XGBoost adds value.

The model estimates sale price from leakage-safe features. It should be interpreted as a predictive model over this dataset, not as a causal explanation of real estate prices.

## 9. Fair-Value and Anomaly Logic

Calibration uses out-of-fold fair-value estimates. This prevents in-sample fitted predictions from being treated as honest uncertainty evidence.

For holdout rows, the workflow uses forward predictions from the selected model. Pricing anomaly labels compare observed sale prices with model-supported fair-value intervals and mark records as:

- within expected range
- potentially over-valued
- potentially under-valued
- insufficient history

The runtime property table now uses fallback scoring from the selected final model so low-support rows can still be reviewed. Low-support rows are marked through evidence and slice-risk fields instead of being silently dropped from the review surface.

## 10. Uncertainty Layer

The uncertainty layer uses localized conformal prediction residual quantiles by predicted price decile and segment. Calibration is taken from the chronological validation holdout immediately before the test window, then evaluated on later test transactions. The default alpha is 0.10, so the nominal target is 90% coverage; slice-level coverage, especially for the upper price band, is reported separately because real-estate residuals drift over time and thin zipcode segments can be unstable.

The highest predicted price bands receive an explicit upper-tail interval correction. This intentionally widens intervals for expensive properties, where residual variance is larger and under-coverage is more costly. The tradeoff is fewer model-flagged candidates and wider high-price intervals; these flags should therefore be interpreted as review prioritization, not final valuation decisions.

The interval method is lightweight and reproducible. It is not a fully heteroscedastic uncertainty model, so the pipeline writes interval-width distributions by price band and segment, plus q-hat min/median/max audit fields, to make width behavior explicit.

## 11. Explainability

The pipeline generates global feature importance and SHAP summary outputs at runtime. These outputs explain model behavior, not causality.

Local driver summaries support review of anomalous cases, but they should not be treated as proof that a feature caused a sale price.

## 12. Reproducibility Contract

The expected verification path is:

1. Install dependencies.
2. Download the dataset through `scripts/download_data.py` or `scripts/quickstart.py`.
3. Run `scripts/run_pipeline.py`.
4. Run `python -m pytest -q`.

The repository is intentionally kept as a framework. Generated data, model artifacts, figures, tables, reports, and final narrative materials are excluded from version control.
