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

Predictive features are derived only from observed property, temporal, renovation, and geospatial-context fields available in the source dataset.

The feature layer separates descriptive-only variables from predictive variables. For example, `price_per_sqft` may be useful for human-readable analysis, but it is not allowed in the model branch because it is directly target-derived.

## 6. Split Design

The workflow uses chronological train, validation, and test splits. This prevents future transactions from influencing model selection, preprocessing, or evaluation for earlier transactions.

All learned preprocessing objects, encoders, scalers, imputers, clustering mappings, and model parameters are fit on training data or training folds only.

## 7. Market Representation

KMeans is used as contextual market grouping, not as a definitive market-boundary estimate. Segment labels are model features that help represent coarse market context.

The clustering workflow is fit on training data, then assigned to validation/test rows through the fitted clustering pipeline. This avoids fitting market structure on the full dataset before model evaluation.

## 8. Valuation Core

The official valuation core is XGBoost. Model selection is validation-driven and CPU-friendly. The framework stores metrics, selected parameters, and fitted artifacts at runtime so users can inspect the run they produced locally.

The model estimates sale price from leakage-safe features. It should be interpreted as a predictive model over this dataset, not as a causal explanation of real estate prices.

## 9. Fair-Value and Anomaly Logic

Training-era anomaly analysis uses out-of-fold fair-value estimates. This prevents in-sample fitted predictions from being treated as honest fair-value evidence.

For holdout rows, the workflow uses forward predictions from the selected model. Pricing anomaly labels compare observed sale prices with model-supported fair-value intervals and mark records as:

- within expected range
- potentially over-valued
- potentially under-valued
- insufficient history

The insufficient-history path is intentional. It avoids forcing a confident conclusion when local support is weak.

## 10. Uncertainty Layer

The uncertainty layer uses localized conformal prediction residual quantiles. It provides practical interval estimates around fair-value predictions and reports empirical coverage on the holdout period.

The interval method is lightweight and reproducible. It is not a fully heteroscedastic uncertainty model, and difficult slices such as upper-price properties should still receive human review.

## 11. Explainability

The pipeline generates global feature importance and optional SHAP summaries at runtime when dependencies and fitted artifacts are available. These outputs explain model behavior, not causality.

Local driver summaries support review of anomalous cases, but they should not be treated as proof that a feature caused a sale price.

## 12. Reproducibility Contract

The expected verification path is:

1. Install dependencies.
2. Download the dataset through `scripts/download_data.py` or `scripts/quickstart.py`.
3. Run `scripts/run_pipeline.py`.
4. Run `python -m pytest -q`.

The repository is intentionally kept as a framework. Generated data, model artifacts, figures, tables, reports, and final narrative materials are excluded from version control.
