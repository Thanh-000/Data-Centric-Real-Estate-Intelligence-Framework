# Final Submission Summary

## Final System

This repository contains one final validated DC-REIF system for the King County House Sales dataset.

- valuation core: `XGBoost`
- contextual market grouping: `KMeans`
- uncertainty method: `conformal_prediction_residual_quantile_localized`
- decision layer: Pricing Anomaly Detection on sale-price data

## Validation Commands

Validated with:

- `python scripts/download_data.py`
- `python -m pytest -q`
- `python scripts/run_pipeline.py`
- `python scripts/build_report_results.py`
- `python scripts/build_diagnostics.py`

Notebook validation was also attempted with:

- `python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_dc_reif_king_county.ipynb`

## Validation Outcomes

- dataset download and checksum reuse: passed
- test suite: passed
- end-to-end pipeline: passed
- report-results build: passed
- diagnostics build: passed

## Official Metrics

- selected model: `xgboost`
- validation RMSE: `111003.71`
- test RMSE: `118687.65`
- validation MAE: `65533.19`
- test MAE: `69904.96`
- validation R2: `0.8960`
- test R2: `0.9007`
- segment count: `3`
- silhouette score: `0.1774`
- davies-bouldin index: `1.7775`
- interval coverage: `0.9330`
- average interval width: `372280.01`
- conformal q-hat: `140076.25`

## Environment Note

On this local Windows + OneDrive environment, `nbconvert` may fail before notebook execution starts because Jupyter cannot apply Windows ACLs to its kernel connection file (`SetFileSecurity: Access is denied`). This is an environment-specific launcher issue rather than a pipeline or notebook-logic failure.

## Recommended Validation Path

For reproducible final review, use the script-first workflow locally or in Colab:

1. `python scripts/download_data.py`
2. `python scripts/run_pipeline.py`
3. `python scripts/build_report_results.py`
4. `python scripts/build_diagnostics.py`
5. `python -m pytest -q`
