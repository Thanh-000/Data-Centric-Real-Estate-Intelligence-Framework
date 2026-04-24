# Future Architecture Notes

This folder contains future-facing architecture notes that were created during earlier modernization work. They are kept as roadmap material only.

These notes do **not** describe the current implemented repository as if it were already a full DC-REIF 2.0 system. The active implementation today is the dataset-aligned King County workflow documented in the project root [README.md](</c:/Users/Admin/OneDrive/Desktop/New folder (4)/README.md>).

## Current Implemented System

- dataset: King County House Sales
- baseline model: `LinearRegression`
- main valuation model: `RandomForestRegressor`
- candidate challenger path: `XGBoost` in the separate v1.6 experiment
- segmentation: `KMeans`
- uncertainty: conformal residual quantile intervals
- final decision layer: Pricing Anomaly Detection on sale-price data

## Why These Notes Remain

- to preserve future research ideas without mixing them into the active workflow
- to show that some organizational wrapper packages were introduced during refactoring
- to keep a clear boundary between implemented functionality and roadmap ideas

## Read This Folder As

- future architecture notes
- possible research directions
- non-authoritative design sketches

Do not read this folder as the primary description of what the repository currently runs end to end.
