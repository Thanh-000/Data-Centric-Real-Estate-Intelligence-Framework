# Beyond Price Prediction: A Data-Centric Real Estate Intelligence Framework (DC-REIF)

DC-REIF is a reproducible Python project for the **King County House Sales** dataset. The project is intentionally **data-centric**, **single-core**, and **trust-aware**:

- governance and integrity checks come first
- one baseline model and one main valuation model only
- submarket clustering is used as market-context encoding
- anomaly analysis uses **out-of-fold fair values**
- outputs are explainable and Colab-compatible

This repository performs **Pricing Anomaly Detection / Valuation Gap Analysis on sale prices**. It does **not** claim strict asking-price mispricing detection.

## Architecture

### 1. Control Plane
- raw data manifest with SHA-256 checksum logging
- leakage prevention
- train-only preprocessing
- out-of-fold valuation protocol
- reproducible paths and configuration
- local, GitHub-cloned, and Colab-compatible execution

### 2. Data Governance & Integrity Layer
- raw CSV ingestion
- schema audit
- missingness and duplicate checks
- invalid-value checks
- deterministic cleaning with flag-first logic

### 3. Market Representation Layer
- structural, spatial, temporal, and age features
- KMeans submarket encoding from ex-ante variables only
- cluster profiling with silhouette and Davies-Bouldin diagnostics

### 4. Reliable Valuation Core
- baseline: `LinearRegression`
- main model: `RandomForestRegressor`
- chronological split design
- out-of-fold fair value estimates
- RMSE, MAE, MAPE, and R2 evaluation

### 5. Trust & Decision Support Layer
- conformal-style residual interval estimation
- pricing anomaly flags and valuation-gap scoring
- feature importance and optional SHAP summary output
- final property intelligence table

## Repository Layout

```text
.
|-- AGENTS.md
|-- Makefile
|-- README.md
|-- data/
|   |-- artifacts/
|   |-- interim/
|   |-- processed/
|   `-- raw/
|-- notebooks/
|   `-- 01_dc_reif_king_county.ipynb
|-- outputs/
|   |-- figures/
|   |-- final_report_pack/
|   |-- reports/
|   `-- tables/
|-- scripts/
|   |-- bootstrap_env.sh
|   |-- build_report_results.py
|   |-- download_data.py
|   |-- run_pipeline.py
|   `-- run_tests.sh
|-- src/
|   `-- dc_reif/
`-- tests/
```

Raw data and generated outputs are ignored by `.gitignore`, with a small set of intentional summary artifacts allowed for documentation and report use.

## Dataset

- Dataset: King County house sales
- Canonical file name: `kc_house_data.csv`
- Default public mirror:
  `https://raw.githubusercontent.com/randellmwania/Kings-County-Housing-Project/master/data/kc_house_data.csv`
- Default SHA-256 for that mirror:
  `970a5ee2b0294257cdb18952813df2dd05974f923a9a07ae17fa1af39da71dce`

The default workflow downloads from an anonymous direct URL. If you switch to an authenticated host such as Kaggle, the downloader fails clearly and explains what credentialed step is needed.

## Setup

### Local Python Environment

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional convenience command:

```bash
make install
```

## Automated Data Download

Run the downloader directly:

```bash
python scripts/download_data.py
```

Supported configuration sources:

- environment variables
- `src/dc_reif/config.py`
- command-line arguments

Relevant options:

- `DATA_URL`
- `DATA_FILENAME`
- `DATA_DIR`
- `DATA_CHECKSUM`
- `USE_ARIA2`
- `FORCE_DOWNLOAD`

Example:

```bash
DATA_URL=https://example.com/kc_house_data.csv \
DATA_FILENAME=kc_house_data.csv \
USE_ARIA2=true \
python scripts/download_data.py
```

### `aria2c` Behavior

- If `aria2c` is in `PATH` and `USE_ARIA2=true`, the downloader uses:
  `aria2c -x 8 -s 8 -k 1M -d <dir> -o <filename> <url>`
- If `aria2c` is missing, the system logs a warning and falls back automatically.
- Fallback order is:
  `requests` streaming, then `wget` if available, then `urllib`

### Auth-Protected Sources

- Direct anonymous file URLs are the default path.
- Kaggle-style browser URLs are rejected with a clear message.
- Optional Kaggle CLI support is available for `kaggle://owner/dataset/file.csv` references if the Kaggle CLI and credentials are configured.

## Run End-to-End After Cloning from GitHub

After cloning:

```bash
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
python -m pytest -q
```

The pipeline entrypoint will:

1. download the dataset if it is missing
2. write the raw-data manifest
3. validate schema and data quality
4. clean and feature-engineer the dataset
5. fit submarket clustering on train data only
6. train the baseline and main valuation models
7. generate out-of-fold fair values
8. estimate uncertainty intervals
9. compute pricing anomaly flags
10. save figures, tables, reports, and serialized artifacts

## Build the Report-Ready Results Pack

Run:

```bash
python scripts/build_report_results.py
```

Or with `make`:

```bash
make report-results
```

This presentation-layer script reads the existing frozen pipeline outputs and assembles the official reusable summary artifacts used across the final report, slide deck, notebook, and GitHub documentation.

Canonical source-of-truth files:

- `outputs/reports/final_results_master.json`
- `outputs/reports/final_results_master.md`
- `outputs/reports/final_results_master.csv`

Report-ready package:

- `outputs/final_report_pack/01_core_metrics.md`
- `outputs/final_report_pack/02_results_summary_table.csv`
- `outputs/final_report_pack/03_anomaly_summary_table.csv`
- `outputs/final_report_pack/04_selected_figures_manifest.md`
- `outputs/final_report_pack/05_selected_tables_manifest.md`
- `outputs/final_report_pack/06_caption_bank.md`
- `outputs/final_report_pack/07_case_examples.csv`
- `outputs/final_report_pack/08_core_metrics_table.tex`

## Run the Notebook Locally

```bash
python -m notebook notebooks/01_dc_reif_king_county.ipynb
```

The notebook uses the same package modules as the scripts and is designed to run top-to-bottom once the dataset is available.

## Run on Google Colab

Open `notebooks/01_dc_reif_king_county.ipynb` in Colab after pushing the repository to GitHub.

Suggested flow:

1. Push this repository to GitHub.
2. In Colab, open the notebook from the GitHub tab.
3. Run the bootstrap cell.
4. Let the notebook install requirements in Colab.
5. Optionally mount Google Drive.
6. Run the remaining cells top-to-bottom.

Colab compatibility details:

- path resolution uses `pathlib`
- helper functions detect Colab and optionally mount Drive
- outputs default to `/content/outputs` in Colab
- if Drive is mounted and a configured Drive folder exists, it can be used instead
- if automatic download is unavailable, upload `kc_house_data.csv` manually, place it in Drive, or set `DATA_PATH` to the uploaded file location

## Tests

Run:

```bash
python -m pytest -q
```

The test suite covers:

- schema expectations
- leakage-safe feature selection
- chronological split logic
- preprocessing behavior with train-fit encoders
- end-to-end pipeline output columns
- report-results summary assembly

## Outputs

Typical saved artifacts:

- `data/artifacts/raw_data_manifest.json`
- `data/processed/kc_house_data_clean.csv`
- `data/processed/kc_house_features.csv`
- `outputs/tables/data_quality_report.csv`
- `outputs/tables/model_comparison.csv`
- `outputs/tables/cluster_profiles.csv`
- `outputs/tables/property_intelligence_table.csv`
- `outputs/figures/feature_importance.png`
- `outputs/figures/shap_summary.png`
- `outputs/reports/pipeline_summary.md`
- `outputs/reports/final_results_master.json`
- `outputs/final_report_pack/`
- serialized model artifacts under `data/artifacts/`

## Current Verified Run

The current repository was exercised locally on **April 23, 2026** against the default public mirror.

Observed saved summary metrics from that run:

- selected model: `random_forest`
- validation RMSE: `129084.55`
- test RMSE: `147002.45`
- test empirical interval coverage: `0.886`

## Limitations

- This system operates on realized sale prices, so anomaly outputs are valuation-gap signals rather than strict asking-price mispricing labels.
- Public mirrors of the King County dataset can encode a few variables differently; the cleaning layer standardizes common numeric and text forms.
- The uncertainty layer is intentionally lightweight and practical for a student-scale CPU workflow.
- The project does not claim causal inference.

## Convenience Commands

```bash
make install
make download
make run
make report-results
make notebook
make test
make clean
```
