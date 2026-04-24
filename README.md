# DC-REIF: Final Dataset-Aligned Real Estate Intelligence System for King County

This repository contains the single final validated DC-REIF implementation for the King County House Sales dataset. It is a student-scale, data-centric, trust-aware real estate intelligence project with one active valuation workflow, one active report-results layer, and one official sale-price Pricing Anomaly Detection output.

The implemented system is intentionally narrow:

- dataset: King County House Sales
- official valuation model: `XGBoost`
- segmentation: `KMeans`
- uncertainty method: conformal residual-quantile intervals
- decision product: **Pricing Anomaly Detection / Valuation Gap Analysis** on **sale-price** data

Historical model comparisons and pre-consolidation artifacts are isolated under `archive/legacy_validation/` and are not part of the active workflow.

## What Is Implemented

### Data Governance
- automated dataset download with checksum verification and `aria2c` fallback
- raw-data manifest logging and schema validation
- deterministic cleaning and data-quality flags
- leakage-safe feature policy and train-only preprocessing

### Market Representation
- structural, spatial, temporal, renovation, and geospatial-context features derived from the King County dataset only
- KMeans submarket representation and cluster profiling

### Valuation And Trust Layer
- one official valuation model: `XGBoost`
- chronological train / validation / test design
- out-of-fold fair value estimates
- conformal residual-quantile uncertainty intervals
- sale-price Pricing Anomaly Detection with insufficient-history abstention
- feature importance and optional SHAP outputs

### Reporting Layer
- one canonical `final_results_master` summary
- one `final_report_pack` for report, slides, notebook, and GitHub documentation reuse
- slice-level diagnostics by segment and price band

## Repository Layout

```text
repo-root/
|-- .github/
|-- archive/
|-- configs/
|-- data/
|-- docs/
|-- notebooks/
|-- outputs/
|-- scripts/
|-- src/dc_reif/
|-- submission/
`-- tests/
```

Main public entrypoints:

- `python scripts/download_data.py`
- `python scripts/run_pipeline.py`
- `python scripts/build_report_results.py`
- `python scripts/build_diagnostics.py`

Main notebook:

- `notebooks/01_dc_reif_king_county.ipynb`

## Official Results

The official metrics are documented in:

- `docs/reporting/official_metrics.md`
- `outputs/reports/final_results_master.json`
- `outputs/reports/final_results_master.md`
- `outputs/reports/final_results_master.csv`

Current official values:

- selected model: `xgboost`
- validation RMSE: `110852.91`
- test RMSE: `121629.66`
- validation MAE: `65525.94`
- test MAE: `70271.43`
- validation R2: `0.8963`
- test R2: `0.8957`
- segment count: `6`
- silhouette score: `0.1839`
- davies-bouldin index: `1.6291`
- interval method: `conformal_prediction_residual_quantile`
- interval coverage: `0.8831`
- average interval width: `280938.25`
- conformal q-hat: `140469.12`
- anomaly counts:
  - within expected range: `16630`
  - potentially over-valued: `1125`
  - potentially under-valued: `781`
  - insufficient history: `3061`

## Setup

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or:

```bash
make install
```

## Automated Data Download

Run:

```bash
python scripts/download_data.py
```

Supported configuration:

- `DATA_URL`
- `DATA_FILENAME`
- `DATA_DIR`
- `DATA_CHECKSUM`
- `USE_ARIA2`
- `FORCE_DOWNLOAD`

Behavior:

- prefers `aria2c` when installed and enabled
- falls back to `requests`, then `wget`, then `urllib`
- verifies checksum for the default public mirror
- fails clearly for auth-protected browser-only links

## Run End-to-End

```bash
python scripts/run_pipeline.py
python scripts/build_report_results.py
python scripts/build_diagnostics.py
python -m pytest -q
```

Or with `make`:

```bash
make download
make run
make report-results
make diagnostics
make test
```

## Report-Ready Results Layer

Build the canonical results summary with:

```bash
python scripts/build_report_results.py
```

Build the report diagnostics pack with:

```bash
python scripts/build_diagnostics.py
```

Canonical source-of-truth artifacts:

- `outputs/reports/final_results_master.json`
- `outputs/reports/final_results_master.md`
- `outputs/reports/final_results_master.csv`

Core report-pack artifacts:

- `outputs/final_report_pack/01_core_metrics.md`
- `outputs/final_report_pack/02_results_summary_table.csv`
- `outputs/final_report_pack/03_anomaly_summary_table.csv`
- `outputs/final_report_pack/04_selected_figures_manifest.md`
- `outputs/final_report_pack/05_selected_tables_manifest.md`
- `outputs/final_report_pack/06_caption_bank.md`
- `outputs/final_report_pack/07_case_examples.csv`
- `outputs/final_report_pack/08_core_metrics_table.tex`
- `outputs/final_report_pack/09_error_by_segment.csv`
- `outputs/final_report_pack/10_error_by_price_band.csv`
- `outputs/final_report_pack/11_coverage_by_segment.csv`
- `outputs/final_report_pack/12_coverage_by_price_band.csv`
- `outputs/final_report_pack/13_anomaly_by_segment.csv`
- `outputs/final_report_pack/14_anomaly_by_price_band.csv`
- `outputs/final_report_pack/14_improved_segment_profiles.csv`
- `outputs/final_report_pack/15_anomaly_casebook.csv`
- `outputs/final_report_pack/16_interpretation_notes.md`
- `outputs/final_report_pack/17_geospatial_feature_notes.md`

## Google Colab

The repository remains Colab-compatible by design.

Suggested Colab sequence:

1. Clone the repository into `/content`.
2. Install `requirements.txt`.
3. Optionally install `aria2` with `apt-get`.
4. Run `python scripts/download_data.py`.
5. Run `python scripts/run_pipeline.py`.
6. Run `python scripts/build_report_results.py`.
7. Run `python scripts/build_diagnostics.py`.
8. Optionally run `python -m pytest -q`.

If automatic download is unavailable, upload `kc_house_data.csv` manually and point `DATA_DIR` at the upload location. The project does not require text, image, remote-sensing, or external API inputs.

## Docs

Reporting references:

- `docs/reporting/figure_inventory.md`
- `docs/reporting/table_inventory.md`
- `docs/reporting/official_metrics.md`

Future architecture notes and retired comparison material:

- `docs/roadmap/`
- `archive/legacy_validation/`

## Repository Hygiene

- raw data remains excluded from version control
- generated outputs remain ignored except for intentional small summary artifacts
- line endings are normalized via `.gitattributes`
- GitHub workflows and templates are included under `.github/`

## Limitations

- The implemented system remains student-scale and tabular.
- Pricing anomaly outputs must not be described as strict asking-price mispricing.
- The implementation is dataset-aligned, not multimodal, listing-aware, or causal.
- The official system uses one final validated valuation model rather than a live model-zoo workflow.

## Convenience Commands

```bash
make install
make download
make run
make report-results
make diagnostics
make market-representation
make uncertainty-calibration
make notebook
make test
make smoke
make clean
```
