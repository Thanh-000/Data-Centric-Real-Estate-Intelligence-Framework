# DC-REIF: Final Validated King County Real Estate Intelligence System

## Project Overview

This repository contains the final validated DC-REIF system for the King County House Sales dataset. The implementation is deliberately narrow, data-centric, and trust-aware: it focuses on one valuation core, one contextual market-grouping workflow, one uncertainty layer, and one report-ready decision-support output.

The system operates on realized **sale-price** data only. Its downstream decision product is **Pricing Anomaly Detection / Valuation Gap Analysis** for observed sale transactions.

## Final System Architecture

The active workflow contains one public system only:

- **Data governance and integrity:** automated download, checksum verification, schema validation, deterministic cleaning, and quality flags
- **Feature policy:** leakage-safe structural, temporal, renovation, and geospatial-context features derived from the King County dataset only
- **Valuation core:** `XGBoost`
- **Contextual market grouping:** `KMeans` used as market-context encoding, not as a definitive market-boundary estimate
- **Uncertainty layer:** localized conformal prediction residual quantile intervals
- **Decision support:** report-ready Pricing Anomaly Detection on sale-price data with abstention for insufficient history

## Final Official Results

The official metrics are stored in:

- `docs/reporting/official_metrics.md`
- `outputs/reports/final_results_master.json`
- `outputs/reports/final_results_master.md`
- `outputs/reports/final_results_master.csv`

Current official values:

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
- interval method: `conformal_prediction_residual_quantile_localized`
- interval coverage: `0.9330`
- average interval width: `372280.01`
- conformal q-hat: `140076.25`
- anomaly counts:
  - within expected range: `17555`
  - potentially over-valued: `665`
  - potentially under-valued: `316`
  - insufficient history: `3061`

## Local Setup

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Or:

```bash
make install
```

Download the dataset:

```bash
python scripts/download_data.py
```

Run the full workflow:

```bash
python scripts/run_pipeline.py
python scripts/build_report_results.py
python scripts/build_diagnostics.py
python -m pytest -q
```

Equivalent `make` commands:

```bash
make download
make run
make report-results
make diagnostics
make test
```

## Colab Setup

The repository is Colab-compatible by design.

Recommended Colab sequence:

1. Clone the repository into `/content`.
2. Install `requirements.txt`.
3. Optionally install `aria2`.
4. Run `python scripts/download_data.py`.
5. Run `python scripts/run_pipeline.py`.
6. Run `python scripts/build_report_results.py`.
7. Run `python scripts/build_diagnostics.py`.
8. Optionally run `python -m pytest -q`.

If automatic download is unavailable, place `kc_house_data.csv` in a reachable data directory and point `DATA_DIR` to that location.

## Repository Structure

```text
repo-root/
|-- .github/
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

Primary entrypoints:

- `python scripts/download_data.py`
- `python scripts/run_pipeline.py`
- `python scripts/build_report_results.py`
- `python scripts/build_diagnostics.py`
- `notebooks/01_dc_reif_king_county.ipynb`

## Official Result Artifacts

Canonical summary artifacts:

- `outputs/reports/final_results_master.json`
- `outputs/reports/final_results_master.md`
- `outputs/reports/final_results_master.csv`

Core report pack artifacts:

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
- `outputs/final_report_pack/18_selected_xgboost_parameters.md`
- `outputs/final_report_pack/19_segmentation_selection_summary.md`
- `outputs/final_report_pack/20_local_conformal_summary.md`

## Methodological Safeguards

- No target-derived variables such as `price_per_sqft` are used in the predictive branch.
- All preprocessing, clustering transforms, and learned mappings are fit on training data or training folds only.
- Out-of-fold fair values are used for anomaly analysis.
- The anomaly layer is framed for sale-price valuation gaps, not for listing-side decisions.
- The data download workflow preserves checksum verification and graceful fallback behavior.

Detailed safeguards are summarized in `docs/methodology/safeguards.md`.

## Limitations

- The repository is intentionally scoped to the King County House Sales dataset.
- The system is tabular and CPU-friendly.
- KMeans provides contextual market grouping rather than a definitive market-boundary estimate.
- The uncertainty layer is practical and lightweight rather than fully heteroscedastic.
- The decision layer supports sale-price valuation-gap analysis; it should not be treated as a listing-price policy tool.

## Future Work

Potential future work, not implemented here:

- richer spatial diagnostics within the same dataset scope
- deeper calibration analysis for difficult slices such as the upper price band
- more extensive decision-support reporting for lecturer-facing presentation materials
