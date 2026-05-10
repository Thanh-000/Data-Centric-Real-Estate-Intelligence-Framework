# DC-REIF: King County Real Estate Intelligence Framework

## Project Overview

This repository contains a runnable DC-REIF framework for the King County House Sales dataset. The implementation is deliberately narrow and data-centric: one valuation core, one market-context workflow, one uncertainty layer, and one decision-support output.

The system operates on realized **sale-price** data only. Its downstream decision product is **Pricing Anomaly Detection / Valuation Gap Analysis** for observed sale transactions.

## Final System Architecture

The active workflow contains one public system only:

- **Data governance and integrity:** automated download, checksum verification, schema validation, deterministic cleaning, and quality flags
- **Feature policy:** leakage-safe structural, temporal, renovation, historical-spatial, and geospatial-context features derived from the King County dataset only
- **Valuation core:** `XGBoost`
- **Contextual market grouping:** zipcode-based submarket segments, used as a pragmatic real-estate market context
- **Uncertainty layer:** conformal-inspired residual quantile intervals by predicted price decile and segment, reported as empirical coverage diagnostics rather than theoretical split-conformal guarantees
- **Decision support:** Pricing Anomaly Detection on sale-price data with sensitivity tables, synthetic anomaly recall checks, and fallback scoring for low-support rows

## Reference Results

This repository is distributed as a runnable framework. Generated datasets, model artifacts, figures, tables, reports, and final narrative materials are not committed; users reproduce technical outputs by running the quickstart or manual workflow. The team assembles final narrative materials separately at the end.

Reference metrics from the validated run are stored in:

- `docs/reference_metrics.md`
- `docs/model.md`
- `docs/product_limitations.md`
- `docs/product_persona.md`
- `docs/monitoring_plan.md`

Reference values change when the modeling policy changes. Re-run `python scripts/quickstart.py --install` to regenerate the current metrics under `outputs/tables/` and `outputs/reports/`.

Current validated local run:

- Cleaned rows: `21,597`
- Market segmentation: `zipcode_market`, `53` segments
- Held-out test R2: `0.900`
- Held-out test MAPE: `12.11%`
- Held-out empirical coverage: `96.1%`
- Held-out observed-Q5 coverage: `90.4%`
- Held-out observed-Q5 average interval width: `$847,102`
- Full-portfolio model-flagged cases: `628 / 21,597` (`2.91%`)
- Full-portfolio over-valued candidates: `431`
- Full-portfolio under-valued candidates: `197`
- Withheld `insufficient_history` cases in the current final table: `0`

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

For a first local verification run, use the one-command quickstart. It downloads the public King County dataset, runs the pipeline, and writes product analytics tables for abstention and slice-level review.

```bash
python scripts/quickstart.py --install
```

Equivalent `make` command:

```bash
make quickstart
```

To include the test suite in the same run:

```bash
python scripts/quickstart.py --install --with-tests
```

Manual dataset download:

```bash
python scripts/download_data.py
```

Manual full workflow:

```bash
python scripts/run_pipeline.py
python scripts/analyze_abstention.py
python scripts/evaluate_slices.py
python scripts/evaluate_synthetic_anomalies.py
python -m pytest -q
```

Equivalent `make` commands:

```bash
make download
make run
make analyze-abstention
make evaluate-slices
make evaluate-synthetic
make test
```

Check run health after the pipeline:

```bash
python scripts/health_check.py
```

Equivalent `make` command:

```bash
make health
```

Open the dashboard after running the pipeline:

```bash
streamlit run app/streamlit_app.py
```

Equivalent `make` command:

```bash
make dashboard
```

## Colab Setup

The repository is Colab-compatible by design.

Recommended Colab sequence:

1. Commit and push local dashboard changes before opening Colab. The Colab setup cell clones GitHub, not your local working tree.
2. Open `notebooks/01_dc_reif_king_county.ipynb` from the repository, or upload/open the notebook directly in Colab.
3. Run the notebook from top to bottom. If the notebook is launched standalone in Colab, the setup cell clones this repository automatically.
4. The setup cell installs `requirements.txt`, installs `aria2` in Colab when available, and makes the package importable. It does not mount Google Drive.
5. The data cell downloads the public King County dataset through the project downloader, preferring `aria2`.
6. Optional dashboard: paste a session-only token into `USER_NGROK_AUTHTOKEN` in the final notebook cell, or set `NGROK_AUTHTOKEN` as a Colab secret/environment variable. If neither is set, the cell prompts for the token at runtime with hidden input. Use the printed ngrok URL, not `localhost:8501`.
7. Optionally run `python -m pytest -q` after the notebook or script workflow.

The final Colab dashboard cell refreshes the GitHub checkout if the dashboard source looks stale, checks that the current dashboard source is present, creates missing dashboard outputs by running quickstart, waits for Streamlit to become healthy, and prints the Streamlit log path if startup fails.

If automatic download is unavailable, place `kc_house_data.csv` in a reachable data directory and point `DATA_DIR` to that location.

## Repository Structure

```text
repo-root/
|-- .github/
|-- app/
|-- configs/
|-- data/
|-- docs/
|-- notebooks/
|-- outputs/
|-- scripts/
|-- src/dc_reif/
`-- tests/
```

Primary entrypoints:

- `python scripts/download_data.py`
- `python scripts/run_pipeline.py`
- `python scripts/analyze_abstention.py`
- `python scripts/evaluate_slices.py`
- `streamlit run app/streamlit_app.py`
- `notebooks/01_dc_reif_king_county.ipynb`

## Model Layer

The model is code-defined and trained at runtime, not committed as a binary artifact. The official valuation core is XGBoost, implemented in `src/dc_reif/valuation.py` and called by `src/dc_reif/pipeline.py`.

After running the workflow, fitted artifacts such as `data/artifacts/xgboost_pipeline.joblib` are generated locally and ignored by Git. See `docs/model.md` for the model layout and artifact policy.

## Static Contracts vs Runtime Data

`configs/data_contracts/` contains static dataset contracts: the default source URL, checksum, required schema, and leakage policy. These files are committed.

`data/` is a runtime workspace. It only keeps `.gitkeep` files in a fresh checkout. Downloaded raw data, processed feature tables, manifests, and model artifacts are generated when users run the workflow and remain ignored by Git.

## Generated Technical Outputs

The following files are generated after running `python scripts/quickstart.py --install` or the manual workflow. They are intentionally not committed in the framework checkout.

Core pipeline outputs:

- `data/raw/kc_house_data.csv`
- `data/processed/kc_house_data_clean.csv`
- `data/processed/kc_house_features.csv`
- `data/artifacts/*.joblib`
- `outputs/tables/valuation_metrics.csv`
- `outputs/tables/model_baseline_comparison.csv`
- `outputs/tables/xgboost_selection_grid.csv`
- `outputs/tables/cluster_profiles.csv`
- `outputs/tables/feature_importance.csv`
- `outputs/tables/notebook_local_shap_contributions.csv`
- `outputs/tables/property_intelligence_table.csv`
- `outputs/tables/model_flagged_cases.csv`
- `outputs/tables/anomaly_threshold_sensitivity.csv`
- `outputs/tables/synthetic_anomaly_recall.csv`
- `outputs/tables/interval_width_*.csv`
- `outputs/tables/test_error_by_price_band.csv`
- `outputs/tables/test_interval_coverage_by_price_band.csv`
- `outputs/tables/abstention_*.csv`
- `outputs/tables/slice_metrics_*.csv`
- `outputs/reports/pipeline_summary.md`
- `outputs/reports/trust_summary.md`
- `outputs/reports/*_summary.json`
- `outputs/figures/*.png`
- `outputs/figures/shap_summary.png`
- `outputs/figures/residual_variance_by_predicted_bin.png`

## Methodological Safeguards

- No target-derived variables such as `price_per_sqft` are used in the predictive branch.
- The default split is time-based: train through 2014-12-31, validation through 2015-03-31, and test after 2015-03-31 when those periods are present.
- All preprocessing, segment mappings, and model parameters are fit on training data or training folds only.
- Out-of-fold fair values are used for calibration and evaluation; fallback scoring keeps the runtime property table reviewable for low-support rows.
- Coverage values are empirical diagnostics under the current chronological, localized, upper-tail-adjusted protocol. They are not theoretical guarantees of standard split conformal prediction.
- The anomaly layer is framed for sale-price valuation gaps, not for listing-side decisions.
- The data download workflow preserves checksum verification and graceful fallback behavior.

Detailed methodology is summarized in `docs/methodology.md`.

## Main Review Artifacts

For review or report writing, start with these generated files:

- `outputs/reports/pipeline_summary.md`
- `outputs/reports/trust_summary.md`
- `outputs/tables/property_intelligence_table.csv`
- `outputs/tables/model_flagged_cases.csv`
- `outputs/tables/valuation_metrics.csv`
- `outputs/figures/shap_summary.png`

`trust_summary.md` is the compact answer to "how far should we trust this run?" It reports model performance, conformal coverage, high-price coverage, interval width, model-flagged counts, and known limitations.

## What This System Can and Cannot Do

Can:

- reproduce an end-to-end sale-price review workflow
- flag unusual realized sale prices for human review
- quantify model uncertainty with empirical residual-quantile intervals
- explain model behavior with feature importance and SHAP outputs
- support triage through a Streamlit dashboard

Cannot:

- replace licensed appraisal or local market judgment
- prove causal drivers of price
- guarantee current-market validity without retraining on current data
- make lending, investment, or listing decisions automatically

## Limitations

- The repository is intentionally scoped to the King County House Sales dataset.
- The system is tabular and CPU-friendly.
- Zipcode segments provide pragmatic market context but are not a substitute for production school-district, parcel, or neighborhood-boundary data.
- The uncertainty layer is practical and lightweight; interval width is audited by price band and segment to surface heteroscedastic behavior.
- The current interval policy is not a direct quantile-regression model; direct heteroscedastic models remain future work.
- The decision layer supports sale-price valuation-gap analysis; it should not be treated as a listing-price policy tool.

## Future Work

The current repository is a strong analytical framework, but not yet a complete stakeholder-facing DA product. Product gaps and the prioritized roadmap are documented in `docs/product_limitations.md`.
