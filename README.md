# DC-REIF: Final Validated King County Real Estate Intelligence System

## Project Overview

This repository contains the final validated DC-REIF system framework for the King County House Sales dataset. The implementation is deliberately narrow, data-centric, and trust-aware: it focuses on one valuation core, one contextual market-grouping workflow, one uncertainty layer, and one decision-support output.

The system operates on realized **sale-price** data only. Its downstream decision product is **Pricing Anomaly Detection / Valuation Gap Analysis** for observed sale transactions.

## Final System Architecture

The active workflow contains one public system only:

- **Data governance and integrity:** automated download, checksum verification, schema validation, deterministic cleaning, and quality flags
- **Feature policy:** leakage-safe structural, temporal, renovation, and geospatial-context features derived from the King County dataset only
- **Valuation core:** `XGBoost`
- **Contextual market grouping:** `KMeans` used as market-context encoding, not as a definitive market-boundary estimate
- **Uncertainty layer:** localized conformal prediction residual quantile intervals
- **Decision support:** Pricing Anomaly Detection on sale-price data with abstention for insufficient history

## Reference Results

This repository is distributed as a runnable framework. Generated datasets, model artifacts, figures, tables, reports, and final narrative materials are not committed; users reproduce technical outputs by running the quickstart or manual workflow. The team assembles final narrative materials separately at the end.

Reference metrics from the validated run are stored in:

- `docs/reference_metrics.md`
- `docs/model.md`
- `docs/product_limitations.md`
- `docs/product_persona.md`
- `docs/monitoring_plan.md`

Expected reference values:

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

For a first local verification run, use the one-command quickstart. It downloads the public King County dataset and runs the pipeline:
It also writes product analytics tables for abstention and slice-level review.

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
python -m pytest -q
```

Equivalent `make` commands:

```bash
make download
make run
make analyze-abstention
make evaluate-slices
make test
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

1. Open `notebooks/01_dc_reif_king_county.ipynb` from the repository, or upload/open the notebook directly in Colab.
2. Run the notebook from top to bottom. If the notebook is launched standalone in Colab, the setup cell clones this repository automatically.
3. The setup cell installs `requirements.txt`, installs `aria2` in Colab when available, and makes the package importable. It does not mount Google Drive.
4. The data cell downloads the public King County dataset through the project downloader, preferring `aria2`.
5. Optional: set `NGROK_AUTHTOKEN` as a Colab secret or environment variable, then run the final notebook cell to expose the Streamlit dashboard through ngrok.
6. Optionally run `python -m pytest -q` after the notebook or script workflow.

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
- `outputs/tables/cluster_profiles.csv`
- `outputs/tables/feature_importance.csv`
- `outputs/tables/property_intelligence_table.csv`
- `outputs/tables/abstention_*.csv`
- `outputs/tables/slice_metrics_*.csv`
- `outputs/reports/pipeline_summary.md`
- `outputs/reports/*_summary.json`
- `outputs/figures/*.png`

## Methodological Safeguards

- No target-derived variables such as `price_per_sqft` are used in the predictive branch.
- All preprocessing, clustering transforms, and learned mappings are fit on training data or training folds only.
- Out-of-fold fair values are used for anomaly analysis.
- The anomaly layer is framed for sale-price valuation gaps, not for listing-side decisions.
- The data download workflow preserves checksum verification and graceful fallback behavior.

Detailed methodology is summarized in `docs/methodology.md`.

## Limitations

- The repository is intentionally scoped to the King County House Sales dataset.
- The system is tabular and CPU-friendly.
- KMeans provides contextual market grouping rather than a definitive market-boundary estimate.
- The uncertainty layer is practical and lightweight rather than fully heteroscedastic.
- The decision layer supports sale-price valuation-gap analysis; it should not be treated as a listing-price policy tool.

## Future Work

The current repository is a strong analytical framework, but not yet a complete stakeholder-facing DA product. Product gaps and the prioritized roadmap are documented in `docs/product_limitations.md`.
