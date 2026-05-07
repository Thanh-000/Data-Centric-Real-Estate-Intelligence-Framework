# Model Layer

This repository does not commit a pre-trained model artifact. It is distributed as a runnable framework, so users train and verify the model locally by running the pipeline.

## Where the Model Code Lives

- `src/dc_reif/valuation.py`: defines the valuation model workflow, model search space, XGBoost estimator setup, out-of-fold prediction logic, and final selected model fitting.
- `src/dc_reif/valuation_core/`: exposes the valuation API used by the pipeline.
- `src/dc_reif/pipeline.py`: calls `train_and_select_model(...)`, saves valuation metrics, creates feature importance outputs, and writes fitted runtime artifacts.

## Official Valuation Core

The official valuation core is XGBoost via `XGBRegressor`. The model is selected and fitted during the pipeline run rather than loaded from a committed binary.

The validated workflow includes:

- chronological train / validation / test splitting
- default King County date cutoffs: train through 2014-12-31, validation through 2015-03-31, test after 2015-03-31
- validation-driven XGBoost configuration selection
- log-target XGBoost candidates for heteroscedastic sale-price behavior
- median, linear regression, and random forest baseline comparison
- leakage-safe preprocessing through a fitted sklearn pipeline
- out-of-fold fair-value estimates for training-era anomaly analysis
- forward holdout predictions for test-era evaluation
- runtime feature importance and SHAP summary outputs

## Runtime Model Artifacts

After running:

```bash
python scripts/run_pipeline.py
```

or:

```bash
python scripts/quickstart.py --install
```

the fitted artifacts are generated under:

- `data/artifacts/xgboost_pipeline.joblib`
- `data/artifacts/submarket_clustering.joblib`
- `outputs/tables/valuation_metrics.csv`
- `outputs/tables/model_baseline_comparison.csv`
- `outputs/tables/xgboost_selection_grid.csv`
- `outputs/reports/xgboost_selection_summary.json`
- `outputs/figures/feature_importance.png`
- `outputs/figures/shap_summary.png`
- `outputs/tables/notebook_local_shap_contributions.csv` when the notebook explainability section is run
These files are intentionally ignored by Git. They can be regenerated from source, configuration, and the downloaded dataset.

## Why Model Artifacts Are Not Committed

The project is meant to be rerunnable and inspectable. Keeping fitted binaries out of version control avoids stale artifacts, reduces repository size, and forces validation to happen through the reproducible pipeline.
