# Reference Metrics

This repository is a runnable framework. Reference metrics are generated locally rather than committed as fixed final results.

After code changes to the split policy, segmentation policy, feature set, or uncertainty policy, old reference values are no longer authoritative. Reproduce the current values with:

```bash
python scripts/quickstart.py --install
```

Primary generated metric files:

- `outputs/tables/valuation_metrics.csv`
- `outputs/tables/model_baseline_comparison.csv`
- `outputs/tables/xgboost_selection_grid.csv`
- `outputs/tables/test_error_by_price_band.csv`
- `outputs/tables/test_interval_coverage_by_price_band.csv`
- `outputs/tables/local_conformal_by_price_band.csv`
- `outputs/tables/local_conformal_by_segment.csv`
- `outputs/tables/interval_width_predicted_price_band.csv`
- `outputs/tables/interval_width_segment_label.csv`
- `outputs/tables/anomaly_threshold_sensitivity.csv`
- `outputs/tables/synthetic_anomaly_recall.csv`
- `outputs/reports/local_conformal_calibration_summary.json`
- `outputs/reports/uncertainty_metrics.json`
- `outputs/figures/shap_summary.png`
- `outputs/figures/residual_variance_by_predicted_bin.png`
- `outputs/tables/notebook_local_shap_contributions.csv`

Current validated local run:

- Cleaned rows: `21,597`
- Market segmentation: `zipcode_market`, `53` segments
- Held-out test R2: `0.900`
- Held-out test MAPE: `12.11%`
- Held-out empirical coverage: `96.1%`
- Held-out observed-Q5 empirical coverage: `90.4%`
- Held-out observed-Q5 average interval width: `$847,102`
- Full-portfolio model-flagged cases: `628 / 21,597` (`2.91%`)
- Full-portfolio over-valued candidates: `431`
- Full-portfolio under-valued candidates: `197`
- Withheld `insufficient_history` cases in the final table: `0`
- Synthetic shock recall: `27.5%` at +/-30%, `50.0%` at +/-40%, and `68.0%` at +/-50%.

The previous validated run used KMeans segmentation and an older abstention policy. It should be treated as a legacy baseline, not as the expected output for the current framework.

Coverage values are empirical diagnostics under the current chronological, localized, upper-tail-adjusted protocol. They are not theoretical guarantees of standard split conformal prediction.
