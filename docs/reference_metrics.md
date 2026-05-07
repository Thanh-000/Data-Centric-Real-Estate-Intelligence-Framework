# Reference Metrics

This repository is a runnable framework. Reference metrics are generated locally rather than committed as fixed final results.

After code changes to the split policy, segmentation policy, feature set, or uncertainty policy, old reference values are no longer authoritative. Reproduce the current values with:

```bash
python scripts/quickstart.py --install
```

Primary generated metric files:

- `outputs/tables/valuation_metrics.csv`
- `outputs/tables/model_baseline_comparison.csv`
- `outputs/tables/test_error_by_price_band.csv`
- `outputs/tables/test_interval_coverage_by_price_band.csv`
- `outputs/tables/local_conformal_by_price_band.csv`
- `outputs/tables/local_conformal_by_segment.csv`
- `outputs/tables/interval_width_predicted_price_band.csv`
- `outputs/tables/interval_width_segment_label.csv`
- `outputs/tables/anomaly_threshold_sensitivity.csv`
- `outputs/reports/local_conformal_calibration_summary.json`
- `outputs/reports/uncertainty_metrics.json`

The previous validated run used KMeans segmentation and an older abstention policy. It should be treated as a legacy baseline, not as the expected output for the current framework.
