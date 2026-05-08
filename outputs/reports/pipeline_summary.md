# DC-REIF Pipeline Summary

- Selected valuation model: xgboost
- Target strategy: raw
- High-price sample weight: 1.00
- Market segmentation: zipcode_market (53 segments)
- Validation report: C:\Users\Admin\OneDrive\Desktop\New folder (4)\outputs\reports\data_quality_report.json
- Cleaned rows retained: 21597
- Local conformal global q-hat: 144381.56
- Selected interval method: conformal
- Quantile comparison rationale: kept conformal: overall coverage below 90%; Q5 coverage below 88%
- Test empirical coverage: 0.961
- Test Q5 empirical coverage: 0.904
- Test average interval width: 429416.55
- Potentially under-valued sales: 197
- Potentially over-valued sales: 431

This system performs Pricing Anomaly Detection on realized sale prices and should not be interpreted as a listing-price decision rule.

## Model Baseline Comparison

| Model | Test RMSE | Test MAE | Test MAPE | Test R2 |
|---|---:|---:|---:|---:|
| xgboost_selected | $120,357 | $67,590 | 12.11% | 0.900 |
| random_forest | $134,764 | $73,329 | 12.70% | 0.875 |
| linear_regression | $146,528 | $85,013 | 15.92% | 0.852 |
| median_baseline | $396,530 | $225,739 | 39.19% | -0.086 |