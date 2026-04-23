# Final Results Master Summary

## Core Valuation Metrics
- Selected model: `random_forest`
- Baseline model: `linear_regression`
- Validation RMSE: `129084.55`
- Test RMSE: `147002.45`
- Validation MAE: `70975.29`
- Test MAE: `79150.17`
- Validation R2: `0.8594`
- Test R2: `0.8476`

## Segmentation Results
- Segment count: `6`
- Silhouette score: `0.1784`
- Davies-Bouldin index: `1.4803`
- Segment profiles: `outputs/tables/cluster_profiles.csv`

## Uncertainty Results
- Interval method: `conformal_prediction_residual_quantile`
- Interval coverage: `0.8858`
- Average interval width: `309509.04`
- Conformal q-hat: `154754.52`

## Pricing Anomaly Results
- Within expected range: `16639`
- Potentially over-valued: `1158`
- Potentially under-valued: `739`
- Insufficient history: `3061`

## Explainability Results
- Top features: `grade, sqft_living, lat, long, sqft_living15`
- Feature importance file: `outputs/tables/feature_importance.csv`
- SHAP summary file: `outputs/figures/shap_summary.png`