# Final Results Master Summary

## Core Valuation Metrics
- Selected model: `xgboost`
- Validation RMSE: `110852.91`
- Test RMSE: `121629.66`
- Validation MAE: `65525.94`
- Test MAE: `70271.43`
- Validation R2: `0.8963`
- Test R2: `0.8957`

## Segmentation Results
- Segment count: `6`
- Silhouette score: `0.1839`
- Davies-Bouldin index: `1.6291`
- Segment profiles: `outputs/tables/cluster_profiles.csv`

## Uncertainty Results
- Interval method: `conformal_prediction_residual_quantile`
- Interval coverage: `0.8831`
- Average interval width: `280938.25`
- Conformal q-hat: `140469.12`

## Pricing Anomaly Results
- Within expected range: `16630`
- Potentially over-valued: `1125`
- Potentially under-valued: `781`
- Insufficient history: `3061`

## Explainability Results
- Top features: `grade_living_interaction, location_grade_interaction, waterfront, distance_to_seattle_core, waterfront_view_score`
- Feature importance file: `outputs/tables/feature_importance.csv`
- SHAP summary file: `outputs/figures/shap_summary.png`