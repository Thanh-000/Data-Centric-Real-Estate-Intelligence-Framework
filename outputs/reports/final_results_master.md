# Final Results Master Summary

## Core Valuation Metrics
- Selected model: `xgboost`
- Validation RMSE: `111003.71`
- Test RMSE: `118687.65`
- Validation MAE: `65533.19`
- Test MAE: `69904.96`
- Validation R2: `0.896`
- Test R2: `0.9007`

## Segmentation Results
- Segment count: `3`
- Silhouette score: `0.1774`
- Davies-Bouldin index: `1.7775`
- Segment profiles: `outputs/tables/cluster_profiles.csv`

## Uncertainty Results
- Interval method: `conformal_prediction_residual_quantile_localized`
- Interval coverage: `0.933`
- Average interval width: `372280.01`
- Conformal q-hat: `140076.25`

## Pricing Anomaly Results
- Within expected range: `17555`
- Potentially over-valued: `665`
- Potentially under-valued: `316`
- Insufficient history: `3061`

## Explainability Results
- Top features: `location_grade_interaction, grade_living_interaction, segment_label_segment_1, waterfront, distance_to_seattle_core`
- Feature importance file: `outputs/tables/feature_importance.csv`
- SHAP summary file: `outputs/figures/shap_summary.png`