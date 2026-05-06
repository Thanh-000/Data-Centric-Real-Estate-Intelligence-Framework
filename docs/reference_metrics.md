# Reference Metrics

These metrics document the validated reference run for the runnable DC-REIF framework. They are provided for comparison after users regenerate outputs locally.

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
