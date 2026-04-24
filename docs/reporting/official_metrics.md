# Official Metrics

The following results define the single official DC-REIF system for the current dataset-aligned implementation.

- selected model: `xgboost`
- validation RMSE: `110852.91`
- test RMSE: `121629.66`
- validation MAE: `65525.94`
- test MAE: `70271.43`
- validation R2: `0.8963`
- test R2: `0.8957`
- segment count: `6`
- silhouette score: `0.1839`
- davies_bouldin_index: `1.6291`
- interval method: `conformal_prediction_residual_quantile`
- interval coverage: `0.8831`
- average interval width: `280938.25`
- conformal q-hat: `140469.12`
- anomaly counts:
  - within expected range: `16630`
  - potentially over-valued: `1125`
  - potentially under-valued: `781`
  - insufficient history: `3061`

Legacy baseline-versus-candidate comparison material is archived under `archive/legacy_validation/`.
