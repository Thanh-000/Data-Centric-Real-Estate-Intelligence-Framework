# Trust Summary

## Intended Use

This system is a model-assisted triage tool for realized sale-price review. It flags candidates for human review; it is not an automated appraisal engine or a final pricing authority.

## Model Performance

- Test R2: 0.900
- Test MAPE: 12.11%
- Test RMSE: $120,357
- Test MAE: $67,590

## Uncertainty

Coverage values are empirical diagnostics under the current chronological, localized, upper-tail-adjusted protocol; they are not theoretical guarantees of standard split conformal prediction.

- Global empirical coverage: 96.1%
- High-price Q5 empirical coverage: 90.4%
- Average interval width: $429,417
- High-price Q5 average interval width: $847,102

### Interval Method Comparison

- Selected decision-layer interval method: `conformal`
- conformal: coverage 96.1%, Q5 coverage 90.4%, average width $429,417, 30% synthetic recall 27.5%
- quantile_xgb: coverage 76.7%, Q5 coverage 64.3%, average width $236,536, 30% synthetic recall 84.0%
- Quantile XGBoost was not selected because overall coverage below 90%; Q5 coverage below 88%.

## Decision Layer

- Model-flagged cases: 628
- Potentially over-valued: 431
- Potentially under-valued: 197
- Withheld insufficient-history cases: 0
- Within model range: 20,969

## Known Limitations

- The data is the static King County 2014-2015 transaction dataset.
- High-price properties require wider intervals, so flags are less sharp in that segment.
- Feature importance and SHAP describe model behavior, not causal proof.
- All model-flagged cases should be reviewed with local market context before business use.