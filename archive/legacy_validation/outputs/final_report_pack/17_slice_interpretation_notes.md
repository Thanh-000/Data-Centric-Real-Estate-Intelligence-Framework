# Slice Interpretation Notes

## Frozen Baseline
- Official frozen model remains `random_forest` with validation/test RMSE 129084.55 / 147002.45.
- Highest segment MAE remains `segment_5` at 145605.64; lowest segment MAE is `segment_4` at 34867.83.
- Weakest price band by MAE is `Q5` at 167534.96.
- Lowest segment coverage appears in `segment_5` at 0.7261.
- Highest over-valued anomaly rate by segment is `segment_0` at 0.139; by price band it is `Q5` at 0.226.

## Candidate v1.6 Track
- Candidate selected model is `xgboost` with validation/test RMSE 110852.91 / 121629.66.
- The hardest candidate price band remains `Q5` at MAE 144661.67, with weakest coverage in `Q5` at 0.6564.

These diagnostics support sale-price Pricing Anomaly Detection and should not be interpreted as strict asking-price mispricing estimates.