# Slice Interpretation Notes

## Official System
- The official valuation model is `xgboost` with validation/test RMSE 110852.91 / 121629.66.
- Highest segment MAE remains `segment_4` at 114710.05; lowest segment MAE is `segment_0` at 34226.69.
- Weakest price band by MAE is `Q5` at 144661.67.
- Lowest segment coverage appears in `segment_4` at 0.7508.
- Highest over-valued anomaly rate by segment is `segment_4` at 0.132; by price band it is `Q5` at 0.209.

These diagnostics support sale-price Pricing Anomaly Detection and should not be interpreted as strict asking-price mispricing estimates.