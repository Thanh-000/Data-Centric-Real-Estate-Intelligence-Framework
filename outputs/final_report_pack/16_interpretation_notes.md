# Slice Interpretation Notes

## Official System
- The official valuation model is `xgboost` with validation/test RMSE 111003.71 / 118687.65.
- Highest segment MAE remains `segment_1` at 114671.59; lowest segment MAE is `segment_2` at 35243.26.
- Weakest price band by MAE is `Q5` at 144298.74.
- Weakest price-band coverage appears in `Q5` at 0.8536.
- Lowest segment coverage appears in `segment_1` at 0.9213.
- Highest over-valued anomaly rate by segment is `segment_1` at 0.044; by price band it is `Q5` at 0.103.

These diagnostics support sale-price Pricing Anomaly Detection and should be read as valuation-gap evidence for realized transactions.