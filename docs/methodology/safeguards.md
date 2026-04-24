# Methodological Safeguards

The final validated DC-REIF system follows these methodological safeguards throughout the active workflow:

- No target-derived variables such as `price_per_sqft` are used in the predictive branch.
- All preprocessing, clustering transforms, imputers, encoders, scalers, and learned mappings are fit on training data or training folds only.
- Out-of-fold fair values are used for anomaly analysis; in-sample fitted values are not used for Pricing Anomaly Detection.
- The decision layer is framed for realized sale-price valuation gaps, not listing-price claims.
- Raw data is excluded from version control.
- Paths remain portable for local and Colab execution.
- The automated download workflow preserves checksum verification, `aria2c` preference when available, and graceful fallback behavior.
