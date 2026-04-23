# Repository Instructions

- Preserve the DC-REIF framing: this project is data-centric, single-core, and trust-aware.
- Never use target-derived variables such as `price_per_sqft` in the predictive branch.
- Keep preprocessing leakage-safe: all imputers, encoders, scalers, clustering transforms, and learned mappings must be fit on training data or training folds only.
- Use out-of-fold fair value estimates for anomaly analysis; never use in-sample fitted values for pricing anomaly detection.
- Keep `RandomForestRegressor` as the default main valuation model unless the user explicitly changes the modeling scope.
- Maintain the project as GitHub-ready and Colab-compatible: no hard-coded local paths and no committed raw dataset.
- Preserve the automated download workflow with `aria2c` preference, checksum verification, and graceful fallback behavior.
- Keep anomaly language aligned to sale-price data: use `Pricing Anomaly Detection` or `Valuation Gap Analysis`, not strict asking-price mispricing.
- Prefer small, modular, report-friendly code and lightweight CPU-friendly dependencies.

