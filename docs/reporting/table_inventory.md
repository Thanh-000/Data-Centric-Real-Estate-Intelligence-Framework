# Table Inventory

- `outputs/tables/valuation_metrics.csv`: official validation and holdout metrics for the single active valuation model
- `outputs/tables/xgboost_selection_grid.csv`: validation-driven XGBoost tuning summary for the official model
- `outputs/tables/cluster_profiles.csv`: KMeans contextual market-grouping summary
- `outputs/tables/segmentation_selection_grid.csv`: KMeans selection diagnostics across evaluated K values
- `outputs/tables/feature_importance.csv`: explainability ranking
- `outputs/tables/local_conformal_by_segment.csv`: localized conformal calibration summary by segment
- `outputs/tables/local_conformal_by_price_band.csv`: localized conformal calibration summary by predicted price band
- `outputs/tables/property_intelligence_table.csv`: master decision-product table
- `outputs/reports/final_results_master.csv`: flattened official results summary
- `outputs/final_report_pack/02_results_summary_table.csv`: report-ready metrics extract
- `outputs/final_report_pack/03_anomaly_summary_table.csv`: report-ready anomaly counts
- `outputs/final_report_pack/07_case_examples.csv`: curated case-study rows
