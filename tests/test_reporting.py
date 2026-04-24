from dc_reif.reporting import markdown_summary_block


def test_reporting_summary_block_mentions_core_sections():
    summary = {
        "core_valuation_metrics": {
            "selected_model": "xgboost",
            "validation_rmse": 1,
            "test_rmse": 1,
            "validation_mae": 1,
            "test_mae": 1,
            "validation_r2": 1,
            "test_r2": 1,
        },
        "segmentation_results": {
            "segment_count": 6,
            "silhouette_score": 0.1,
            "davies_bouldin_index": 1.0,
            "segment_summary_reference": "outputs/tables/cluster_profiles.csv",
        },
        "uncertainty_results": {
            "interval_method": "conformal_prediction_residual_quantile",
            "interval_coverage": 0.8,
            "average_interval_width": 10,
            "conformal_qhat": 5,
        },
        "pricing_anomaly_results": {
            "within_expected_range": 1,
            "potentially_over_valued": 1,
            "potentially_under_valued": 1,
            "insufficient_history": 1,
        },
        "explainability_results": {
            "top_features": ["grade"],
            "feature_importance_file": "outputs/tables/feature_importance.csv",
            "shap_summary_file": None,
        },
    }
    block = markdown_summary_block(summary)
    assert "Core Valuation Metrics" in block
    assert "Pricing Anomaly Results" in block
